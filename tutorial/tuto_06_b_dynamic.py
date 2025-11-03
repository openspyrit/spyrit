# %% [markdown]
"""
# Dynamic Scene Reconstruction Tutorial with SpyRIT

This tutorial demonstrates how to reconstruct dynamic scenes using single-pixel imaging
with the SpyRIT library. We'll cover:

1. **Setup**: Loading images and defining deformation fields
2. **Forward modeling**: Simulating measurements from dynamic scenes
3. **Measurement operators**: Different approaches for dynamic acquisition
4. **Dynamic matrix construction**: Building the system matrix for reconstruction
5. **Reconstruction**: Solving the inverse problem with regularization
6. **Evaluation**: Comparing static vs dynamic reconstruction quality

### Key concepts:
    - Dynamic forward operator H_dyn that accounts for motion during acquisition
    - Motion compensation in single-pixel imaging systems
    - Regularized reconstruction with finite differences
"""

# %% Import libraries
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import math
import time

from IPython.display import clear_output

from spyrit.misc.disp import torch2numpy, imagesc, contrib_map
from spyrit.misc.statistics import transform_gray_norm
import spyrit.core.torch as spytorch
import spyrit.misc.metrics as score

from spyrit.core.noise import Poisson
from spyrit.core.prep import Unsplit 
from spyrit.core.warp import AffineDeformationField



#%% Configuration and Setup
"""
Configuration for dynamic scene reconstruction:
- img_size: Full image resolution (pixels)
- n: Measurement pattern size (defines FOV)
- und: Undersampling factor (1 = no undersampling)
- M: Total number of measurements per frame
"""
img_size = 88  # Full image side's size in pixels
n = 64         # Measurement pattern side's size in pixels (Field of View)
und = 1        # Undersampling factor (1 = full sampling)

M = n ** 2 // und  # Number of (positive, negative) measurements

print(f"Image size: {img_size}x{img_size}")
print(f"Measurement FOV: {n}x{n}")
print(f"Measurements per frame: {M}")

# Dataset parameters
i = 0  # Image index (modify to change the image)
spyritPath = '../data/data_online/'
imgs_path = os.path.join(spyritPath, "spyrit/")

# Computation parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dtype = torch.float64          # High precision for reconstruction stability
simu_interp = 'bilinear'       # Interpolation for motion simulation
mode = 'bilinear'              # Interpolation for building forward operator

time_dim = 1                   # Time dimension index in tensors

# Derived parameters
meas_shape = (n, n)
img_shape = (img_size, img_size)
amp_max = (img_shape[0] - meas_shape[0]) // 2  # Border size for centering FOV 

# %% Load and prepare test image
"""
Load a test image for dynamic reconstruction demonstration.
The image is normalized to [0,1] range for consistent processing.
"""
# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=img_size)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

x, _ = next(iter(dataloader))
print(f"Loaded image shape: {x.shape}")

# Select and prepare image
x = x[i : i + 1, :, :, :].to(dtype=dtype, device=device)
x = x.detach().clone()
x = (x - x.min()) / (x.max() - x.min())  # Normalize to [0,1]

# Visualize the original image
x_plot = x.view(img_shape).cpu()
imagesc(x_plot, r"Original Image")


# %% Define motion model and deformation fields
"""
We simulate a pulsating motion using affine transformations (see tuto_06_warp.py).
Both forward and inverse deformation fields are needed for the tutorial:
- Forward field: for simulation & reconstruction with image warping
- Inverse field: for reconstruction with pattern warping
"""
with torch.no_grad():
    # Motion parameters
    a = 0.2          # Scaling amplitude
    T = 1000         # Period of motion cycle
    
    print(f"Motion parameters:")
    print(f"  Amplitude: {a:.2f}")
    print(f"  Period: {T} time units")

    def s(t):
        return 1 + a * math.sin(t * 2 * math.pi / T) 

    def f(t):
        return torch.tensor(
            [
                [1 / s(t), 0, 0],
                [0, s(t), 0],
                [0, 0, 1],
            ],
            dtype=dtype,
        )
    
    def f_inv(t):
            return torch.tensor(
                [
                    [s(t), 0, 0],
                    [0, 1 / s(t), 0],
                    [0, 0, 1],
                ],
                dtype=dtype,
            )

    # Create time vector for 2*M frames (positive + negative measurements)
    time_vector = torch.linspace(0, 2 * T, 2 * M)
    print(f"Time vector: {2*M} frames from 0 to {2*T}ms")
    
    # Create instances of affine deformation fields
    def_field = AffineDeformationField(f, time_vector, img_shape, dtype=dtype, device=device)
    def_field_inv = AffineDeformationField(f_inv, time_vector, img_shape, dtype=dtype, device=device)
    
    print(f"Created deformation fields with shape: {def_field.field.shape}")


    # %% Simulate dynamic image sequence
    """
    Apply the deformation field to create a dynamic image sequence.
    This simulates how the scene changes over time during acquisition.
    """
    x_motion = def_field(x, 0, 2 * M, mode=simu_interp)
    x_motion = x_motion.moveaxis(time_dim, 1)
    print(f"Dynamic sequence shape: {x_motion.shape}")
    print(f"Generated {x_motion.shape[1]} frames for acquisition simulation")
   
    # %% Visualize motion sequence
    """
    Display a few frames to visualize the motion pattern.
    This helps verify that the deformation is working as expected.
    """
    print("Visualizing motion sequence...")
    for frame in range(int(n / und ** 0.5)):
        plt.close()
        plt.imshow(x_motion[0, n * frame, 0, amp_max:img_size-amp_max, amp_max:img_size-amp_max].view(meas_shape).cpu().numpy(), cmap="gray")  # in X
        plt.suptitle("frame %d" % (n * frame), fontsize=16)
        plt.colorbar()
        plt.pause(0.01)
        clear_output(wait=True)


    # %% Create measurement patterns using Hadamard basis
    """
    Generate Hadamard measurement patterns for single-pixel imaging.
    The HadamSplit2d operator creates positive/negative pattern pairs.
    """
    from spyrit.core.meas import HadamSplit2d

    meas_op_stat = HadamSplit2d(h=n, M=M, order=None, dtype=dtype, device=device)
    H_split = meas_op_stat.A  # Measurement matrix (positive + negative patterns)
    
    print(f"Hadamard patterns shape: {H_split.shape}")
    print(f"  {H_split.shape[0]} patterns total ({M} positive + {M} negative)")

    # %% Configure noise model
    """
    Set up noise model for realistic acquisition simulation (see tuto_02_noise.py).
    Options: Identity (no noise) or Poisson noise model.
    """
    torch.manual_seed(100)  # For reproducible results

    alpha = 50  # Noise parameter (higher = less noise)

    # Choose noise model:
    noise_op = torch.nn.Identity()  # No noise for clear demonstration
    # noise_op = Poisson(alpha=alpha, g=1 / alpha)  # Uncomment for Poisson noise
    
    noise_type = "No noise" if isinstance(noise_op, torch.nn.Identity) else f"Poisson (α={alpha})"
    print(f"Noise model: {noise_type}")
  
  

    # %% Dynamic measurement simulation - Method comparison
    """
    We extend the different measurements operators from the static case (see tuto_01_*.py) to dynamic scenes.

    Compare different measurement operators for dynamic scenes:
    1. DynamicLinear: General purpose, flexible but slower
    2. DynamicLinearSplit: Optimized for split measurements  
    3. DynamicHadamSplit2d: Specialized for Hadamard patterns
    """

    # %% METHOD 1: DynamicLinear (General purpose)
    print("\n=== Testing DynamicLinear ===")
    from spyrit.core.meas import DynamicLinear

    meas_op = DynamicLinear(H=H_split, time_dim=time_dim, meas_dims=(-2, -1), 
                           img_shape=img_shape, meas_shape=meas_shape, 
                           noise_model=noise_op, dtype=dtype, device=device)
    
    t1 = time.time()
    y1 = meas_op(x_motion)
    t2 = time.time()

    print(f"  Computation time: {t2 - t1:.3f}s")
    print(f"  Output shape: {y1.shape}")


    # %% METHOD 2: DynamicLinearSplit (Optimized for split measurements)
    print("\n=== Testing DynamicLinearSplit ===")
    from spyrit.core.meas import DynamicLinearSplit

    # Create difference patterns (positive - negative)
    H = H_split[::2] - H_split[1::2]
    print(f"  Split patterns shape: {H.shape}")

    meas_op = DynamicLinearSplit(H=H, time_dim=time_dim, meas_dims=(-2, -1), 
                                img_shape=img_shape, meas_shape=meas_shape, 
                                noise_model=noise_op, dtype=dtype, device=device)
    
    t1 = time.time()
    y1 = meas_op(x_motion)
    t2 = time.time()

    print(f"  Computation time: {t2 - t1:.3f}s")
    print(f"  Output shape: {y1.shape}")


    # %% METHOD 3: DynamicHadamSplit2d (Specialized for Hadamard, fastest)
    print("\n=== Testing DynamicHadamSplit2d (Recommended) ===")
    from spyrit.core.meas import DynamicHadamSplit2d

    meas_op = DynamicHadamSplit2d(time_dim=time_dim, h=n, M=M, order=None,
                                 fast=True, reshape_output=False, img_shape=img_shape,
                                 noise_model=noise_op, white_acq=None,
                                 dtype=dtype, device=device)
    
    t1 = time.time()
    y1 = meas_op(x_motion)
    t2 = time.time()

    print(f"  Computation time: {t2 - t1:.3f}s")
    print(f"  Output shape: {y1.shape}")
    print(f"  This operator will be used for reconstruction.")


    # %% Process measurements for reconstruction
    """
    The Unsplit operation combines positive/negative measurements using the differential strategy.
    """
    prep_op = Unsplit()
    y2 = prep_op(y1)

    print(f"\nMeasurement processing:")
    print(f"  Raw measurements shape: {y1.shape}")
    print(f"  Processed measurements shape: {y2.shape}")

    
    # %% Static reconstruction baseline
    """
    Compute a static reconstruction for comparison.
    This ignores motion and treats all measurements as from a static scene.
    """
    print(f"\n=== Static Reconstruction (Baseline) ===")
    x_stat = meas_op_stat.fast_pinv(y2)
    print(f"Static reconstruction shape: {x_stat.shape}")
    
    # Quick quality check for static reconstruction
    x_ref_fov = x[0, 0, amp_max:meas_shape[0] + amp_max, amp_max:meas_shape[1] + amp_max]
    static_psnr = score.psnr(torch2numpy(x_stat), torch2numpy(x_ref_fov))
    print(f"Static reconstruction PSNR: {static_psnr:.2f} dB")

    plt.figure(figsize=(6, 6))
    plt.imshow(torch2numpy(x_stat).squeeze(), cmap='gray')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Static Reconstruction \n (strong artifacts due to motion)")
    plt.axis("off")
    plt.show()

    # %% Build dynamic system matrix
    """
    Construct the dynamic forward operator H_dyn that accounts for motion.
    
    Two approaches:
    1. warping=False: Warp the image. Need to pass the forward deformation field.
    2. warping=True: Warp the patterns. Need to pass the inverse deformation field.
    
    The matrix H_dyn maps the reference image to measurements accounting for motion.
    """
    print(f"\n=== Dynamic System Matrix Construction ===")
    
    # Reconstruction configuration
    in_X = False       # Reconstruct in extended FOV (False) or SPC FOV only (True)  
    warping = False    # Use pattern or image warping approach
 
    print(f"Configuration:")
    print(f"  Reconstruction space: {r'SPC FOV $X$' if in_X else r'Extended FOV $X_{ext}$'}")
    print(f"  Warping mode: {'Pattern warping' if warping else 'Image warping'}")

    # Build the dynamic system matrix
    if warping:
        print("Building H_dyn using pattern warping (inverse deformation)...")
        meas_op.build_H_dyn(def_field_inv, warping=warping, mode=mode, verbose=False)
    else:
        print("Building H_dyn using image warping (forward deformation)...")
        meas_op.build_H_dyn(def_field, warping=warping, mode=mode, verbose=False)

    H_dyn_diff = meas_op.H_dyn_diff
    print(f"Dynamic system matrix (differential strategy AFTER motion compensation) shape: {H_dyn_diff.shape}")

    # Adapt system matrix based on reconstruction space
    if in_X:
        print("Extracting SPC FOV region from full system matrix by dropping columns...")
        reco_shape = meas_shape
        H_dyn_diff = H_dyn_diff.reshape((H_dyn_diff.shape[0], img_size, img_size))
        H_dyn_diff = H_dyn_diff[:, amp_max:-amp_max, amp_max:-amp_max]
        H_dyn_diff = H_dyn_diff.reshape((H_dyn_diff.shape[0], n ** 2))
    else:
        reco_shape = img_shape
    
    print(f"Final system matrix shape: {H_dyn_diff.shape}")
    print(f"Reconstruction will be in {reco_shape} space")

        
    # %% Verify forward model accuracy
    """
    Test the dynamic forward model by computing the residual of the forward model.
    Without measurement noise and using dtype=float64, the residual should be very small (\approx 1e-12). 
    """
    H_dyn_x = meas_op.forward_H_dyn(x)
    residual_norm = torch.norm(y1 - H_dyn_x).item()
    relative_error = residual_norm / torch.norm(y1).item()
    
    print(f"\n=== Forward Model Verification ===")
    print(f"  Predicted measurements shape: {H_dyn_x.shape}")
    print(f"  Residual norm: {residual_norm:.2e}")
    print(f"  Relative error: {relative_error:.2e}")

    # Visualize residual pattern
    plt.figure(figsize=(10, 7))
    residual_2d = abs(y1 - H_dyn_x).squeeze().cpu().numpy().reshape((2 * n, n))
    plt.imshow(residual_2d, cmap='Spectral')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f'Forward Model Residual |y - H_dyn·x| \n Max: {residual_2d.max():.2e}')
    plt.show()

    
    # %% Visualize contribution map
    """
    Visualize the contribution each measurements to the pixel reconstruction in percentage.
    All measurements were used to recover the white pixel, whereas only a few measurements 
    contributed to the reddish pixels reconstruction.
    """
    contrib = contrib_map(H_dyn_diff.cpu().numpy(), n)


    # %% Visualize dynamic matrix evolution
    """
    Show how the system matrix changes over time due to motion.
    Each frame shows the dynamic pattern for one measurement.
    """
    print(f"\nVisualizing dynamic matrix evolution...")
    H_dyn_diff_np = torch2numpy(H_dyn_diff)
    for frame in range(int(n / und ** 0.5)):
        plt.clf()
        if in_X:
            plt.imshow(H_dyn_diff_np[n * frame, :].reshape(meas_shape), cmap='gray')  # in X
        else:
            plt.imshow(H_dyn_diff_np[n * frame, :].reshape(img_shape), cmap='gray')  # in X_{ext}
        plt.suptitle("frame %d" % (n * frame), fontsize=16)
        plt.colorbar()
        plt.pause(0.01)
        clear_output(wait=True)


    # %% Prepare for reconstruction
    """
    Move computations to CPU for optimized linear algebra.
    Analyze the system matrix conditioning for reconstruction parameter selection.
    """
    print(f"\n=== Preparing for Reconstruction ===")
    print("Moving to CPU for optimized linear algebra...")
    H_dyn_diff = H_dyn_diff.cpu()
    y2 = y2.cpu()

    # %% Analyze system conditioning  
    """
    Compute singular values to understand the inverse problem difficulty.
    High condition number indicates need for regularization.
    """
    print("Analyzing system matrix conditioning...")
    sing_vals = torch.linalg.svdvals(H_dyn_diff)
    condition_number = (sing_vals[0] / sing_vals[-1]).item()
    sigma_max = sing_vals[0].item()
    sigma_min = sing_vals[-1].item()
    
    print(f"Singular value spectrum:")
    print(f"  Maximum: {sigma_max:.2e}")
    print(f"  Minimum: {sigma_min:.2e}")
    print(f"  Condition number: {condition_number:.2e}")
    

    # %% Configure regularization
    """
    Set up regularization for the ill-posed inverse problem.
    Options:
    - 'L2': Identity matrix (Tikhonov regularization)
    - 'H1': First-order finite differences (smoothing prior)
    """
    type_reg = 'H1'  # Regularization type
    
    print(f"\nSetting up {type_reg} regularization...")

    if type_reg == "H1":
        # First-order finite differences with Neumann boundary conditions
        Dx, Dy = spytorch.neumann_boundary(reco_shape)
        D2 = Dx.T @ Dx + Dy.T @ Dy
        print("  Using edge-preserving H1 regularization")
    elif type_reg == 'L2':
        # Simple L2 regularization
        D2 = torch.eye(reco_shape[0] * reco_shape[1])
        print("  Using Tikhonov L2 regularization")

    D2 = D2.to(dtype=dtype)
    print(f"  Regularization matrix shape: {D2.shape}")

    # %% Solve regularized reconstruction problem
    """
    Solve the regularized least squares problem:
    min_x \frac{1}{2} \| H_dyn x - y \|^2 + \frac{\tilde{\eta}}{2} \| D·x \|^2

    where \tilde{\eta} is the normalized regularization parameter scaled by the maximum singular value.
    """
    eta = 1e-5  # Regularization parameter (adjust based on noise level)
    
    print(f"\n=== Dynamic Reconstruction ===")
    print(f"Regularization parameter \eta: {eta:.1e}")
    
    start_time = time.time()
    x_dyn = torch.linalg.solve(H_dyn_diff.T @ H_dyn_diff + eta * sigma_max ** 2 * D2,  H_dyn_diff.T @ y2.reshape(-1))
    solve_time = time.time() - start_time
    
    print(f"Reconstruction completed in {solve_time:.2f}s")
    print(f"Solution shape: {x_dyn.shape}")

    # %% Visualize reconstruction results
    """
    Compare the original image, static reconstruction, and dynamic reconstruction.
    This shows the improvement gained by accounting for motion.
    """
    x_dyn_plot = x_dyn.view(reco_shape)

    print(f"\n=== Reconstruction Comparison ===")
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    
    # Original reference image
    ref_img = x.view(reco_shape if reco_shape == img_shape else meas_shape).cpu().numpy()
    if reco_shape != img_shape:
        ref_img = x[0, 0, amp_max:meas_shape[0] + amp_max, amp_max:meas_shape[1] + amp_max].cpu().numpy()
    
    im0 = ax[0].imshow(ref_img, cmap='gray')
    ax[0].set_title('Reference Image', fontsize=14)
    ax[0].axis('off')
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    # Static reconstruction 
    if in_X:
        static_img = x_stat.view(reco_shape).cpu().numpy()
    else:
        # Embed static reconstruction in full image size
        static_img = torch.zeros(img_shape, dtype=dtype, device=device)
        static_img[amp_max:meas_shape[0] + amp_max, amp_max:meas_shape[1] + amp_max] = x_stat.view(meas_shape)

    im1 = ax[1].imshow(static_img.cpu().numpy(), cmap='gray')
    ax[1].set_title('Static Reconstruction \n (Ignores Motion)', fontsize=14)
    ax[1].axis('off')
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    # Dynamic reconstruction
    im2 = ax[2].imshow(x_dyn_plot.cpu().numpy(), cmap='gray')
    ax[2].set_title('Dynamic Reconstruction \n (Motion Compensated)', fontsize=14)
    ax[2].axis('off')
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    #%% Quantitative evaluation
    """
    Compute image quality metrics to quantify reconstruction improvement.
    Metrics are computed in the FOV region for fair comparison.
    """
    # Extract FOV region for metric calculation
    if in_X:
        x_dyn_in_X = x_dyn_plot
    else:
        x_dyn_in_X = x_dyn_plot[amp_max:-amp_max, amp_max:-amp_max]
        
    x_ref_in_X = x[0, 0, amp_max:-amp_max, amp_max:-amp_max]
    
    # Calculate metrics for both reconstructions
    psnr_static = score.psnr(torch2numpy(x_stat), torch2numpy(x_ref_in_X))
    ssim_static = score.ssim(torch2numpy(x_stat), torch2numpy(x_ref_in_X))
    
    psnr_dynamic = score.psnr(torch2numpy(x_dyn_in_X), torch2numpy(x_ref_in_X))
    ssim_dynamic = score.ssim(torch2numpy(x_dyn_in_X), torch2numpy(x_ref_in_X))

    print(f"\n=== Quantitative Results (in the SPC FOV X) ===")
    print(f"{'Method':<25} {'PSNR (dB)':<12} {'SSIM'}")
    print("-" * 48)
    print(f"{'Static':<25} {psnr_static:<12.2f} {ssim_static:.3f}")
    print(f"{'Dynamic':<25} {psnr_dynamic:<12.2f} {ssim_dynamic:.3f}")
    
    improvement = psnr_dynamic - psnr_static

    print(f"Dynamic reconstruction achieved a PSNR improvement of {improvement:.2f} dB over static reconstruction.")


    # %% Generate dynamic reconstruction sequence
    """
    Create the full dynamic sequence by warping the reconstructed reference frame
    through the deformation field. This gives us a time-resolved reconstruction.
    """
    print(f"\n=== Generating Dynamic Sequence ===")
    print("Warping reconstructed reference through deformation field...")
    
    x_dyn_video = def_field(x_dyn_plot.view(1, 1, *reco_shape), 0, 2 * M, mode=simu_interp)
    x_dyn_video = x_dyn_video.moveaxis(time_dim, 1) 

    print(f"Generated dynamic sequence shape: {x_dyn_video.shape}")

    # %% Visualize dynamic reconstruction sequence
    """
    Display the reconstructed dynamic sequence to verify motion compensation quality.
    Compare with original dynamic sequence if desired.
    """
    print("Displaying reconstructed dynamic sequence...")
    n_frames_display = n
    n_frames = 2 * M
    x_min, x_max = x_dyn_video.min().item(), x_dyn_video.max().item()
    for frame in range(n_frames):
        n_frame = n_frames_display * frame
        if n_frame >= n_frames:
            break
        plt.close()
        x_frame = x_dyn_video[0, n_frame, 0, amp_max:img_size-amp_max, amp_max:img_size-amp_max].view(meas_shape).cpu().numpy()
        plt.imshow(x_frame, cmap="gray", vmin=x_min, vmax=x_max)  # in X
        plt.suptitle("frame %d" % (n_frame), fontsize=16)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.pause(0.01)
        clear_output(wait=True)
    
    print(f"\n=== Tutorial Complete ===")
    print("Summary of achievements:")
    print("  Generated dynamic scene with pulsating motion")
    print("  Simulated a dynamic single-pixel acquisition")
    print("  Built the motion-compensated system matrix H_dyn")
    print(r"  Reconstructed a time-resolved reconstruction \textit{a posteriori}")
    
    print("\nKey insight: Accounting for motion during acquisition significantly")
    print("improves reconstruction quality in dynamic single-pixel imaging.")

# %%
