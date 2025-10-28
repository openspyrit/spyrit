# %% [markdown]
"""
# Tutorial for deformation field generation with SpyRIT

This tutorial demonstrates how to create and apply deformation fields to
simulate motion in images using the SpyRIT library.

Topics covered:
1. Creating affine deformation fields (translation, rotation, scaling)
2. Creating elastic deformation fields for realistic motion
3. Visualizing deformed image sequences

"""

# %% Import bib
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import math

from IPython.display import clear_output

from spyrit.misc.disp import imagesc
from spyrit.misc.statistics import transform_gray_norm

from spyrit.core.warp import AffineDeformationField, ElasticDeformation



#%% Set parameters and load image
n = 64          # size of the FOV side in pixels
img_size = 88   # full image side's size in pixels
n_frames = 100  # number of frames in the dynamic sequence

i = 0  # Image index (modify to change the image)
spyritPath = '../data/data_online/'
imgs_path = os.path.join(spyritPath, "spyrit/")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dtype = torch.float64
simu_interp = 'bilinear'  # interpolation order for motion simulation

time_dim = 1  # time dimension index in tensors

fov_shape = (n, n)
img_shape = (img_size, img_size)
amp_max = (img_shape[0] - fov_shape[0]) // 2

# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=img_size)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

x, _ = next(iter(dataloader))
print(f"Shape of input images: {x.shape}")

# Select image
x = x[i : i + 1, :, :, :].to(dtype=dtype, device=device)
x = x.detach().clone()
x = (x - x.min()) / (x.max() - x.min())

x_plot = x.view(img_shape).cpu()
imagesc(x_plot, r"Original image $x$")


# %% Affine deformation
"""
Affine deformation examples:
1. Translation (diagonal movement) 
2. Rotation (spinning motion)
3. Surface-preserving scaling (sort of breathing motion)

IMPORTANT: SpyRIT uses normalized coordinates [-1, 1]
To convert pixels to normalized: normalized = 2 * pixels / image_size
"""
T = 1000  # time of a period

def translation(t):
    """Translation transformation - diagonal movement."""
    d_pix_tot = 10  # amplitude of translation in pixels
    assert d_pix_tot < amp_max, "Translation amplitude too large for image size!"
    d_normalized = 2 * d_pix_tot / img_size # Convert to normalized coordinates
    
    d_pix_unit = d_normalized / (2 * T)  # normalized amplitude per time unit (for a time vector of length 2T)
    tx = d_pix_unit * t
    ty = -d_pix_unit * t
    
    return torch.tensor(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ],
        dtype=dtype,
    )

def rotation(t):
    """Rotation transformation - spinning motion."""
    theta = 2 * math.pi * t / T  # One full rotation per period T
    return torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta),  math.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=dtype,
    )

def s(t):
    a = 0.2  # amplitude in normalized coordinates
    return 1 + a * math.sin(t * 2 * math.pi / T) 

def pulsation(t):
    """Surface-preserving transformation - pulsating motion."""
    return torch.tensor(
        [
            [1 / s(t), 0, 0],
            [0, s(t), 0],
            [0, 0, 1],
        ],
        dtype=dtype,
    )

# Choose which transformation to use (try different ones!)
time_vector = torch.linspace(0, 2 * T, n_frames)
transformation_function = translation  # Change this to 'translation', 'rotation', or 'pulsation' to try others

def_field = AffineDeformationField(transformation_function, time_vector, img_shape, dtype=dtype, device=device)


# %% Simulate motion
print(f"Applying {transformation_function.__name__} transformation...")
x_motion = def_field(x, 0, n_frames, mode=simu_interp)

x_motion = x_motion.moveaxis(time_dim, 1)
print("x_motion.shape:", x_motion.shape)


# %% PLOT ENTIRE DEFORMATION WITHIN THE FOV
n_frames_display = 5

x_min, x_max = x_motion.min().item(), x_motion.max().item()
for frame in range(n_frames):
    n_frame = n_frames_display * frame
    if n_frame >= n_frames:
        break
    plt.close()
    plt.imshow(x_motion[0, n_frame, 0, amp_max:img_size-amp_max, amp_max:img_size-amp_max].view(fov_shape).cpu().numpy(), cmap="gray", vmin=x_min, vmax=x_max)  # in X
    # plt.imshow(x_motion[0, n_frame, 0].view(img_shape).cpu().numpy(), cmap="gray", vmin=x_min, vmax=x_max)  # in X_{ext}
    plt.suptitle("frame %d" % (n_frame), fontsize=16)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.pause(0.1)
    clear_output(wait=True)


# %% Random elastic deformation
"""
Elastic deformation creates more realistic, non-parametric motion that can 
simulate tissue deformation or fluid motion.

Parameters:
- magnitude_amp: Controls magnitude of deformations (in pixels)
- smoothness: Controls spatial correlation (higher = smoother)
- n_interpolation: Number of keyframes for temporal interpolation
"""
magnitude_amp = 300   # Magnitude in pixels
smoothness = 5       # Spatial smoothness parameter
n_interpolation = 3  # Temporal interpolation points

def_field = ElasticDeformation(magnitude_amp, smoothness, img_shape, n_frames, n_interpolation, dtype=dtype, device=device)
elastic_std = def_field._compute_field_std()
print(f"Generated random elastic deformation field has an std of {elastic_std:.2f} pixels.")

# %% Simulate motion
x_motion = def_field(x, 0, n_frames, mode=simu_interp)
x_motion = x_motion.moveaxis(time_dim, 1)
print("x_motion.shape:", x_motion.shape)


# %% PLOT ENTIRE DEFORMATION WITHIN THE FOV
n_frames_display = 5

x_min, x_max = x_motion.min().item(), x_motion.max().item()
for frame in range(n_frames):
    n_frame = n_frames_display * frame
    if n_frame >= n_frames:
        break
    plt.close()
    x_frame = x_motion[0, n_frame, 0, amp_max:img_size-amp_max, amp_max:img_size-amp_max].view(fov_shape).cpu().numpy()
    plt.imshow(x_frame, cmap="gray", vmin=x_min, vmax=x_max)  # in X
    plt.suptitle("frame %d" % (n_frame), fontsize=16)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.pause(0.01)
    clear_output(wait=True)

# %% Visualize the deformation field with quiver arrows
interval = torch.linspace(0, img_size - 1, img_size, dtype=torch.float64)
x1, x2 = torch.meshgrid(interval, interval, indexing='xy')

x1, x2 = x1 / img_size * 2 - 1, x2 / img_size * 2 - 1

x1, x2 = x1.cpu().numpy(), x2.cpu().numpy()
field = def_field.field.cpu().numpy()

n_frames_display = 5
for frame in range(n_frames):
    n_frame = n_frames_display * frame
    if n_frame >= n_frames:
        break
    plt.figure(figsize=(6, 6))
    step = 6  # change this to plot fewer or more arrows
    plt.quiver(
        x1[::step, ::step], 
        -x2[::step, ::step], 
        (field[n_frame, ::step, ::step, 0] - x1[::step, ::step]),
        -(field[n_frame, ::step, ::step, 1] - x2[::step, ::step]), 
        angles="xy", scale_units='xy', scale=1
    )
    plt.suptitle("frame %d" % n_frame, fontsize=16)
    plt.pause(0.01)
    clear_output(wait=True)
# %%
