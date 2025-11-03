import numpy as np
import math
import scipy.interpolate
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import nibabel as nib
import time

from pathlib import Path
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
from IPython.display import clear_output

from spas.metadata2 import read_metadata, read_metadata_2arms

from spyrit.misc.statistics import Cov2Var
from spyrit.misc.disp import torch2numpy
from spyrit.core.warp import DeformationField


@dataclass
class ExperimentConfig:
    """Configuration for experimental data processing."""
    n_acq: int = 64
    n: int = 64
    amp_max: int = 0
    zoom_factor: int = 1
    dtype: torch.dtype = torch.float64

@dataclass
class MotionConfig:
    """Configuration for motion estimation parameters."""
    n: int  # Pattern size
    M: int  # Number of measurement patterns
    n_ppg: int  # Number of patterns per gate
    T: float  # Total acquisition time
    frame_ref: int = 0  # Reference frame index
    dtype: torch.dtype = torch.float64


def read_acquisition(data_root: Path, data_folder: str, 
                    data_file_prefix: str) -> Tuple[dict, np.ndarray]:
    """
    Read acquisition data and metadata from experimental files.
    
    Args:
        data_root: Root directory of the data.
        data_folder: Folder containing the data.
        data_file_prefix: Prefix of the data files.
        
    Returns:
        Tuple of (acquisition_parameters, measurement_data).
        
    Raises:
        FileNotFoundError: If metadata or data files are not found.
    """
    data_root = Path(data_root)
    
    # Read metadata
    meta_path = data_root / data_folder / f"{data_file_prefix}_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    try:
        metadata = read_metadata_2arms(meta_path)
    except Exception:
        print("Falling back to single-arm metadata reader")
        metadata = read_metadata(meta_path)
   
    # Read spectral data
    data_path = data_root / data_folder / f"{data_file_prefix}_spectraldata.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Spectral data file not found: {data_path}")
    
    raw = np.load(data_path)
    meas = raw['spectral_data']

    return metadata, meas


def get_frame(movie_path: Union[str, Path], frame_number: int = 0) -> np.ndarray:
    """
    Extract a specific frame from a video file.
    
    Args:
        movie_path: Path to the video file.
        frame_number: Frame number to extract (0-indexed).
        
    Returns:
        Grayscale frame as numpy array.
        
    Raises:
        FileNotFoundError: If video file doesn't exist.
        ValueError: If frame cannot be read or video is invalid.
    """
    movie_path = Path(movie_path)
    if not movie_path.exists():
        raise FileNotFoundError(f"Video file not found: {movie_path}")
    
    cap = cv2.VideoCapture(str(movie_path))

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {movie_path}")

    # Get total frame count for validation
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total_frames:
        cap.release()
        raise ValueError(f"Frame {frame_number} not available. Video has {total_frames} frames.")

    # Set frame position directly (more efficient than iterating)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_number} from video")

    # Convert to grayscale
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    return gray_frame


def empty_acqui(data_root: Path, data_folder: str, data_file_prefix: str, homography: torch.Tensor, 
                config: Optional[ExperimentConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process empty acquisition data for calibration.
    
    Args:
        data_root: Root directory of the data.
        homography: Homography transformation matrix.
        config: Experimental configuration. If None, uses default values.
        
    Returns:
        Tuple of (calibrated_cmos_image, sp_reconstruction).
        
    Raises:
        ImportError: If homography module cannot be imported.
        FileNotFoundError: If required data files are not found.
    """
    try:
        from spyrit.core.meas import HadamSplit2d
        from homography import recalibrate
    except ImportError as e:
        raise ImportError(f"Required modules not available: {e}")

    if config is None:
        config = ExperimentConfig()

    # Load covariance matrix
    stat_folder = Path('./stats/')
    cov_file = stat_folder / f'Cov_{config.n_acq}x{config.n_acq}.npy'
    
    if not cov_file.exists():
        raise FileNotFoundError(f"Covariance file not found: {cov_file}")

    Cov_acq = np.load(cov_file)
    Ord_acq = Cov2Var(Cov_acq)

    Ord = torch.from_numpy(Ord_acq)

    # Create measurement operator
    M = config.n ** 2
    meas_op_stat = HadamSplit2d(M=M, h=config.n, order=Ord, dtype=config.dtype)

    # Read empty acquisition data
    try:
        _, meas_empty = read_acquisition(data_root, data_folder, data_file_prefix)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Empty acquisition data not found: {e}")

    # Process measurements
    meas_empty_torch = torch.from_numpy(meas_empty)
    y_empty = torch.mean(meas_empty_torch, axis=1)
    y_empty = y_empty - y_empty[1]
    y_empty_diff = y_empty[::2] - y_empty[1::2]
    y_empty_diff = y_empty_diff.view(1, -1)

    # Reconstruct SP image
    w = meas_op_stat.fast_pinv(y_empty_diff)
    w_np = torch2numpy(w).reshape((config.n, config.n))
    w_np = np.rot90(w_np, 2)

    # Load and calibrate CMOS image
    cmos_file = data_root / data_folder / f'{data_file_prefix}_IDScam_before_acq.npy'
    if not cmos_file.exists():
        raise FileNotFoundError(f"CMOS image file not found: {cmos_file}")
    
    g_frame0_empty = np.load(cmos_file).astype(np.float64)

    # Apply homography calibration
    homography_inv = torch.linalg.inv(homography)
    g_frame0_empty_tensor = torch.from_numpy(g_frame0_empty).unsqueeze(0).unsqueeze(0).to(config.dtype)
    
    img_cmos_calibrated = recalibrate(
        g_frame0_empty_tensor, 
        (config.n, config.n), 
        homography_inv, 
        amp_max=config.amp_max
    )
    img_cmos_calibrated_np = np.rot90(torch2numpy(img_cmos_calibrated[0, 0]), 2)

    return img_cmos_calibrated_np, w_np



def interp_spectral_response(wavelengths: np.ndarray, plot_interp: bool = False, 
                            interpolation_kind: str = 'quadratic') -> torch.Tensor:
    """
    Interpolate spectral response function for given wavelengths.
    Function designed for IDS UI-3880CP-M-GL Rev 2 camera.
    
    Args:
        wavelengths: Array of wavelengths to interpolate.
        plot_interp: Whether to plot the interpolation.
        interpolation_kind: Type of interpolation ('linear', 'quadratic', 'cubic').
        
    Returns:
        Interpolated spectral response as torch tensor.
        
    Raises:
        ValueError: If interpolation_kind is not supported.
    """
    # Reference spectral response data
    reference_wavelengths = np.array([400 + 50 * i for i in range(13)])
    quantum_efficiency = np.array([
        0.68, 0.75, 0.75, 0.7, 0.58, 0.48, 0.4, 
        0.31, 0.23, 0.17, 0.12, 0.06, 0.04
    ])
    
    # Validate interpolation kind
    valid_kinds = ['linear', 'quadratic', 'cubic']
    if interpolation_kind not in valid_kinds:
        raise ValueError(f"interpolation_kind must be one of {valid_kinds}, got {interpolation_kind}")
    
    # Create interpolation function
    try:
        spectral_response_func = scipy.interpolate.interp1d(
            reference_wavelengths, quantum_efficiency, 
            kind=interpolation_kind, bounds_error=False, fill_value=0.
        )
    except ValueError as e:
        raise ValueError(f"Interpolation failed: {e}")

    # Interpolate for given wavelengths
    interpolated_response = spectral_response_func(wavelengths)

    # Visualization
    if plot_interp:
        plt.figure(figsize=(10, 6))
        plt.plot(reference_wavelengths, quantum_efficiency, "o", 
                color='black', markersize=8, label="Reference values")
        plt.plot(wavelengths, interpolated_response, "--", 
                color="blue", linewidth=2, label=f'{interpolation_kind} interpolation')
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Quantum Efficiency', fontsize=12)
        plt.title('Spectral Response Interpolation', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return torch.from_numpy(interpolated_response.reshape((-1, 1)))


def save_motion_video(x_motion, out_path, amp_max, img_size, fps=820):
    """
    Save a motion video from x_motion tensor.
    x_motion: tensor with shape (batch, time, channel, H, W)
    out_path: pathlib.Path or str
    Crops using amp_max as in the script.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h_crop = img_size - 2 * amp_max
    w_crop = h_crop

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_crop, h_crop), True)
    if not writer.isOpened():
        raise RuntimeError("cv2 VideoWriter failed to open")
    for t in range(x_motion.shape[1]):
        frame_wide = x_motion[0, t, 0].cpu().numpy()

        mn, mx = frame_wide.min(), frame_wide.max()

        frame = frame_wide[amp_max:img_size - amp_max, amp_max:img_size - amp_max]

        if mx > mn:
            frame8 = ((frame - mn) / (mx - mn) * 255.0).astype('uint8')
        else:
            frame8 = (frame * 0).astype('uint8')
        frame_bgr = cv2.cvtColor(frame8, cv2.COLOR_GRAY2BGR)
        writer.write(frame_bgr)
    writer.release()
    print(f"Saved motion video to {out_path}")



class MotionFieldProjector(nn.Module):
    """
    Projects motion fields from CMOS camera perspective to Single Pixel Camera (SPC) perspective.
    
    This class loads pre-computed motion fields from NIfTI files and performs:
        - Geometric transformation via homography (CMOS to SPC coordinate mapping)
        - Temporal interpolation to match SPC acquisition timing for each illumination pattern
        - Reference frame definition
    
    The result is a motion field suitable for dynamic single-pixel imaging applications.
    """

    def __init__(self, deform_path: Union[str, Path], deform_prefix: str, 
                 n: int, M: int, n_ppg: int, T: float, frame_ref: int =0, 
                 homography: torch.Tensor = torch.eye(3),
                 translation: Tuple[float, float] = (0.0, 0.0),
                 device: Optional[Union[str, torch.device]] = torch.device('cpu')):
        """
        Initialize motion estimation module.
        
        Args:
            deform_path: Path to deformation field files.
            deform_prefix: Prefix for deformation files.
            n, M, n_ppg, T, frame_ref: Motion estimation parameters.
            homography: 3x3 homography transformation matrix.
            translation: Translation offset (x, y).
            device: Device to use for computations ('cpu', 'cuda', etc.).
            
        Raises:
            FileNotFoundError: If deformation path doesn't exist.
            ValueError: If homography matrix is not 3x3.
        """
        super().__init__()

        self.deform_path = Path(deform_path)
        if not self.deform_path.exists():
            raise FileNotFoundError(f"Deformation path not found: {deform_path}")
            
        self.deform_prefix = deform_prefix
        self.config = MotionConfig(n=n, M=M, n_ppg=n_ppg, T=T, frame_ref=frame_ref)

        # Setup device
        self.device = device

        # Validate homography matrix
        if homography.shape != (3, 3):
            raise ValueError(f"Homography must be 3x3 matrix, got {homography.shape}")

        self.homography = homography.clone().detach().to(dtype=self.config.dtype, device=self.device)
        self.translation = translation

        # Apply translation to homography
        self.homography[0, 2] += translation[0]
        self.homography[1, 2] += translation[1]

        # Precompute inverse homography
        self.homography_inv = torch.linalg.inv(self.homography)

        # Initialize storage for motion fields
        self.def_field_cmos: Optional[torch.Tensor] = None
        self.def_field_spc: Optional[torch.Tensor] = None
        self.u_cmos: Optional[torch.Tensor] = None
        

    def _load_deformation_movies(self, warping: bool) -> Tuple[torch.Tensor, int, int, int]:
        """
        Load deformation field movies from NIfTI files.
        
        Args:
            warping: If True, use 'direct' mode, else 'inverse' mode.
            
        Returns:
            Tuple of (combined_motion_data, width, height, n_frames).
            
        Raises:
            FileNotFoundError: If deformation files are not found.
            ValueError: If file dimensions are inconsistent.
        """
        mode = 'direct' if warping else 'inverse'
        
        movies = []
        for i in range(2):
            file_path = self.deform_path / f"{self.deform_prefix}_{mode}_{i+1}.img"
            if not file_path.exists():
                raise FileNotFoundError(f"Deformation file not found: {file_path}")
            
            print(f"Loading {file_path}")
            movies.append(nib.load(file_path))

        # Extract dimensions from header
        hdr = movies[0].header
        dims = hdr['dim']
        _, width, height, _, n_frames, _, _, _ = dims
        width, height, n_frames = int(width), int(height), int(n_frames)

        print(f'Loaded {n_frames} frames of size {width}x{height}')

        # Load and combine motion data
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f'Using device: {device} for stacking operation')
        ti = time.time()
        u1_cmos = torch.from_numpy(movies[0].get_fdata().transpose()).to(self.device)
        u2_cmos = torch.from_numpy(movies[1].get_fdata().transpose()).to(self.device)

        u_cmos = torch.stack([u1_cmos, u2_cmos], dim=1).reshape(u1_cmos.shape[0], -1, u1_cmos.shape[2], u1_cmos.shape[3])
        tf = time.time()
        print(f'Time to load deformation movies: {tf - ti:.2f} seconds')
        
        return u_cmos.to(self.config.dtype), width, height, n_frames
    

    def _create_coordinate_grids(self, l: int, amp_max: int, width: int, height: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create coordinate grids for spatial transformations.
        
        Args:
            l: Grid size for SP coordinates.
            amp_max: Amplitude offset for the extended field of view.
            width: CMOS image width.
            height: CMOS image height.
            
        Returns:
            Tuple of (sp_grid, cmos_grid).
        """
        
        # SP coordinate grid
        interval = torch.linspace(0, l - 1, l, dtype=self.config.dtype, device=self.device)
        x1_sp, x2_sp = torch.meshgrid(interval, interval, indexing='xy')
        x1_sp, x2_sp = x1_sp - amp_max, x2_sp - amp_max

        # CMOS coordinate grid
        interval_1 = torch.linspace(0, width - 1, width, dtype=self.config.dtype, device=self.device)
        interval_2 = torch.linspace(0, height - 1, height, dtype=self.config.dtype, device=self.device)
        x1_cmos, x2_cmos = torch.meshgrid(interval_1, interval_2, indexing='xy')
        
        return (x1_sp, x2_sp), (x1_cmos, x2_cmos)
    

    def _apply_homography_vectorized(self, x1_sp: torch.Tensor, x2_sp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply homography transformation using vectorized operations.
        
        Args:
            x1_sp: X coordinates in SP space.
            x2_sp: Y coordinates in SP space.
            
        Returns:
            Tuple of transformed coordinates (x1_new, x2_new).
        """
        # Create homogeneous coordinates
        ones = torch.ones_like(x1_sp)
        coords_homogeneous = torch.stack([x1_sp, x2_sp, ones], dim=0)
        
        # Reshape for efficient batch matrix multiplication
        coords_flat = coords_homogeneous.view(3, -1)  # (3, l*l)
        
        # Transform coordinates
        transformed_coords = self.homography_inv @ coords_flat  # (3, l*l)
        
        # Convert from homogeneous coordinates
        w = transformed_coords[2, :]
        x1_new = (transformed_coords[0, :] / w).view_as(x1_sp)
        x2_new = (transformed_coords[1, :] / w).view_as(x2_sp)
        
        return x1_new, x2_new
    

    def estim_motion_from_CMOS(self, warping: bool, amp_max: int = 0, 
                              show_deform_field: bool = False) -> None:
        """
        Estimate motion field from CMOS camera data.
        
        Args:
            warping: Whether to use direct or inverse motion field from MS motion
            estimation software. It is the same warping parameter as in Dynamic classes
            from `spyrit.core.meas`.
            amp_max: Maximum amplitude for coordinate offset, due to the extended field of view.
            show_deform_field: Whether to visualize the deformation field.
            
        Raises:
            FileNotFoundError: If required files are not found.
        """
        l = self.config.n + 2 * amp_max

        # Load deformation movies
        u_cmos, width, height, n_frames = self._load_deformation_movies(warping)
        self.u_cmos = u_cmos

        # Create coordinate grids
        (x1_sp, x2_sp), (x1_cmos, x2_cmos) = self._create_coordinate_grids(l, amp_max, width, height)

        # Apply homography transformation (vectorized)
        x1_new, x2_new = self._apply_homography_vectorized(x1_sp, x2_sp)

        # Normalize coordinates for grid_sample [-1, 1]
        x1_new_norm = x1_new / (width - 1) * 2 - 1
        x2_new_norm = x2_new / (height - 1) * 2 - 1

        # Create sampling grid efficiently (avoid .repeat())
        grid = torch.stack((x1_new_norm, x2_new_norm), dim=2)  # (l, l, 2)
        grid = grid.unsqueeze(0).expand(n_frames, -1, -1, -1)  # More memory efficient than repeat

        # Create CMOS grid for displacement calculation
        grid_cmos = torch.stack((x1_cmos, x2_cmos), dim=0)  # (2, width, height)
        grid_cmos = grid_cmos.unsqueeze(0).expand(n_frames, -1, -1, -1)  # More memory efficient

        # Calculate displacement and apply interpolation
        du_cmos = u_cmos - grid_cmos

        # Move data to GPU for the compute-intensive grid_sample operation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device} for grid_sample operation')
        t1 = time.time()
        if device.type == 'cuda':
            du_cmos_gpu = du_cmos.to(device)
            grid_gpu = grid.to(device)
            
            du = nn.functional.grid_sample(
                du_cmos_gpu, grid_gpu, mode='bilinear', padding_mode='border', align_corners=True
            ).cpu()  # Move result back to CPU
            
            del du_cmos_gpu, grid_gpu
            torch.cuda.empty_cache()
        else:
            du = nn.functional.grid_sample(
                du_cmos, grid, mode='bilinear', padding_mode='border', align_corners=True
            )
        t2 = time.time()
        print(f"Time to apply grid_sample: {t2 - t1:.2f} seconds")

        
        # Efficient grid creation for final coordinates
        grid2 = torch.stack((x1_new, x2_new), dim=0)
        grid2 = grid2.unsqueeze(0).expand(n_frames, -1, -1, -1)
        u = du + grid2

        # Apply homography transformation to get SP coordinates (old operator A : x_cmos -> x_sp)
        u1_sp, u2_sp = self._apply_homography_to_motion(u)

        # Combine results
        res = torch.stack((u1_sp, u2_sp), dim=1)

        # Add identity transformation for first frame
        frame0 = torch.stack((x1_sp, x2_sp), dim=0).unsqueeze(0)
        res = torch.cat((frame0, res), dim=0)  # cat is more efficient than concatenate

        # Visualize deformation field if requested
        if show_deform_field:
            self._visualize_deformation_field(res, x1_sp, x2_sp, n_frames)

        # Normalize coordinates to [-1, 1] range
        res = (res + amp_max) / l * 2 - 1
        self.def_field_cmos = res


    def _apply_homography_to_motion(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply homography transformation to motion vectors.
        
        Args:
            u: Motion vectors of shape (n_frames, 2, height, width).
            
        Returns:
            Tuple of transformed motion components (u1_sp, u2_sp).
        """
        H = self.homography
        
        # Vectorized homography application
        u1_sp = (H[0, 0] * u[:, 0, :, :] + H[0, 1] * u[:, 1, :, :] + H[0, 2]) / \
                (H[2, 0] * u[:, 0, :, :] + H[2, 1] * u[:, 1, :, :] + H[2, 2])
        
        u2_sp = (H[1, 0] * u[:, 0, :, :] + H[1, 1] * u[:, 1, :, :] + H[1, 2]) / \
                (H[2, 0] * u[:, 0, :, :] + H[2, 1] * u[:, 1, :, :] + H[2, 2])
        
        return u1_sp, u2_sp
    

    def _visualize_deformation_field(self, res: torch.Tensor, x1_sp: torch.Tensor, 
                                   x2_sp: torch.Tensor, n_frames: int) -> None:
        """
        Visualize the computed deformation field.
        
        Args:
            res: Deformation field results.
            x1_sp: X coordinates in SP space.
            x2_sp: Y coordinates in SP space.
            n_frames: Number of frames.
        """
        step = max(1, self.config.n // 6)  # Adjust step for quiver plot density
        res_np = torch2numpy(res[:, :, ::step, ::step])
        x1_sp_np = torch2numpy(x1_sp[::step, ::step])
        x2_sp_np = torch2numpy(x2_sp[::step, ::step])

        plt.figure(figsize=(10, 8))
        for f in range(n_frames - 1):
            plt.clf()
            plt.quiver(
                x1_sp_np, -x2_sp_np, 
                (res_np[f, 0, :, :] - x1_sp_np),
                -(res_np[f, 1, :, :] - x2_sp_np), 
                angles="xy", scale_units='xy', scale=1
            )
            plt.title(f"Deformation Field - Frame {f}", fontsize=16)
            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")
            plt.pause(0.01)
            clear_output(wait=True)
        plt.close()


    def def_reference(self) -> None:
        """
        Define reference frame by subtracting reference deformation from all frames.
        
        This method normalizes the deformation field so that the reference frame
        has zero deformation, making all other deformations relative to it.
        
        Raises:
            RuntimeError: If CMOS deformation field is not computed yet.
        """
        if self.def_field_cmos is None:
            raise RuntimeError("CMOS deformation field not computed. Call estim_motion_from_CMOS first.")
            
        n_frames, _, l, _ = self.def_field_cmos.shape

        # Validate reference frame index
        if self.config.frame_ref >= n_frames:
            raise ValueError(f"Reference frame {self.config.frame_ref} >= number of frames {n_frames}")

        # Create identity grid for reference
        interval = torch.linspace(0, l - 1, l, dtype=self.config.dtype)
        x1, x2 = torch.meshgrid(interval, interval, indexing='xy')
        x1_norm = x1 / l * 2 - 1
        x2_norm = x2 / l * 2 - 1

        # Calculate reference deformation
        dx1_ref = self.def_field_cmos[self.config.frame_ref, 0, :, :] - x1_norm
        dx2_ref = self.def_field_cmos[self.config.frame_ref, 1, :, :] - x2_norm
        
        # Subtract reference deformation from all frames
        self.def_field_cmos[:, 0, :, :] -= dx1_ref
        self.def_field_cmos[:, 1, :, :] -= dx2_ref

    def interpolate_between_frames(self) -> None:
        """
        Interpolate deformation field between frames for SPC acquisition timing.
        
        This method creates a temporally dense deformation field that matches
        the SPC acquisition pattern timing.
        
        Raises:
            RuntimeError: If CMOS deformation field is not computed yet.
        """
        if self.def_field_cmos is None:
            raise RuntimeError("CMOS deformation field not computed. Call estim_motion_from_CMOS first.")
            
        n_frames, _, l, _ = self.def_field_cmos.shape

        # Calculate timing parameters
        n_hppg = math.ceil(2 * self.config.M / n_frames)  # Hadamard patterns per gate period
        n_last_pat = 2 * self.config.M - n_hppg * (n_frames - 1)  # Patterns in last frame
        n_wppg = self.config.n_ppg - n_hppg  # White patterns per gate period

        print(f"Interpolation parameters: n_hppg={n_hppg}, n_last_pat={n_last_pat}, n_wppg={n_wppg}")

        # Calculate total pattern count and timing
        n_patterns = self.config.n_ppg * (n_frames - 1) + n_wppg + n_last_pat
        dt = self.config.T / n_patterns
        t_acq_cmos = n_wppg * dt

        # Initialize SPC deformation field
        self.def_field_spc = torch.zeros(
            (2 * self.config.M, 2, l, l), 
            dtype=self.config.dtype, device=self.device
        )

        print(f"Creating SPC deformation field with {2 * self.config.M} patterns")

        # Interpolate for each frame
        for f in range(n_frames):
            t_f, t_fp1, u_f, u_fp1 = self._get_frame_timing_and_deformation(
                f, n_frames, dt, t_acq_cmos
            )
            
            g_beg, g_end = self._get_pattern_indices(
                f, n_frames, n_wppg, n_last_pat
            )

            # Interpolate patterns within this frame interval
            for k in range(g_beg, g_end):
                t_k = k * dt + dt / 2  # add dt / 2 to be at the middle of the pattern exposure
                
                # Linear interpolation between frames
                alpha = (t_k - t_f) / (t_fp1 - t_f) if t_fp1 != t_f else 0.0
                interpolated_def = u_f + alpha * (u_fp1 - u_f)
                
                pattern_idx = k - (f + 1) * n_wppg
                if 0 <= pattern_idx < 2 * self.config.M:
                    self.def_field_spc[pattern_idx, :, :, :] = interpolated_def


    def _get_frame_timing_and_deformation(self, f: int, n_frames: int, dt: float, 
                                        t_acq_cmos: float) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        """
        Get timing and deformation data for frame interpolation.
        
        Args:
            f: Current frame index.
            n_frames: Total number of frames.
            dt: Time step.
            t_acq_cmos: CMOS acquisition time offset.
            
        Returns:
            Tuple of (t_f, t_fp1, u_f, u_fp1) - timing and deformation data.
        """
        # Calculate frame timings
        t_f = f * self.config.n_ppg * dt + t_acq_cmos / 2
        t_fp1 = (f + 1) * self.config.n_ppg * dt + t_acq_cmos / 2

        # Get deformation fields
        u_f = self.def_field_cmos[f, :, :, :]
        
        if f != n_frames - 1:
            u_fp1 = self.def_field_cmos[f + 1, :, :, :]
        else:
            u_fp1 = u_f  # Use same deformation for last frame

        return t_f, t_fp1, u_f, u_fp1
    

    def _get_pattern_indices(self, f: int, n_frames: int, n_wppg: int, 
                           n_last_pat: int) -> Tuple[int, int]:
        """
        Get pattern indices for current frame.
        
        Args:
            f: Current frame index.
            n_frames: Total number of frames.
            n_wppg: White patterns per group.
            n_last_pat: Patterns in last frame.
            
        Returns:
            Tuple of (g_beg, g_end) - start and end of Hadamard (SPC) pattern indices.
        """
        g_beg = f * self.config.n_ppg + n_wppg
        
        if f != n_frames - 1:
            g_end = (f + 1) * self.config.n_ppg
        else:
            g_end = g_beg + n_last_pat

        return g_beg, g_end 


    def forward(self, warping: bool, amp_max: int = 0, 
                show_deform_field: bool = False) -> torch.Tensor:
        """
        Complete forward pass for motion estimation.
        
        Args:
            warping: Whether to use direct or inverse motion field from MS motion
            estimation software. It is the same warping parameter as in Dynamic classes
            from `spyrit.core.meas`.
            amp_max: Maximum amplitude of coordinate offset for the extended field of view.
            show_deform_field: Whether to visualize the deformation field.
            
        Returns:
            SPC deformation field of shape (2*M, l, l, 2).
            
        Raises:
            RuntimeError: If any step fails during processing.
        """
        try:
            # Step 1: Convert motion from CMOS perspective to SPC
            print("Step 1: Convert motion from CMOS perspective to SPC...")
            self.estim_motion_from_CMOS(warping, amp_max=amp_max, 
                                      show_deform_field=show_deform_field)
            
            # Step 2: Define reference frame
            print("Step 2: Defining reference frame...")
            self.def_reference()
            
            # Step 3: Interpolate between frames
            print("Step 3: Interpolating between frames...")
            self.interpolate_between_frames()
            
            print("Motion estimation completed successfully!")
            return DeformationField(self.def_field_spc.moveaxis(1, -1))
            
        except Exception as e:
            raise RuntimeError(f"Motion estimation failed: {e}") from e



