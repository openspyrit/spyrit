import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import cv2
import torch
from pathlib import Path
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

from IPython.display import clear_output

from deepinv.optim import prior

from spas.metadata2 import read_metadata, read_metadata_2arms

from spyrit.misc.sampling import reorder, Permutation_Matrix
from spyrit.misc.statistics import Cov2Var
from spyrit.misc.disp import torch2numpy


@dataclass
class ExperimentConfig:
    """Configuration for experimental data processing."""
    n_acq: int = 64
    n: int = 64
    amp_max: int = 0
    zoom_factor: int = 1
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