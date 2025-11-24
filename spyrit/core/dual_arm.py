"""
Module designed for the dual-arm single-pixel camera.

This module contains several classes:
1. KeyPoints: determines the key points between a CMOS camera and a single-pixel camera.
2. ComputeHomography: computes the homography matrix between the two camera views.

"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
from pathlib import Path
import torch.nn as nn
import time
import math

from IPython.display import clear_output

from spyrit.misc.disp import torch2numpy
from spyrit.core.meas import HadamSplit2d
from spyrit.misc.statistics import Cov2Var
from spyrit.core.warp import DeformationField

from spyrit.misc.load_data import read_acquisition
from spyrit.misc.disp import get_frame

@dataclass
class MouseState:
    """State container for mouse interactions."""
    x: int = 0
    y: int = 0
    img: Optional[np.ndarray] = None


# Global state for mouse callbacks (necessary for OpenCV callback system)
_cmos_state = MouseState()
_sp_state = MouseState()


def draw_circle(event: int, x: int, y: int, flags: int, param) -> None:
    """Mouse callback for CMOS image interaction."""
    global _cmos_state
    if event == cv2.EVENT_LBUTTONDBLCLK and _cmos_state.img is not None:
        cv2.circle(_cmos_state.img, (x, y), 2, (255, 0, 0), -1)
        _cmos_state.x, _cmos_state.y = x, y


def draw_circle_2(event: int, x: int, y: int, flags: int, param) -> None:
    """Mouse callback for single-pixel camera image interaction."""
    global _sp_state
    if event == cv2.EVENT_LBUTTONDBLCLK and _sp_state.img is not None:
        cv2.circle(_sp_state.img, (x, y), 1, (255, 0, 0), -1)
        _sp_state.x, _sp_state.y = x, y



class KeyPoints(nn.Module):
    """Determines the key points between two images. Src: CMOS, Dest: SPC"""

    def __init__(self, src_img: np.ndarray, dest_img: np.ndarray, homo_folder: str = ''):
        """
        Initializes the KeyPoints class.

        Args:
            src_img: Source image (CMOS).
            dest_img: Destination image (SPC).
            homo_folder: Folder for saving homography data.
        """
        super().__init__()

        self.src_img = src_img
        self.dest_img = dest_img
        self.homo_folder = homo_folder

    def place_hand_keypoints(self, win_up_factor: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Manually place keypoints on both images using mouse interaction.
        
        Args:
            win_up_factor: Window upscaling factor for SP image display.
            
        Returns:
            Tuple of (src_points, dest_points) as numpy arrays.
        """
        src_points, dest_points = [], []
        global _cmos_state, _sp_state
        
        _cmos_state.img = self.src_img.copy()
        _sp_state.img = np.rot90(self.dest_img, 2).copy()

        n = self.dest_img.shape[0]
        print(f'Image size: {n}, upscaling factor: {win_up_factor}')

        # Setup OpenCV windows
        cv2.namedWindow('CMOS', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('CMOS', draw_circle)

        cv2.namedWindow('SP', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SP', win_up_factor * n, win_up_factor * n)
        cv2.setMouseCallback('SP', draw_circle_2)

        print("DOUBLE CLIC puis appuyer sur 'a' pour placer un point sur l'image CMOS")
        print("DOUBLE CLIC puis appuyer sur 'z' pour placer un point sur l'image SP")
        print("Appuyer sur 'q' pour quitter")

        while True:
            cv2.imshow('CMOS', _cmos_state.img)
            cv2.imshow('SP', _sp_state.img)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('a'):
                point = (_cmos_state.x, _cmos_state.y)
                print(f"CMOS point: {point}")
                src_points.append(point)
                print(f"Current src_points: {src_points}")
            elif key == ord('z'):
                point = (_sp_state.x, _sp_state.y)
                print(f"SP point: {point}")
                dest_points.append(point)
                print(f"Current dest_points: {dest_points}")

        cv2.destroyAllWindows()

        # Transform coordinates for SP image (account for rotation)
        dest_points = np.array(dest_points)
        dest_points = np.array([n - 1, n - 1]) - dest_points
        src_points = np.array(src_points)

        # Save keypoints
        data_path = self.homo_folder
        data_path.mkdir(parents=True, exist_ok=True)
        
        np.save(data_path / "handmade_dest_kp.npy", dest_points)
        np.save(data_path / "handmade_src_kp.npy", src_points)

        return src_points, dest_points
    

    def find_sift_keypoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find matching keypoints using SIFT feature detection.

        Returns:
            Tuple of (src_points, dest_points) as numpy arrays.
        """
        # Convert to 8-bit images
        img1 = (255 * self.src_img).astype(np.uint8)
        img2 = (255 * self.dest_img).astype(np.uint8)

        ## CODE FROM OPENCV EXAMPLE
        coord_p1, coord_p2 = [], []
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                p1 = kp1[m.queryIdx].pt
                p2 = kp2[m.trainIdx].pt
                if p1 not in coord_p1 and p2 not in coord_p2:
                    coord_p1.append(p1), coord_p2.append(p2)
                    good.append([m])

        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        plt.imshow(img3, cmap="gray"), plt.show()

        return np.array(coord_p1), np.array(coord_p2)
    

    def find_shi_tomasi_keypoints(self, max_corners: int = 20, quality_level: float = 0.1, 
                                  min_distance: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find corners using Shi-Tomasi corner detector.
        
        TODO: This method doesn't match keypoints between images yet. Is it possible?
        
        Args:
            max_corners: Maximum number of corners to detect.
            quality_level: Quality level for corner detection.
            min_distance: Minimum distance between corners.
            
        Returns:
            Tuple of (src_points, dest_points) as numpy arrays.
        """
        warnings.warn(
            "Shi-Tomasi keypoints are not matched between images yet. "
            "Feel free to contribute if you need this functionality."
        )
        
        img1 = (255 * self.src_img).astype(np.uint8)
        img2 = (255 * self.dest_img).astype(np.uint8)
        
        corners_cmos = cv2.goodFeaturesToTrack(
            img1, maxCorners=max_corners, qualityLevel=quality_level, 
            minDistance=min_distance, useHarrisDetector=False
        )
        corners_spc = cv2.goodFeaturesToTrack(
            img2, maxCorners=max_corners, qualityLevel=quality_level, 
            minDistance=min_distance, useHarrisDetector=False
        )
        
        if corners_cmos is None or corners_spc is None:
            return np.array([]), np.array([])
            
        src_points = corners_cmos[:, 0, :]
        dest_points = corners_spc[:, 0, :]

        return src_points, dest_points

    def external_keypoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load keypoints from external files.
        
        Returns:
            Tuple of (src_points, dest_points) as numpy arrays.
        """
        data_path = Path("../data/exp_data") / self.homo_folder
        
        src_file = data_path / "external_src_kp.npy"
        dest_file = data_path / "external_dest_kp.npy"
        
        if not src_file.exists() or not dest_file.exists():
            raise FileNotFoundError(
                f"External keypoint files not found: {src_file} or {dest_file}"
            )
        
        src_points = np.load(src_file)
        dest_points = np.load(dest_file)

        return src_points, dest_points

    def forward(self, kp_method: str, read_hand_kp: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main method to find keypoints using specified method.
        
        Args:
            kp_method: Method to use ('hand', 'sift', 'shi-tomasi', 'external').
            read_hand_kp: Whether to read existing hand-placed keypoints.
            
        Returns:
            Tuple of (src_points, dest_points) as numpy arrays.
        """
        if kp_method == 'hand':
            if read_hand_kp:
                data_path = self.homo_folder / kp_method
                src_points = np.load(data_path / "handmade_src_kp.npy")
                dest_points = np.load(data_path / "handmade_dest_kp.npy")
            else:
                src_points, dest_points = self.place_hand_keypoints()

        elif kp_method == 'sift':
            src_points, dest_points = self.find_sift_keypoints()

            # print(dest_points[4], src_points[4])
            # src_points, dest_points = np.delete(src_points, 4, axis=0), np.delete(dest_points, 4, axis=0) #point aberrant pour le chat
            # src_points, dest_points = np.delete(src_points, 2, axis=0), np.delete(dest_points, 2, axis=0)
          
        elif kp_method == 'shi-tomasi':
            src_points, dest_points = self.find_shi_tomasi_keypoints()

        elif kp_method == 'external':
            src_points, dest_points = self.external_keypoints()

        else:
            raise ValueError(
                f"Unknown keypoint method: {kp_method}. "
                f"Supported methods: 'hand', 'sift', 'shi-tomasi', 'external'"
            )

        return src_points, dest_points
    

def recalibrate(X: torch.Tensor, shape: Tuple[int, int], homography_inv: torch.Tensor, 
                amp_max: int = 0) -> torch.Tensor:
    """
    Recalibrate tensor X using inverse homography transformation.
    
    Args:
        X: Input tensor of shape (batch_size, n_wav, height, width).
        shape: Target shape (n, m).
        homography_inv: Inverse homography matrix (3x3).
        amp_max: Maximum amplitude offset.
        
    Returns:
        Calibrated tensor.
    """
    n, m = shape
    batch_size, n_wav, height, width = X.shape
    dtype = X.dtype
    device = X.device

    # Create coordinate meshgrid
    y_coords = torch.linspace(0, n - 1, n, dtype=dtype, device=device)
    x_coords = torch.linspace(0, m - 1, m, dtype=dtype, device=device)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
    
    # Apply amplitude offset
    x_grid = x_grid - amp_max
    y_grid = y_grid - amp_max
    
    # Create homogeneous coordinates
    ones = torch.ones_like(x_grid)
    coords_homogeneous = torch.stack([x_grid, y_grid, ones], dim=2)  # (n, m, 3)
    
    # Apply inverse homography transformation
    # Reshape for batch matrix multiplication
    coords_flat = coords_homogeneous.view(-1, 3, 1)  # (n*m, 3, 1)
    
    # Transform coordinates
    transformed_coords = torch.bmm(
        homography_inv.unsqueeze(0).expand(coords_flat.shape[0], -1, -1),
        coords_flat
    )  # (n*m, 3, 1)
    
    # Convert from homogeneous coordinates
    transformed_coords = transformed_coords.squeeze(-1)  # (n*m, 3)
    w = transformed_coords[:, 2]
    x_new = transformed_coords[:, 0] / w
    y_new = transformed_coords[:, 1] / w
    
    # Reshape back to grid format
    x_new = x_new.view(n, m)
    y_new = y_new.view(n, m)
    
    # Normalize coordinates to [-1, 1] for grid_sample
    x_new_norm = x_new / (width - 1) * 2 - 1
    y_new_norm = y_new / (height - 1) * 2 - 1
    
    # Create grid for grid_sample (note: grid_sample expects (x, y) order)
    grid = torch.stack((x_new_norm, y_new_norm), dim=2)  # (n, m, 2)
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch_size, n, m, 2)
    
    # Apply interpolation
    X_calibrated = nn.functional.grid_sample(
        X, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    return X_calibrated



class ComputeHomography(nn.Module):
    """Computes the homography of an experimental acquisition using DLT. Src: CMOS, Dest: SPC"""

    def __init__(self, data_root: Path, data_folder: str, data_file_prefix: str, 
                 n: int, n_acq: int):
        """
        Initializes the ComputeHomography class.
        
        Args:
            data_root: Root directory of the data.
            data_folder: Folder containing the data.
            data_file_prefix: Prefix of the data files.
            n: Size of the reconstructed image.
            n_acq: Size of the acquisition.
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.data_folder = data_folder
        self.data_file_prefix = data_file_prefix
        self.n = n
        self.n_acq = n_acq


    def DLT(self, points_source: np.ndarray, points_target: np.ndarray) -> np.ndarray:
        """
        Computes homography using Direct Linear Transform (DLT) method.
        
        Requires at least 4 corresponding point pairs.
        
        Args:
            points_source: Source image keypoints (Nx2).
            points_target: Target image keypoints (Nx2).
            
        Returns:
            3x3 homography matrix.
        """
        if len(points_source) < 4 or len(points_target) < 4:
            raise ValueError("At least 4 point correspondences are required for DLT")
            
        A = self._construct_A(points_source, points_target)
        
        # Solve using SVD
        _, _, vh = np.linalg.svd(A, full_matrices=True)
        
        # Solution is the last column of V (last row of V^T)
        homography = vh[-1].reshape((3, 3))

        return homography / homography[2, 2]

    def _construct_A(self, points_source: np.ndarray, points_target: np.ndarray) -> np.ndarray:
        """
        Construct matrix A for DLT algorithm.
        
        Args:
            points_source: Source image keypoints.
            points_target: Target image keypoints.
            
        Returns:
            A matrix for SVD decomposition.
        """
        assert points_source.shape == points_target.shape, \
            "Source and target points must have the same shape"
        
        num_points = points_source.shape[0]

        matrices = []
        for i in range(num_points):
            partial_A = self._construct_A_partial(points_source[i], points_target[i])
            matrices.append(partial_A)

        return np.concatenate(matrices, axis=0)


    def _construct_A_partial(self, point_source, point_target):
        x, y, z = point_source[0], point_source[1], 1
        x_t, y_t, z_t = point_target[0], point_target[1], 1

        A_partial = np.array([
            [0, 0, 0, -z_t * x, -z_t * y, -z_t * z, y_t * x, y_t * y, y_t * z],
            [z_t * x, z_t * y, z_t * z, 0, 0, 0, -x_t * x, -x_t * y, -x_t * z]
        ])

        return A_partial


    def forward(self, kp_method: str, homo_folder: str = '', read_homography: bool = False, 
                save_homography: bool = True, read_hand_kp: bool = False, 
                snapshot: bool = True, show_calib: bool = False) -> torch.Tensor:
        """
        Compute the homography between the CMOS and single pixel cameras.
        
        Args:
            kp_method: Keypoint detection method.
            homo_folder: Folder for homography data.
            read_homography: Whether to load existing homography.
            save_homography: Whether to save computed homography.
            read_hand_kp: Whether to read existing hand-placed keypoints.
            snapshot: Whether to use snapshot or video for CMOS data.
            show_calib: Whether to show calibration visualization.
            
        Returns:
            Computed homography matrix as torch tensor.
        """
        # Create output directory
        output_dir = self.data_root / homo_folder / kp_method
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load CMOS image
        g_frame0 = self._load_cmos(snapshot)
        
        # Load single pixel camera reconstruction
        f_stat_np = self._load_spc()

        # Normalize images
        g_frame0 = (g_frame0 - g_frame0.min()) / (g_frame0.max() - g_frame0.min())
        f_stat_np = (f_stat_np - f_stat_np.min()) / (f_stat_np.max() - f_stat_np.min())

        # Load or compute homography
        if read_homography:
            homography_np = self._load_homography(output_dir)
            homography = torch.from_numpy(homography_np)
        else:
            homography = self._compute_homography(
                g_frame0, f_stat_np, kp_method, homo_folder, 
                read_hand_kp, save_homography, output_dir
            )

        # Visualize calibration if requested
        if show_calib:
            self._visualize_calibration(g_frame0, f_stat_np, homography)

        return homography
    

    def _load_cmos(self, snapshot: bool) -> np.ndarray:
        """Load CMOS camera data from snapshot or video."""
        if snapshot:
            file_path = self.data_root / self.data_folder / f"{self.data_file_prefix}_IDScam_before_acq.npy"
            if not file_path.exists():
                raise FileNotFoundError(f"CMOS snapshot file not found: {file_path}")
            return np.load(file_path)
        else:
            video_path = self.data_root / self.data_folder / f"{self.data_file_prefix}_video.avi"
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            return get_frame(str(video_path), 0)
        

    def _load_spc(self) -> np.ndarray:
        """Load single pixel camera raw data and reconstruct an image using orthogonality of Hadamard patterns."""
        # Load measurement data
        _, meas = read_acquisition(self.data_root, self.data_folder, self.data_file_prefix)

        # Load covariance matrix
        stat_folder = Path('./stats/')
        cov_file = stat_folder / f'Cov_{self.n_acq}x{self.n_acq}.npy'
        
        if not cov_file.exists():
            raise FileNotFoundError(f"Covariance file not found: {cov_file}")
            
        Cov_acq = np.load(cov_file)
        Ord_acq = Cov2Var(Cov_acq)
        Ord = torch.from_numpy(Ord_acq)

        # Create measurement operator
        if self.n < self.n_acq:
            Ord = Ord[:self.n, :self.n]
            meas_op_stat = HadamSplit2d(M=self.n**2, h=self.n, order=Ord, dtype=torch.float64)
            meas = meas[:2 * self.n ** 2]
        else:
            meas_op_stat = HadamSplit2d(M=self.n**2, h=self.n_acq, order=Ord, dtype=torch.float64)

        # Process measurements
        m = meas[::2, :] - meas[1::2, :]
        m_pan = np.mean(m, axis=1)
        m_pan = torch.from_numpy(m_pan.reshape((1, -1)))

        # Reconstruct image
        f_stat = meas_op_stat.fast_pinv(m_pan)
        return torch2numpy(f_stat).reshape((self.n, self.n))
    

    def _load_homography(self, output_dir: Path) -> np.ndarray:
        """Load existing homography matrix."""
        homo_file = output_dir / 'homography.npy'
        if not homo_file.exists():
            raise FileNotFoundError(f"Homography file not found: {homo_file}")
        return np.load(homo_file)
    

    def _compute_homography(self, g_frame0: np.ndarray, f_stat_np: np.ndarray, 
                           kp_method: str, homo_folder: str, read_hand_kp: bool,
                           save_homography: bool, output_dir: Path) -> torch.Tensor:
        """Compute homography using keypoint detection."""
        # Find keypoints
        kp_finder = KeyPoints(g_frame0, f_stat_np, self.data_root / homo_folder)
        src_points, dest_points = kp_finder(kp_method, read_hand_kp=read_hand_kp)

        if len(src_points) == 0 or len(dest_points) == 0:
            raise ValueError("No keypoints found. Cannot compute homography.")

        # Visualize keypoints
        self._visualize_keypoints(g_frame0, f_stat_np, src_points, dest_points)

        # Compute homography using DLT
        homography_np = self.DLT(src_points, dest_points)
        homography = torch.from_numpy(homography_np)

        # Save homography if requested
        if save_homography:
            np.save(output_dir / "homography.npy", homography_np)

        return homography
    

    def _visualize_keypoints(self, g_frame0: np.ndarray, f_stat_np: np.ndarray, 
                           src_points: np.ndarray, dest_points: np.ndarray) -> None:
        """Visualize detected keypoints on both images."""
        colors = np.random.rand(len(src_points))
        
        # CMOS image keypoints
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(g_frame0, cmap="gray")
        plt.scatter(src_points[:, 0], src_points[:, 1], c=colors, s=50, marker='o')
        plt.title("Keypoints on CMOS image", fontsize=16)
        plt.axis('off')

        # SP image keypoints
        plt.subplot(1, 2, 2)
        plt.imshow(f_stat_np, cmap="gray")
        plt.scatter(dest_points[:, 0], dest_points[:, 1], c=colors, s=50, marker='o')
        plt.title("Keypoints on SP image", fontsize=16)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


    def _visualize_calibration(self, g_frame0: np.ndarray, f_stat_np: np.ndarray, 
                             homography: torch.Tensor) -> None:
        """Visualize calibration results."""
        homography_inv = torch.linalg.inv(homography)

        # Calibrate CMOS image
        g_frame0_tensor = torch.from_numpy(g_frame0).unsqueeze(0).unsqueeze(0)
        img_cmos_calibrated = recalibrate(
            g_frame0_tensor, 
            (f_stat_np.shape[0], f_stat_np.shape[0]), 
            homography_inv
        )
        # img_cmos_calibrated = tensor2img(img_cmos_calibrated)[:, :, 0]
        img_cmos_calibrated = torch2numpy(img_cmos_calibrated.moveaxis(1, -1))[0, :, :, 0]

        # Create visualization
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))
        
        axs[0].imshow(img_cmos_calibrated, cmap="gray")
        axs[0].set_title("CMOS camera calibrated", fontsize=20)
        axs[0].axis('off')
        
        axs[1].imshow(f_stat_np, cmap="gray")
        axs[1].set_title("SP reconstruction", fontsize=20)
        axs[1].axis('off')
        
        diff_img = f_stat_np - img_cmos_calibrated
        im = axs[2].imshow(diff_img, cmap="Spectral")
        axs[2].set_title("Difference: SP - CMOS", fontsize=20)
        axs[2].axis('off')
        fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()



@dataclass
class MotionConfig:
    """Configuration for motion estimation parameters."""
    n: int  # Pattern size
    M: int  # Number of measurement patterns
    n_ppg: int  # Number of patterns per gate
    T: float  # Total acquisition time
    frame_ref: int = 0  # Reference frame index
    dtype: torch.dtype = torch.float64


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
                 dtype: Optional[torch.dtype] = torch.float64,
                 device: Optional[Union[str, torch.device]] = torch.device('cpu')):
        """
        Initialize motion estimation module.
        
        Args:
            deform_path: Path to deformation field files.
            deform_prefix: Prefix for deformation files.
            n, M, n_ppg, T, frame_ref: Motion estimation parameters.
            homography: 3x3 homography transformation matrix.
            translation: Translation offset (x, y).
            dtype: Data type for computations (torch.float32, torch.float64, etc.).
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
        self.config = MotionConfig(n=n, M=M, n_ppg=n_ppg, T=T, frame_ref=frame_ref, dtype=dtype)

        # Setup device
        self.dtype = dtype
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
        

    def _load_deformation_movies(self, warping: str) -> Tuple[torch.Tensor, int, int, int]:
        """
        Load deformation field movies from NIfTI files.
        
        Args:
            warping: If 'pattern', use 'direct' mode, if 'image' use 'inverse' mode.
            
        Returns:
            Tuple of (combined_motion_data, width, height, n_frames).
            
        Raises:
            FileNotFoundError: If deformation files are not found.
            ValueError: If file dimensions are inconsistent.
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required to load NIfTI files. Please install it (e.g. via 'pip install nibabel').")


        if warping == 'image':
            mode = 'inverse'
        elif warping == 'pattern':
            mode = 'direct'
        else:
            raise ValueError(f"Invalid warping mode: {warping}. Use 'image' or 'pattern'.")
        
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
    

    def estim_motion_from_CMOS(self, warping: str, amp_max: int = 0, 
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


    def forward(self, warping: str, amp_max: int = 0, 
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
        


