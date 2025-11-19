# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:06:19 2020

@author: crombez
"""

import os
import glob
import numpy as np
import PIL
import math

import torch
from pathlib import Path
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

from spyrit.misc.statistics import Cov2Var
from spyrit.misc.disp import torch2numpy

from spyrit.core.meas import HadamSplit2d
from spyrit.core.prep import Unsplit



def Files_names(Path, name_type):
    files = glob.glob(Path + name_type)
    print
    files.sort(key=os.path.getmtime)
    return [os.path.basename(x) for x in files]


def load_data_recon_3D(Path_files, list_files, Nl, Nc, Nh):
    Data = np.zeros((Nl, Nc, Nh))

    for i in range(0, 2 * Nh, 2):
        Data[:, :, i // 2] = np.rot90(
            np.array(PIL.Image.open(Path_files + list_files[i]))
        ) - np.rot90(np.array(PIL.Image.open(Path_files + list_files[i + 1])))

    return Data


# Load the data of the hSPIM and compresse the spectrale dimensions to do the reconstruction for every lambda
# odl convention the set of data has to be arranged in such way that the positive part of the hadamard motifs comes first
def load_data_Comp_1D_old(Path_files, list_files, Nh, Nl, Nc):
    Data = np.zeros((Nl, Nh))

    for i in range(0, 2 * Nh, 2):
        Data[:, i // 2] = Sum_coll(
            np.rot90(np.array(PIL.Image.open(Path_files + list_files[i])), 3), Nl, Nc
        ) - Sum_coll(
            np.rot90(np.array(PIL.Image.open(Path_files + list_files[i + 1])), 3),
            Nl,
            Nc,
        )

    return Data


# Load the data of the hSPIM and compresse the spectrale dimensions to do the reconstruction for every lambda
# new convention the set of data has to be arranged in such way that the negative part of the hadamard motifs comes first
def load_data_Comp_1D_new(Path_files, list_files, Nh, Nl, Nc):
    Data = np.zeros((Nl, Nh))

    for i in range(0, 2 * Nh, 2):
        Data[:, i // 2] = Sum_coll(
            np.rot90(np.array(PIL.Image.open(Path_files + list_files[i + 1])), 3),
            Nl,
            Nc,
        ) - Sum_coll(
            np.rot90(np.array(PIL.Image.open(Path_files + list_files[i])), 3), Nl, Nc
        )

    return Data


def download_girder(
    server_url: str,
    hex_ids: Union[str, list[str]],
    local_folder: str,
    file_names: Union[str, list[str]] = None,
):
    """
    Downloads data from a Girder server and saves it locally.

    This function first creates the local folder if it does not exist. Then, it
    connects to the Girder server and gets the file names for the files
    whose name are not provided. For each file, it checks if it already exists
    by checking if the file name is already in the local folder. If not, it
    downloads the file.

    Args:
        server_url (str): The URL of the Girder server.

        hex_id (str or list[str]): The hexadecimal id of the file(s) to download.
        If a list is provided, the files are downloaded in the same order and
        are saved in the same folder.

        local_folder (str): The path to the local folder where the files will
        be saved. If it does not exist, it will be created.

        file_name (str or list[str], optional): The name of the file(s) to save.
        If a list is provided, it must have the same length as hex_id. Each
        element equal to `None` will be replaced by the name of the file on the
        server. If None, all the names will be obtained from the server.
        Default is None. All names include the extension.

    Raises:
        ValueError: If the number of file names provided does not match the
        number of files to download.

    Returns:
        list[str]: The absolute paths to the downloaded files.
    """
    # leave import in function, so that the module can be used without
    # girder_client
    import girder_client

    # check the local folder exists
    if not os.path.exists(local_folder):
        print("Local folder not found, creating it... ", end="")
        os.makedirs(local_folder)
        print("done.")

    # connect to the server
    gc = girder_client.GirderClient(apiUrl=server_url)

    # create lists if strings are provided
    if type(hex_ids) is str:
        hex_ids = [hex_ids]
    if file_names is None:
        file_names = [None] * len(hex_ids)
    elif type(file_names) is str:
        file_names = [file_names]

    if len(file_names) != len(hex_ids):
        raise ValueError("There must be as many file names as hex ids.")

    abs_paths = []

    # for each file, check if it exists and download if necessary
    for id, name in zip(hex_ids, file_names):

        if name is None:
            # get the file name
            name = gc.getFile(id)["name"]

        # check the file exists
        if not os.path.exists(os.path.join(local_folder, name)):
            # connect to the server to download the file
            print(f"Downloading {name}... ", end="\r")
            gc.downloadFile(id, os.path.join(local_folder, name))
            print(f"Downloading {name}... done.")

        else:
            print("File already exists at", os.path.join(local_folder, name))

        abs_paths.append(os.path.abspath(os.path.join(local_folder, name)))

    return abs_paths[0] if len(abs_paths) == 1 else abs_paths



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
    try:
        from spas.metadata2 import read_metadata, read_metadata_2arms
    except ImportError:
        raise ImportError("Single-pixel acquisition software (SPAS) package is required to read metadata. Please install it (see https://github.com/openspyrit/spas).")

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
        from spyrit.core.dual_arm import recalibrate
    except ImportError:
        raise ImportError("There may be circular imports issues, please report if this error occurs.")

    if config is None:
        config = ExperimentConfig(
            n_acq=64,
            n=64,
            amp_max=0,
            zoom_factor=1,
            dtype=homography.dtype
        )

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
    prep_op = Unsplit()
    meas_empty_torch = torch.from_numpy(meas_empty).to(config.dtype)
    y_empty = torch.mean(meas_empty_torch, axis=1).view(1, -1)
    y_empty_diff = prep_op(y_empty)

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


def generate_synthetic_tumors(
    x: torch.Tensor,
    tumor_params: List[dict],
) -> torch.Tensor:
    """
    Creates synthetic Gaussian tumors to a tensor of shape (batch, n_wav, *img_shape).
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch, n_wav, *img_shape)
        tumor_params (List[dict]): List of tumor parameters. Each dict should contain:
            - 'center': (row, col) center position of the tumor
            - 'sigma_x': Standard deviation of the Gaussian in the x direction
            - 'sigma_y': Standard deviation of the Gaussian in the y direction
            - 'amplitude': Amplitude of the tumor
            - 'channels': List of channel indices to add the tumor to (if None, adds to all channels)
            - 'angle' (optional): Rotation angle in degrees (counter-clockwise). Default is 0.
        
    Returns:
        torch.Tensor: Tensor with synthetic tumors added
    """
    dtype = x.dtype
    device = x.device
    
    _, n_wav, h, w = x.shape
    
    tumors = torch.zeros_like(x, dtype=dtype, device=device)
    
    # Create coordinate grids
    y_axis = torch.arange(h, dtype=dtype, device=device)
    x_axis = torch.arange(w, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y_axis, x_axis, indexing='ij')
    
    for tumor_param in tumor_params:
        center = tumor_param['center']
        sigma_x = float(tumor_param['sigma_x'])
        sigma_y = float(tumor_param['sigma_y'])
        amplitude = float(tumor_param['amplitude'])
        channels = tumor_param.get('channels', None)
        # Optional rotation angle in degrees (default 0). Positive rotates counter-clockwise.
        angle_deg = float(tumor_param.get('angle', 0.0))
        theta = math.radians(angle_deg)

        # Coordinates relative to center
        x_rel = xx - float(center[1])
        y_rel = yy - float(center[0])

        # Rotate coordinates into the Gaussian's principal axes (apply R(-theta))
        c = math.cos(theta)
        s = math.sin(theta)
        x_rot = c * x_rel + s * y_rel
        y_rot = -s * x_rel + c * y_rel

        # Avoid division by zero
        sigma_x = max(sigma_x, 1e-8)
        sigma_y = max(sigma_y, 1e-8)

        # Generate rotated ellipsoidal Gaussian
        gauss = amplitude * torch.exp(
            -(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2))
        )

        if channels is None:
            channels = list(range(n_wav))

        tumors[:, channels, :, :] += gauss.unsqueeze(0).unsqueeze(0)
                
    return tumors, torch.clamp(x + tumors, 0.0, 1.0)