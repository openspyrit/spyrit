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