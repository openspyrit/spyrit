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
import sys
import glob
import numpy as np
import PIL
from typing import Union


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
