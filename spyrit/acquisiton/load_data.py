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
from PIL import Image

sys.path.append('/home/crombez/Documents/PhD/python/libreries/')

from matrix_tools import *

def Files_names(Path,name_type):
    files = glob.glob(Path+name_type)
    print
    files.sort(key=os.path.getmtime)
    return([os.path.basename(x) for x in files])

def load_data_recon_3D(Path_files,list_files,Nl,Nc,Nh):
    Data = np.zeros((Nl,Nc,Nh))
    
    for i in range(0,2*Nh,2):

        Data[:,:,i//2] = np.rot90(np.array(Image.open(Path_files+list_files[i])))-np.rot90(np.array(Image.open(Path_files+list_files[i+1])))
  
    return(Data)   


# Load the data of the hSPIM and compresse the spectrale dimensions to do the reconstruction for every lambda 
# odl convention the set of data has to be arranged in such way that the positive part of the hadamard motifs comes first
def load_data_Comp_1D_old(Path_files,list_files,Nh,Nl,Nc):
    Data = np.zeros((Nl,Nh))
    
    for i in range(0,2*Nh,2):

        Data[:,i//2] = Sum_coll(np.rot90(np.array(Image.open(Path_files+list_files[i])),3),Nl,Nc)-Sum_coll(np.rot90(np.array(Image.open(Path_files+list_files[i+1])),3),Nl,Nc)

    return(Data)

# Load the data of the hSPIM and compresse the spectrale dimensions to do the reconstruction for every lambda 
# new convention the set of data has to be arranged in such way that the negative part of the hadamard motifs comes first
def load_data_Comp_1D_new(Path_files,list_files,Nh,Nl,Nc):
    Data = np.zeros((Nl,Nh))
    
    for i in range(0,2*Nh,2):

        Data[:,i//2] = Sum_coll(np.rot90(np.array(Image.open(Path_files+list_files[i+1])),3),Nl,Nc)-Sum_coll(np.rot90(np.array(Image.open(Path_files+list_files[i])),3),Nl,Nc)

    return(Data)
