# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:48:37 2020

@author: crombez
"""
import sys
import numpy as np
sys.path.append('/home/crombez/Documents/PhD/python/libreries/')
from misc.walsh_hadamard import walsh2


def recon_Walsh(Data,n): # Return the product of the a 2D matrix with the Walsh ordered matrix of the same size
    return(np.dot(Data,walsh_ordered(n)))

def recon_Walsh_evry_lambda(Data,Nl,Nc,Nh): # Return the products of the each 2D matrix of the 3D matrix with the Walsh ordered matrix of the same size
    M_recon = np.zeros((Nl,Nh,Nc))
    Waslh = walsh2(Nh)
    for i in range(Nc):
        M_recon[:,:,i] = np.dot(Data[:,i,:],Waslh)
    return(M_recon)   
