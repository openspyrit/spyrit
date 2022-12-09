# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:37:27 2020

@author: crombez
"""
import numpy as np
from scipy.stats import rankdata

def Permutation_Matrix(mat):
    r"""Returns permutation matrix from sampling map
                
    Args:
        mat: sampling map, where high value means high significance.
        
    Shape:
        - Input: (n,n)
        - Output: (n*n, n*n)
    """
    (nx, ny) = mat.shape;
    Reorder = rankdata(-mat, method = 'ordinal');
    Columns = np.array(range(nx*ny));
    P = np.zeros((nx*ny, nx*ny));
    P[Reorder-1, Columns] = 1;
    return P

def expend_vect(Vect,N1,N2): # Expened a vectors of siez N1 to N2
    V_out = np.zeros(N2)
    S = int(N2/N1)
    j = 0
    ad = 0
    for i in range(N1):
        for j in range(0,S):
            V_out[i+j+ad] = Vect[i]
        ad += S-1
    return(V_out)

def data_conv_hadamard(H,Data,N):
    
    for i in range(N):
        H[:,:,i] = H[:,:,i]*Data
    return(H)

def Sum_coll(Mat,N_lin,N_coll): # Return the sum of all the raw of the N1xN2 matrix
    Mturn = np.zeros(N_lin)
    
    for i in range(N_coll):
        Mturn += Mat[:,i]
        
    return(Mturn)

def compression_1D(H,Nl,Nc,Nh): #Compress a Matrix of N1xN2xN3 into a matrix of N1xN3 by summing the raw
    H_1D = np.zeros((Nl,Nh))
    for i in range(Nh):
        H_1D[:,i] = Sum_coll(H[:,:,i],Nl,Nc)
    
    return(H_1D)

def normalize_mat_2D(Mat): # Normalise a N1xN2 matrix by is maximum value
    Max = np.amax(Mat)
    return(Mat*(1/Max))

def normalize_by_median_mat_2D(Mat): # Normalise a N1xN2 matrix by is median value
    Median = np.median(Mat)
    return(Mat*(1/Median))

def remove_offset_mat_2D(Mat): # Substract the mean value of the matrix
    Mean = np.mean(Mat)
    return(Mat-Mean)
    

def resize(Mat,Nl,Nc,Nh): # Re-size a matrix of N1xN2 into N1xN3
    Mres = np.zeros((Nl,Nc))
    for i in range(Nl):
        Mres[i,:] = expend_vect(Mat[i,:],Nh,Nc)
    return(Mres)

def stack_depth_matrice(Mat,Nl,Nc,Nd): # Stack a 3 by 3 matrix along its third dimensions
    M_out = np.zeros((Nl,Nc))
    for i in range(Nd):
        M_out += Mat[:,:,i]
    return(M_out)
    
#fuction that need to be better difended 
    
def smooth(y, box_pts): #Smooth a vectors
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def reject_outliers(data, m=2): #Remove 
    return np.where(abs(data - np.mean(data)) < m * np.std(data),data,0)

def clean_out(Data,Nl,Nc,Nh,m=2):
    
    Mout = np.zeros((Nl,Nc,Nh))                
    for i in range(Nh):
        Mout[:,:,i] = reject_outliers(Data[:,:,i],m)
    return(Data)
        
