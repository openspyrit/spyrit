# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

#from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import numpy as np
#import torchvision
#from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
#import time
#import os
#import copy
from ..misc.pattern_choice import Hadamard, matrix2conv, split
from collections import OrderedDict
#import cv2
from scipy.stats import rankdata
#from itertools import cycle;
#from pathlib import Path

from ..misc.disp import *
#import spyrit.misc.walsh_hadamard as wh
from spyrit.misc.statistics import *
import math


#######################################################################
# 1. Determine the important Hadamard Coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Helps determining the statistical best 
# Hadamard patterns for a given image size
# 

def hadamard_opt_spc(M ,root, nx, ny):
    msk = np.ones((nx,ny))
    had_mat = np.load(root+'{}x{}'.format(nx,ny)+'.npy');
    had_comp = np.reshape(rankdata(-had_mat, method = 'ordinal'),(nx, ny));
    msk[np.absolute(had_comp)>M]=0;
    
    conv = Hadamard(msk); 

    return conv


def img2mask(Value_map, M):
    (nx, ny) = Value_map.shape;
    msk = np.ones((nx, ny));
    ranked_data = np.reshape(rankdata(-Value_map, method = 'ordinal'),(nx, ny));
    msk[np.absolute(ranked_data)>M]=0;
    return msk


def Permutation_Matrix_root(root):
    """
        Returns Permutaion Matrix For The Hadamard Coefficients that ranks
        The Coefficients according to the Matrix defined By root.
    """
    had_mat = np.load(root);
    (nx, ny) = had_mat.shape;
    Reorder = rankdata(-had_mat, method = 'ordinal');
    Columns = np.array(range(nx*ny));
    P = np.zeros((nx*ny, nx*ny));
    P[Reorder-1, Columns] = 1;
    return P

def Permutation_Matrix(mat):
    """
        Returns permutation matrix from sampling map
        
    Args:
        mat (np.ndarray): A a n-by-n sampling map, where high value means high significance.
        
    Returns:
        P (np.ndarray): A n*n-by-n*n permutation matrix
    """
    (nx, ny) = mat.shape;
    Reorder = rankdata(-mat, method = 'ordinal');
    Columns = np.array(range(nx*ny));
    P = np.zeros((nx*ny, nx*ny));
    P[Reorder-1, Columns] = 1;
    return P

def subsample(H, mat, M):
    """
        Subsample forward operator from sampling map
        
    Args:
        H (np.ndarray): Full forward operator, a m-by-n array
        mat (np.ndarray): Sampling map
        M (int): number of measurements to keep, with M <= m
        
    Returns:
        Hsub (np.ndarray): Subsampled forward operator, a M-by-n array
    """
    Perm = Permutation_Matrix(mat)
    Hsub = np.dot(Perm,H);
    Hsub = Hsub[:M,:];
    return Hsub


def maximum_Variance_Pattern(Cov,H,M):
    """
        Returns the patterns corresponding to coefficient that have the maximun
        variance for a given image database
    """
    Var = Cov2Var(Cov)
    Perm = Permutation_Matrix(Var)
    Pmat = np.dot(Perm,H);
    Pmat = Pmat[:M,:];
    return Pmat, Perm

def permutation_from_ind(ind):
    """
        Returns 
    """
    n = len(ind)
    Columns = np.array(range(n));
    P = np.zeros((n, n));
    P[ind-1, Columns] = 1;
    return P

def ranking_matrix(mat):
    """
        Ranks the coefficient of a matrix

    """
    (nx, ny) = mat.shape;
    ind = rankdata(-mat, method = 'ordinal').reshape(nx, ny);
    return ind

def Variance_ranking(Cov):
    """
        Returns rank of the variance given the covariance
        
    Args:
        Cov (np.ndarray): Covariance matrix.
        
    Returns:
        Ind (np.ndarray): Ranking between 1 and length of Cov
    """
    Var = Cov2Var(Cov)
    Ind = ranking_matrix(Var);
    return Ind

def Variance_mask(Cov,eta=0.5):
    """Return a mask indicating the coefficients with maximum variance

    Args:
        Cov (np.ndarray): Covariance matrix.
        eta (float): Sampling ratio between 0 and 1

    Returns:
        mask (boolean array): 1 to keep, 0 otherwise
    """
    ind = Variance_ranking(Cov)
    (nx, ny) = ind.shape;
    M = math.ceil(eta*ind.size)
    print(M)
    mask = np.zeros_like(ind, dtype=bool)
    mask[ind<M] = 1
    
    return mask

    
def meas2img(meas, Ord):
    """Return image from measurement vector

    Args:
        meas (ndarray): Measurement vector.
        Ord (ndarray): Order matrix

    Returns:
        Img (ndarray): Measurement image
    """
    y = np.pad(meas, (0, Ord.size-len(meas)))
    Perm = Permutation_Matrix(Ord)
    Img = np.dot(np.transpose(Perm),y).reshape(Ord.shape)
    return Img

def img2meas(img, Ord):
    """Return measurement vector from image (not TESTED)

    Args:
        im (np.ndarray): Image.
        Ord (np.ndarray): Order matrix

    Returns:
        meas (np.ndarray): Measurement vector
    """
    Perm = Permutation_Matrix(Ord)
    meas = np.dot(Perm, np.ravel(img))
    return meas

def meas2img_torch(meas, Ord):
    """Return image from measurement vector (NOT TESTED, requires too much memory?)

    Args:
        meas (torch.Tensor): Measurement vector.
        Ord (np.ndarray): Order matrix

    Returns:
        Img (torch.Tensor): Measurement image
    """
    y = nn.functional.pad(meas, (0, Ord.size-meas.shape[2]))
    Perm = torch.from_numpy(Permutation_Matrix(Ord).astype('float32'))
    Perm = Perm.to(meas.device)
    Perm = torch.transpose(Perm,0,1)
    Img = torch.matmul(Perm,meas) # Requires too much memory
    
    return Img

def Hadamard_stat_completion_matrices(Cov_had, Mean_had, CR):
    img_size, ny = Mean_had.shape;
    
    # choice of patterns
    Var = Cov2Var(Cov_had)
    P = Permutation_Matrix(Var)
    H = wh.walsh2_matrix(img_size)/img_size

    Sigma = np.dot(P,np.dot(Cov_had,np.transpose(P)))
    mu = np.dot(P, np.reshape(Mean_had, (img_size**2,1)))
    mu1 = mu[:CR];

    Sigma1 = Sigma[:CR,:CR]
    Sigma21 = Sigma[CR:,:CR]
    
    W_p = np.zeros((img_size**2,CR))
    W_p[:CR,:] = np.eye(CR);
    W_p[CR:,:] = np.dot(Sigma21, np.linalg.inv(Sigma1));
    
    W = np.dot(H,np.dot(np.transpose(P),W_p));
    b = np.dot(H,np.dot(np.transpose(P),mu));
    return W, b, mu1, P, H

def stat_completion_matrices(P, H, Cov_had, Mean_had, CR):
    
    img_size, ny = Mean_had.shape;

    Sigma = np.dot(P,np.dot(Cov_had,np.transpose(P)))
    mu = np.dot(P, np.reshape(Mean_had, (img_size**2,1)))
    mu1 = mu[:CR];

    Sigma1 = Sigma[:CR,:CR]
    Sigma21 = Sigma[CR:,:CR]
    
    W_p = np.zeros((img_size**2,CR))
    W_p[:CR,:] = np.eye(CR);
    W_p[CR:,:] = np.dot(Sigma21, np.linalg.inv(Sigma1));
    
    W = np.dot(H,np.dot(np.transpose(P),W_p));
    b = np.dot(H,np.dot(np.transpose(P),mu));
    return W, b, mu1

def Hadamard_stat_completion_extract(img,CR, P, H):
    img_size, ny = img.shape;
    f = np.reshape(img, (img_size**2,1))
    y = np.dot(P, np.dot(H, f))
    m = y[:CR];
    return m


def Hadamard_stat_completion(W, b, mu1, m):
    nxny , col = b.shape;
    img_size = int(round(np.sqrt(nxny)));
    f_star = b + np.dot(W,(m-mu1))
    img_rec = np.reshape(f_star,(img_size,img_size));
    return img_rec;

def Hadamard_stat_completion_comp(Cov, Mean, Im, CR):
    """Reconstruct (not TESTED)

    Args:
        Cov (np.ndarray): Covariance matrix.
        Mean (np.ndarray): Mean matrix.
        Im (np.ndarray): Data matrix.

    Returns:
        meas (np.ndarray): Measurement vector
    """
    img_size, ny = Im.shape;
    Var = Cov2Var(Cov)
    P = Permutation_Matrix(Var)
    H = wh.walsh2_matrix(img_size)/img_size

    Sigma = np.dot(P,np.dot(Cov,np.transpose(P)))
    mu = np.dot(P, np.reshape(Mean, (img_size**2,1)))
    mu1 = mu[:CR];

    Sigma1 = Sigma[:CR,:CR]
    Sigma21 = Sigma[CR:,:CR]
    
    W_p = np.zeros((img_size**2,CR))
    W_p[:CR,:] = np.eye(CR);
    W_p[CR:,:] = np.dot(Sigma21, np.linalg.inv(Sigma1));
    
    W = np.dot(H,np.dot(np.transpose(P),W_p));
    b = np.dot(H,np.dot(np.transpose(P),mu));

    f = np.reshape(Im, (img_size**2,1))
    y = np.dot(P, np.dot(H, f))
    m = y[:CR];
    f_star = b + np.dot(W,(m-mu1))
    img_rec = np.reshape(f_star,(img_size, img_size));

    return img_rec;

###############################################################################
# 2. NEW Convolutional Neural Network
###############################################################################
#==============================================================================
# A. NO NOISE
#==============================================================================    
class compNet(nn.Module):
    def __init__(self, n, M, Mean, Cov, variant=0, H=None, Ord=None):
        super(compNet, self).__init__()
        
        self.n = n;
        self.M = M;
        
        self.even_index = range(0,2*M,2);
        self.uneven_index = range(1,2*M,2);
        
        #-- Hadamard patterns (full basis)
        if type(H)==type(None):
            H = wh.walsh2_matrix(self.n)/self.n
        H = n*H; #fht hadamard transform needs to be normalized
        
        #-- Hadamard patterns (undersampled basis)
        if type(Ord)==type(None):         
            Ord = Cov2Var(Cov)
            
        Perm = Permutation_Matrix(Ord)
        Pmat = np.dot(Perm,H);
        Pmat = Pmat[:M,:];
        Pconv = matrix2conv(Pmat);

        #-- Denoising parameters 
        Sigma = np.dot(Perm,np.dot(Cov,np.transpose(Perm)));
        diag_index = np.diag_indices(n**2);
        Sigma = Sigma[diag_index];
        Sigma = n**2/4*Sigma[:M];   # Multiplication by n**2 as H <- nH  leads to Cov <- n**2 Cov 
                                    # Division by 4 to get the covariance of images in [0 1], not [-1 1]
        Sigma = torch.Tensor(Sigma)
        self.sigma = Sigma.view(1,1,M)
        self.sigma.requires_grad = False

        P1 = np.zeros((n**2,1))
        P1[0] = n**2
        mean = n*np.reshape(Mean,(self.n**2,1))+P1
        mu = (1/2)*np.dot(Perm, mean)
        #mu = np.dot(Perm, np.reshape(Mean, (n**2,1)))
        mu1 = torch.Tensor(mu[:M])
        self.mu_1 = mu1.view(1,1,M)
        self.mu_1.requires_grad = False

        #-- Measurement preprocessing
        self.Patt = Pconv;
        P, T = split(Pconv, 1);
        self.P = P;
        self.T = T;
        self.P.bias.requires_grad = False;
        self.P.weight.requires_grad = False;
        self.Patt.bias.requires_grad = False;
        self.Patt.weight.requires_grad = False;
        self.T.weight.requires_grad=False;
        self.T.weight.requires_grad=False;

        #-- Pseudo-inverse to determine levels of noise.
        Pinv = (1/n**2)*np.transpose(Pmat);
        self.Pinv = nn.Linear(M,n**2, False)
        self.Pinv.weight.data=torch.from_numpy(Pinv);
        self.Pinv.weight.data=self.Pinv.weight.data.float();
        self.Pinv.weight.requires_grad=False;
        # definir la bonne pseudo inv


        #-- Measurement to image domain
        if variant==0:
            #--- Statistical Matrix completion (no mean)
            print("Measurement to image domain: statistical completion (no mean)")
            
            self.fc1 = nn.Linear(M,n**2, False)
            
            W, b, mu1 = stat_completion_matrices(Perm, H, Cov, Mean, M)
            W = (1/n**2)*W; 

            self.fc1.weight.data=torch.from_numpy(W);
            self.fc1.weight.data=self.fc1.weight.data.float();
            self.fc1.weight.requires_grad=False;
        
        if variant==1:
            #--- Statistical Matrix completion  
            print("Measurement to image domain: statistical completion")
            
            self.fc1 = nn.Linear(M,n**2)
            
            W, b, mu1 = stat_completion_matrices(Perm, H, Cov, Mean, M)
            W = (1/n**2)*W; 
            b = (1/n**2)*b;
            b = b - np.dot(W,mu1);
            self.fc1.bias.data=torch.from_numpy(b[:,0]);
            self.fc1.bias.data=self.fc1.bias.data.float();
            self.fc1.bias.requires_grad = False;
            self.fc1.weight.data=torch.from_numpy(W);
            self.fc1.weight.data=self.fc1.weight.data.float();
            self.fc1.weight.requires_grad=False;
        
        elif variant==2:
            #--- Pseudo-inverse
            print("Measurement to image domain: pseudo inverse")
            
            self.fc1 = self.Pinv;
       
        elif variant==3:
            #--- FC is learnt
            print("Measurement to image domain: free")
            
            self.fc1 = nn.Linear(M,n**2)
            
        #-- Image correction
        self.recon = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0)),
          ('relu2', nn.ReLU()),
          ('conv3', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
        ]));

    def forward(self, x):
        b,c,h,w = x.shape;
        x = self.forward_acquire(x, b, c, h, w)
        x = self.forward_reconstruct(x, b, c, h, w)
        return x
    #--------------------------------------------------------------------------
    # Forward functions (with grad)
    #--------------------------------------------------------------------------
    def forward_acquire(self, x, b, c, h, w):
        #--Scale input image
        x = (x+1)/2; 
        #--Acquisition
        x = x.view(b*c, 1, h, w);
        x = self.P(x);
        x = F.relu(x); ## x[:,:,1] = -1/N0 ????
        print("No noise")
        x = x.view(b, c, 2*self.M); 
        return x
    
    def forward_reconstruct(self, x, b, c, h, w):
        print("CompNet")
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w)
        return x
    
    def forward_reconstruct_pinv(self, x, b, c, h, w):
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.pinv(x, b, c, h, w);
        return x 
    
    def forward_reconstruct_mmse(self, x, b, c, h, w):
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        return x
    
    def forward_preprocess(self, x, b, c, h, w):
        #- Pre-processing (use batch norm to avoid division by N0 ?)
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1,h,w).to(x.device)),(b,c,self.M));
        return x
    
    def forward_maptoimage(self, x, b, c, h, w):
        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b, c, h, w)
        return x
        
    
    def forward_postprocess(self, x, b, c, h, w):
        x = x.view(b*c, 1, h, w)
        x = self.recon(x)
        x = x.view(b, c, h, w)
        return x
    
    def pinv(self, x, b, c, h, w):
        x = self.Pinv(x);
        x = x.view(b, c, h, w)
        return x
    
    #--------------------------------------------------------------------------
    # Evaluation functions (no grad)
    #--------------------------------------------------------------------------
    def acquire(self, x, b, c, h, w):
        with torch.no_grad():
            b,c,h,w = x.shape
            x = self.forward_acquire(x, b, c, h, w)
        return x
    
    def evaluate_fcl(self, x):
        with torch.no_grad():
           b,c,h,w = x.shape
           x = self.forward_acquire(x, b, c, h, w)
           x = self.forward_reconstruct_mmse(x, b, c, h, w)
        return x
     
    def evaluate_Pinv(self, x):
        with torch.no_grad():
           b,c,h,w = x.shape
           x = self.forward_Pinv(x, b, c, h, w)
        return x
    
    def evaluate(self, x):
        with torch.no_grad():
           x = self.forward(x)
        return x
    
    def reconstruct(self, x, b, c, h, w):
        with torch.no_grad():
            x = self.forward_reconstruct(x, b, c, h, w)
        return x
   
#==============================================================================    
# B. NOISY MEASUREMENTS (NOISE LEVEL IS VARYING)
#==============================================================================
class noiCompNet(compNet):
    def __init__(self, n, M, Mean, Cov, variant, N0=2500, sig=0.5, H=None, Ord=None):
        super().__init__(n, M, Mean, Cov, variant, H, Ord)
        self.N0 = N0;
        self.sig = sig;
        self.max = nn.MaxPool2d(kernel_size = n);
        print("Varying N0 = {:g} +/- {:g}".format(N0,sig*N0))
        
    def forward_acquire(self, x, b, c, h, w):
        #--Scale input image      
        a = self.N0*(1+self.sig*(torch.rand(x.shape[0])-0.5)).to(x.device)
        print('alpha in [{:.2f}--{:.2f}] photons'.format(min(a).item(),max(a).item()))
        x = a.view(-1,1,1,1)*(x+1)/2;
        
        #--Acquisition
        x = x.view(b*c, 1, h, w);
        x = self.P(x);
        x = F.relu(x);     # x[:,:,1] = -1/N0 ????
        x = x.view(b, c, 2*self.M); # x[:,:,1] < 0??? 
        
        #--Measurement noise (Gaussian approximation of Poisson)
        x = x + torch.sqrt(x)*torch.randn_like(x);  
        return x
    
    def forward_preprocess(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        x = x.view(b*c, 1, 2*self.M)
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = x/self.N0;
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b,c,self.M)); 

        return x
    
    def forward_reconstruct_expe(self, x, b, c, h, w):
        x, _ = self.forward_preprocess_expe(x, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w)
        return x
    
    def forward_reconstruct_pinv_expe(self, x, b, c, h, w):
        x, _ = self.forward_preprocess_expe(x, b, c, h, w)
        x = self.pinv(x, b, c, h, w)      

        return x
    
    def forward_reconstruct_comp_expe(self, x, b, c, h, w):
        x, _ = self.forward_preprocess_expe(x, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)   
        return x
    
    def forward_preprocess_expe(self, x, b, c, h, w):
        
        x = x.view(b*c, 1, 2*self.M)
        
        #-- Recombining positive and negative values
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        
        #-- Estimating and normalizing by N0 = K*alpha
        x_est = self.pinv(x, b, c, h, w);
        N0 = self.max(x_est)
        N0 = N0.view(b,c,1)
        print(N0)
        
        #--
        N0_est = N0.repeat(1,1,self.M)
        x = torch.div(x,N0_est)
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b,c,self.M))
        return x, N0


#==============================================================================    
# B. NOISY MEASUREMENTS (NOISE LEVEL IS VARYING) + denoising architecture
#==============================================================================
class DenoiCompNet(noiCompNet):
    def __init__(self, n, M, Mean, Cov, variant=0, N0=2500, sig=0.5, H=None, Ord=None):
        super().__init__(n, M, Mean, Cov, variant, N0, sig, H, Ord)
        print("Denoised Measurements")
   
    def forward(self, x):
        b,c,h,w = x.shape;
        x = self.forward_acquire(x, b, c, h, w)
        x = self.forward_reconstruct(x, b, c, h, w)
        return x
    
    def forward_denoise(self, x, var, b, c, h, w):
        sigma = self.sigma.repeat(b,c,1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma+var), x);
        return x
   
    def forward_reconstruct(self, x, b, c, h, w):
        x = x.view(b*c, 1, 2*self.M)
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index]
        var = var/(self.N0)**2
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.forward_denoise(x, var, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w)
        return x

    def forward_reconstruct_comp(self, x, b, c, h, w):
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w)
        return x
    
    def forward_reconstruct_mmse(self, x, b, c, h, w):
        x = x.view(b*c, 1, 2*self.M)
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index]
        var = var/(self.N0)**2
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.forward_denoise(x, var, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        return x
    
    def forward_reconstruct_pinv(self, x, b, c, h, w):
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.pinv(x, b, c, h, w)
        return x
    
    def forward_reconstruct_mmse_expe(self, x, b, c, h, w, mu=0, sig=0, K=1):
        
        
        # If C, s, g are arrays, they must have the same dimensions as  x
        if not np.isscalar(mu):
            mu = mu.view(b*c, 1, 1)

        if not np.isscalar(sig):
            sig = sig.view(b*c, 1, 1)

        if not np.isscalar(K):
            K = K.view(b*c, 1, 1)
            
        x = x.view(b*c, 1, 2*self.M)
        var = K*(x[:,:,self.even_index] + x[:,:,self.uneven_index] - 2*mu) + 2*sig**2       
        x, N0 = self.forward_preprocess_expe(x, b, c, h, w)
        var = var/N0**2  # N.B.: N0 = K*alpha
        x = self.forward_denoise(x, var, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        return x
    
    def forward_reconstruct_expe(self, x, b, c, h, w, mu=0, sig=0, K=1):    
        x = self.forward_reconstruct_mmse_expe(x, b, c, h, w, mu, sig, K)
        x = self.forward_postprocess(x, b, c, h, w)
        return x
    
   

########################################################################
# 2. Define a custom Loss function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Creating custom loss function
# ---------------------------
# Just to make sure that all functions work the same way...   
# i.e., that they take the same number of arguments

class Weight_Decay_Loss(nn.Module):
    
    def __init__(self, loss):
        super(Weight_Decay_Loss,self).__init__()
        self.loss = loss;

    def forward(self,x,y, net):
        mse = self.loss(x,y);
        return mse


