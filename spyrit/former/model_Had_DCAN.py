# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------
#from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from ..misc.pattern_choice import Hadamard, matrix2conv, split
from collections import OrderedDict
import cv2
from scipy.stats import rankdata
from itertools import cycle;
from pathlib import Path
import spyrit.misc.walsh_hadamard as wh
import math
from .recon_functions import *

#######################################################################
# 1. Determine the important Hadamard Coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Helps determining the statistical best 
# Hadamard patterns for a given image size
# 

def optim_had(dataloader, root):
    """ Computes image that ranks the hadamard coefficients
    """
    inputs, classes = next(iter(dataloader))
    inputs = inputs.cpu().detach().numpy();
    (batch_size, channels, nx, ny) = inputs.shape;

    tot_num = len(dataloader)*batch_size;
    Cumulated_had = np.zeros((nx, ny));
    # Iterate over data.
    for inputs,labels in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i,0,:,:];
            h_img = np.abs(fht.fht2(img))/tot_num;
            Cumulated_had += h_img;
    
    Cumulated_had = Cumulated_had / np.max(Cumulated_had) * 255
    np.save(root+'{}x{}'.format(nx,ny)+'.npy', Cumulated_had)
    np.savetxt(root+'{}x{}'.format(nx,ny)+'.txt', Cumulated_had)
    cv2.imwrite(root+'{}x{}'.format(nx,ny)+'.png', Cumulated_had)
    return Cumulated_had 

def hadamard_opt_spc(M ,root, nx, ny):
    msk = np.ones((nx,ny))
    had_mat = np.load(root+'{}x{}'.format(nx,ny)+'.npy');
    had_comp = np.reshape(rankdata(-had_mat, method = 'ordinal'),(nx, ny));
    msk[np.absolute(had_comp)>M]=0;
    
    conv = Hadamard(msk); 

    return conv


def abs_walsh(dataloader, device):
    
    # Estimate tot_num
    inputs, classes = next(iter(dataloader))
    #inputs = inputs.cpu().detach().numpy();
    (batch_size, channels, nx, ny) = inputs.shape;
    tot_num = len(dataloader)*batch_size;
    
    # Init
    n = 0
    output = torch.zeros((nx,ny),dtype=torch.float32)
    H = wh.walsh_matrix(nx).astype(np.float32, copy=False)
    
    # Send to device (e.g., cuda)
    output = output.to(device)
    H = torch.from_numpy(H).to(device)
    
    # Accumulate over all images in dataset
    for inputs,_ in dataloader:
        inputs = inputs.to(device);
        n = n + inputs.shape[0]
        trans = wh.walsh2_torch(inputs,H);
        trans = torch.abs(trans)
        output = output.add(torch.sum(trans,0))
        print(f'Abs:  {n} / (less than) {tot_num} images', end='\n')
    print('', end='\n')
    
    #-- Normalize
    output = output/n;
    output = torch.squeeze(output)
    
    return output

def stat_walsh(dataloader, device, root):
    
    # Get dimensions and estimate total number of images in the dataset
    inputs, classes = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape;
    tot_num = len(dataloader)*b;
    
    # 1. Mean
    
    # Init
    n = 0
    mean = torch.zeros((nx,ny), dtype=torch.float32)
    H = wh.walsh_matrix(nx).astype(np.float32, copy=False)
    
    # Send to device (e.g., cuda)
    mean = mean.to(device)
    H = torch.from_numpy(H).to(device)
    
    # Accumulate sum over all images in dataset
    for inputs,_ in dataloader:
        inputs = inputs.to(device);
        trans = wh.walsh2_torch(inputs,H)
        mean = mean.add(torch.sum(trans,0))
        # print
        n = n + inputs.shape[0]
        print(f'Mean:  {n} / (less than) {tot_num} images', end='\n')
    print('', end='\n')
    
    # Normalize
    mean = mean/n;
    mean = torch.squeeze(mean)
    #torch.save(mean, root+'Average_{}x{}'.format(nx,ny)+'.pth')
    np.save(root / Path('Average_{}x{}'.format(nx,ny)+'.npy'), mean.cpu().detach().numpy())
    
    # 2. Covariance
    
    # Init
    n = 0
    cov = torch.zeros((nx*ny,nx*ny), dtype=torch.float32)
    cov = cov.to(device)
    
    # Accumulate (im - mu)*(im - mu)^T over all images in dataset
    for inputs,_ in dataloader:
        inputs = inputs.to(device);
        trans = wh.walsh2_torch(inputs,H)
        trans = trans - mean.repeat(inputs.shape[0],1,1,1)
        trans = trans.view(inputs.shape[0], nx*ny, 1)
        cov = torch.addbmm(cov, trans, trans.view(inputs.shape[0], 1, nx*ny))
        # print
        n += inputs.shape[0]
        print(f'Cov:  {n} / (less than) {tot_num} images', end='\n')
    print('', end='\n')
    
    # Normalize
    cov = cov/(n-1);
    #torch.save(cov, root+'Cov_{}x{}'.format(nx,ny)+'.pth') # todo?
    np.save(root / Path('Cov_{}x{}'.format(nx,ny)+'.npy'), cov.cpu().detach().numpy())
    
    return mean, cov

def stat_walsh_np(dataloader, root):
    """ 
        Computes Mean Hadamard Image over the whole dataset + 
        Covariance Matrix Amongst the coefficients
    """
    inputs, classes = next(iter(dataloader))
    inputs = inputs.cpu().detach().numpy();
    (batch_size, channels, nx, ny) = inputs.shape;
    tot_num = len(dataloader)*batch_size;
    
    H1d = wh.walsh_ordered(nx)
    
     # Abs matrix
    Mean_had = abs_walsh_ordered(dataloader, H1d, tot_num)
    print("Saving abs")
    np.save(root / Path('Abs_{}x{}'.format(nx,ny)+'.npy'), Mean_had)

    # Mean matrix
    #-- Accumulate over all images in dataset
    n = 0
    Mean_had = np.zeros((nx, ny));
    for inputs,_ in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i,0,:,:];
            h_img = wh.walsh_ordered2(img,H1d);
            Mean_had += h_img;
            n = n+1
        print(f'Mean:  {n} / (less than) {tot_num} images', end='\r')
    print('', end='\n')
    
    #-- Normalize & save
    Mean_had = Mean_had/n;
    print("Saving mean")
    np.save(root / Path('Mean_{}x{}'.format(nx,ny)+'.npy'), Mean_had)
    
    # Covariance matrix    
    n = 0
    Cov_had = np.zeros((nx*ny, nx*ny));
    for inputs,_ in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i,0,:,:];
            h_img = walsh_ordered2(img, H1d);
            Norm_Variable = np.reshape(h_img-Mean_had, (nx*ny,1));
            Cov_had += Norm_Variable*np.transpose(Norm_Variable);
            n = n+1
        print(f'Covariance:  {n} / (less than) {tot_num} images', end='\r')     
    print()
    
    #-- Normalize & save
    Cov_had = Cov_had/(n-1);  
    np.save(root / Path('Cov_{}x{}'.format(nx,ny)+'.npy'), Cov_had)



def Stat_had(dataloader, root):
    """ 
        Computes Mean Hadamard Image over the whole dataset + 
        Covariance Matrix Amongst the coefficients
    """

    inputs, classes = next(iter(dataloader))
    inputs = inputs.cpu().detach().numpy();
    (batch_size, channels, nx, ny) = inputs.shape;
    tot_num = len(dataloader)*batch_size;


    Mean_had = np.zeros((nx, ny));
    for inputs,labels in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i,0,:,:];
            h_img = fht.fht2(img);
            Mean_had += h_img;
    Mean_had = Mean_had/tot_num;


    Cov_had = np.zeros((nx*ny, nx*ny));
    for inputs,labels in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(int(inputs.shape[0]/4)):
            img = inputs[i,0,:,:];
            h_img = fht.fht2(img);
            Norm_Variable = np.reshape(h_img-Mean_had, (nx*ny,1));
            Cov_had += Norm_Variable*np.transpose(Norm_Variable);
    Cov_had = Cov_had/(tot_num-1);

    np.save(root+'Cov_{}x{}'.format(nx,ny)+'.npy', Cov_had)
    np.savetxt(root+'Cov_{}x{}'.format(nx,ny)+'.txt', Cov_had)
    
    np.save(root+'Average_{}x{}'.format(nx,ny)+'.npy', Mean_had)
    np.savetxt(root+'Average_{}x{}'.format(nx,ny)+'.txt', Mean_had)
    cv2.imwrite(root+'Average_{}x{}'.format(nx,ny)+'.png', Mean_had) #Needs conversion to Uint8!
    return Mean_had, Cov_had 


def img2mask(Value_map, M):
    (nx, ny) = Value_map.shape;
    msk = np.ones((nx, ny));
    ranked_data = np.reshape(rankdata(-Value_map, method = 'ordinal'),(nx, ny));
    msk[np.absolute(ranked_data)>M]=0;
    return msk

def Cov2Var(Cov):
    """
    Extracts Variance Matrix from Covarience Matrix
    """
    (Nx, Ny) = Cov.shape;
    diag_index = np.diag_indices(Nx);
    Var = Cov[diag_index];
    Var = np.reshape(Var, (int(np.sqrt(Nx)),int(np.sqrt(Nx))) );
    return Var

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

    
def Hadamard_Transform_Matrix(img_size):
    H = np.zeros((img_size**2, img_size**2))
    for i in range(img_size**2):
        base_function = np.zeros((img_size**2,1));
        base_function[i] = 1;
        base_function = np.reshape(base_function, (img_size, img_size));
        hadamard_function = fht.fht2(base_function);
        H[i, :] = np.reshape(hadamard_function, (1,img_size**2));
    return H

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
    H = Hadamard_Transform_Matrix(img_size);

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
    H = Hadamard_Transform_Matrix(img_size);

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
    def __init__(self, n, M, Mean, Cov, variant=0, H=None, denoi = None, Ord=None):

        super(compNet, self).__init__()
        
        self.n = n;
        self.M = M;

        
        self.even_index = range(0,2*M,2);
        self.uneven_index = range(1,2*M,2);
        
        #-- Hadamard patterns (full basis)
        if type(H)==type(None):
            H = Hadamard_Transform_Matrix(self.n)
        H = n*H; #fht hadamard transform needs to be normalized
        self.H=H
        #-- Hadamard patterns (undersampled basis)

        if type(Ord)==type(None):         
            Ord = Cov2Var(Cov);
            
        Perm = Permutation_Matrix(Ord)
        Pmat = np.dot(Perm,H);
        Pmat = Pmat[:M,:];
        Pconv = matrix2conv(Pmat);

        #-- Denoising parameters 
        Sigma = np.dot(Perm,np.dot(Cov,np.transpose(Perm)));
        diag_index = np.diag_indices(n**2);
        Sigma = Sigma[diag_index];
        Sigma = n**2/4*Sigma[:M]; #(H = nH donc Cov = n**2 Cov)!
        #Sigma = Sigma[:M];
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
        P, _ = split(Pconv, 1);
        self.P = P;
        self.P.bias.requires_grad = False;
        self.P.weight.requires_grad = False;
        self.Patt.bias.requires_grad = False;
        self.Patt.weight.requires_grad = False;


        #-- Pseudo-inverse to determine levels of noise.
        Pinv = (1/n**2)*np.transpose(Pmat);
        self.Pinv = nn.Linear(M,n**2, False)
        self.Pinv.weight.data=torch.from_numpy(Pinv);
        self.Pinv.weight.data=self.Pinv.weight.data.float();
        self.Pinv.weight.requires_grald=False;
        
        Pnorm = (1/n**2)*Pmat;
        self.Pnorm = nn.Linear(M,n**2, False)
        self.Pnorm.weight.data=torch.from_numpy(Pnorm);
        self.Pnorm.weight.data=self.Pnorm.weight.data.float();
        self.Pnorm.weight.requires_grad=False;
        

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
            self.fc1.weight.requires_grad=False;
       
        elif variant==3:
            #--- FC is learnt
            print("Measurement to image domain: free")
            
            self.fc1 = nn.Linear(M,n**2)
            
        #-- Image correction
        
        if denoi == None:
            self.recon = ConvNet()
        else:
            self.recon = denoi

    def forward(self, x):
        b,c,h,w = x.shape;        
        x = self.forward_acquire(x, b, c, h, w);
        x = self.forward_reconstruct(x, b, c, h, w);
        return x
      
    #--------------------------------------------------------------------------
    # Forward functions (with grad)
    #--------------------------------------------------------------------------
    def forward_acquire(self, x, b, c, h, w):
        #--Scale input image
        x = (x+1)/2;
        #--Acquisition
        x = x.view(b*c, 1, h, w);
        x = self.P(x)
        x = F.relu(x); ## x[:,:,1] = -1/N0 ????
        x = x.view(b*c,1, 2*self.M);
        return x
    
    def forward_reconstruct(self, x, b, c, h, w):
        print("CompNet")
        m = self.forward_preprocess(x, b, c, h, w)
        var = torch.zeros_like(m).to(m.device);
        x = self.forward_maptoimage(m, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w, m, var)
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
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M));
        return x
    
    def forward_maptoimage(self, x, b, c, h, w):
        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w)
        return x;
          
    def forward_postprocess(self, x, b, c, h, w, m, var):
        x = self.recon(x,b,c,h,w,m, var)
        x = x.view(b, c, h, w)
        return x
    
    def pinv(self, x, b, c, h, w):
        x = self.Pinv(x);
        x = x.view(b*c,1,h,w)
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
    def __init__(self, n, M, Mean, Cov, variant, N0, sig = 0.1, H=None,denoi = None, Ord=None):
        super().__init__(n, M, Mean, Cov, variant, H,denoi, Ord)
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
        x = x.view(b*c,1, 2*self.M); # x[:,:,1] < 0??? 
        
        #--Measurement noise (Gaussian approximation of Poisson)
        x = x + torch.sqrt(x)*torch.randn_like(x);  
        return x
      
    def forward_preprocess(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = x/self.N0;
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        return x;
      
    def forward_reconstruct(self, x, b, c, h, w):
        print("CompNet")
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        m = self.forward_preprocess(x, b, c, h, w)
        var = var/(self.N0**2)
        x = self.forward_maptoimage(m, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w, m, var)
        return x
      
    def forward_reconstruct_expe(self, x, b, c, h, w):
        """"
        Add C, g, s, and have the fully experimental processing here.
        """
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        m, N0_est = self.forward_preprocess_expe(x, b, c, h, w);
        var = torch.div(var, N0_est**2);
        x = self.forward_maptoimage(m, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w, m, var)
        return x
      
    def forward_reconstruct_pinv_expe(self, x, b, c, h, w):
        x = self.forward_preprocess_expe(x, b, c, h, w)
        x = self.pinv(x, b, c, h, w)      
        #-- Faster alternative
        # x = x[:,:,self.even_index] - x[:,:,self.uneven_index]
        # x = self.pinv(x, b, c, h, w);
        # N0_est = self.max(x);
        # N0_est = N0_est.view(b*c,1,1,1);
        # N0_est = N0_est.repeat(1,1,h,w);
        # x = torch.div(x,N0_est);
        # x=2*x-1;
        return x
    
    def forward_preprocess_expe(self, x, b, c, h, w):
        #-- Recombining positive and negatve values
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        #-- Estimating and normalizing by N0
        x_est = self.pinv(x, b, c, h, w);
        N0_est = self.max(x_est)
        N0_est = N0_est.view(b*c,1,1)
        N0_est = N0_est.repeat(1,1,self.M)
        x = torch.div(x,N0_est)
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M))
        return x, N0_est;
      

#     def forward_maptoimage(self, x, b, c, h, w):
#         #-- Pre-processing (use batch norm to avoid division by N0 ?)
#         var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
#         x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
#         x = 2/self.N0*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
#         #--Projection to the image domain
#         m = x;

#         var = var/(self.N0**2);

#         x = self.fc1(x);
#         x = x.view(b*c,1,h,w) 
#         return x, m, var
         

#     def forward_Pinv(self, x, b, c, h, w):
#         #-- Pre-processing (use batch norm to avoid division by N0 ?)
#         var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
#         x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
#         x = 2/self.N0*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
#         m = x
#         var = var/(self.N0**2);
#         #--Projection to the image domain
#         x = self.Pinv(x);
#         x = x.view(b*c,1,h,w)
#         return x, m, var
 
#     def forward_N0_Pinv(self, x, b, c, h, w):
#         #-- Pre-processing (use batch norm to avoid division by N0 ?)
#         var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
#         x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
#         #--Projection to the image domain
#         x = self.Pinv(x);
#         x = x.view(b*c,1,h,w)
#         N0_est = self.max(x);
#         N0_est = N0_est.view(b*c,1,1,1);
#         N0_est = N0_est.repeat(1,1,h,w);
#         x = torch.div(x,N0_est);
#         x=2*x-1; 
#         return x
     
#     def forward_N0_maptoimage(self, x, b, c, h, w):

   
      
#     def forward_N0_reconstruct(self, x, b, c, h, w):
#         x, m, var = self.forward_N0_maptoimage(x, b, c, h, w)
#         x = self.forward_postprocess(x, b, c, h, w,m, var)
#         return x
 
#     def forward_stat_comp(self, x, b, c, h, w):
#         #-- Pre-processing(Recombining positive and negatve values+normalisation) 
#         var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
#         x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
#         x = x/self.N0;
#         x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 

#         m = x
#         var = var/(self.N0**2)
#         #--Projection to the image domain
#         x = self.fc1(x);
#         x = x.view(b*c,1,h,w) 
#         return x,m,var
    



#==============================================================================    
# B. NOISY MEASUREMENTS (NOISE LEVEL IS VARYING) + denoising architecture
#==============================================================================
class DenoiCompNet(noiCompNet):
    def __init__(self, n, M, Mean, Cov, variant, N0, sig = 0.1, H=None,denoi = None, Ord=None):
        super().__init__(n, M, Mean, Cov, variant, N0, sig, H,denoi, Ord)
        print("Denoised Measurements")
        
    def forward_denoise(self, x, var, b, c, h, w):
        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma + var), x);
        return x

    def forward_reconstruct(self, x, b, c, h, w):
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index]
        m = self.forward_preprocess(x, b, c, h, w)
        var = var/(self.N0**2)
        x = self.forward_denoise(m, var, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w, m, var)
        return x
    
    def forward_reconstruct_comp(self, x, b, c, h, w):
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index]
        m = self.forward_preprocess(x, b, c, h, w)
        var = var/(self.N0**2)
        x = self.forward_maptoimage(m, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w, m, var)
        return x
    
    def forward_reconstruct_mmse(self, x, b, c, h, w):
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index]
        m = self.forward_preprocess(x, b, c, h, w)
        var = var/(self.N0**2)
        x = self.forward_denoise(m, var, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        return x
    
    def forward_reconstruct_pinv(self, x, b, c, h, w):
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.pinv(x, b, c, h, w)
        return x
    
    def forward_reconstruct_expe(self, x, b, c, h, w, C=0, s=0, g=1):
        var = g**2*(x[:,:,self.even_index] + x[:,:,self.uneven_index]) - 2*C*g +2*s**2;
        m, N0_est = self.forward_preprocess_expe(x, b, c, h, w)
        var = torch.div(var, N0_est**2);
        x = self.forward_denoise(m, var, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w, m, var)
        return x
    
    def forward_reconstruct_pinv_expe(self, x, b, c, h, w, C=0, s=0, g=1):
        x, _ = self.forward_preprocess_expe(x, b, c, h, w)
        x = self.pinv(x, b, c, h, w) 
        return x

########################################################################
# 2. Define a custom Loss function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Creating custom loss function
# ---------------------------
# Just to make sure that all functions work the same way...   
# ie that they take the same number of arguments

class Weight_Decay_Loss(nn.Module):
    
    def __init__(self, loss):
        super(Weight_Decay_Loss,self).__init__()
        self.loss = loss;

    def forward(self,x,y, net):
        mse=self.loss(x,y);
        return mse

class Variance_Loss(nn.Module):
    def __init__(self, loss, mean_net):
        super(Variance_Loss,self).__init__()
        self.loss = loss;
        self.mean_net = mean_net;
        #self.mean_net.eval()
        for param in self.mean_net.parameters():
            param.requires_grad = False;

    def forward(self,x,y, net):
        cond_mean = self.mean_net(x);
        var = torch.mul((x-cond_mean),(x-cond_mean));
        mse=self.loss(var, y);
        return mse

class Covariance_Loss(nn.Module):
    def __init__(self, loss, mean_net):
        super(Covariance_Loss,self).__init__()
        self.loss = loss;
        self.mean_net = mean_net;
        #self.mean_net.eval()
        for param in self.mean_net.parameters():
            param.requires_grad = False;

    def forward(self,x,y, net):
        b, c, h, w = x.shape;
        cond_mean = self.mean_net(x);
        x = x.reshape(b*c, 1, h*w, 1);
        cond_mean = cond_mean.reshape(b*c, 1, h*w, 1);
        Cov = (x-cond_mean)*torch.transpose((x-cond_mean), -2, -1);
       
        y = y.reshape(b*c, 1, h*w, 1);
        cov_est = y*torch.transpose(y, -2,-1);

        mse=self.loss(Cov, cov_est);
        return mse


