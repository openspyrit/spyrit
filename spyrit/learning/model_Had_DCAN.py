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
import fht
from ..misc.pattern_choice import Hadamard, matrix2conv, split
from collections import OrderedDict
import cv2
from scipy.stats import rankdata
#from ..misc.disp import *
from itertools import cycle;
from function.reconstruction.recon_functions import *

from ..misc.disp import *


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
    cv2.imwrite(root+'Average_{}x{}'.format(nx,ny)+'.png', Mean_had)#Needs conversion to Uint8!
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


def Permutation_Matrix(had_mat):
    """
        Returns Permutation Matrix For The Hadamard Coefficients that ranks
        The Coefficients according to the Matrix defined By had_mat.
    """
    (nx, ny) = had_mat.shape;
    Reorder = rankdata(-had_mat, method = 'ordinal');
    Columns = np.array(range(nx*ny));
    P = np.zeros((nx*ny, nx*ny));
    P[Reorder-1, Columns] = 1;
    return P

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
    
def Hadamard_Transform_Matrix(img_size):
    H = np.zeros((img_size**2, img_size**2))
    for i in range(img_size**2):
        base_function = np.zeros((img_size**2,1));
        base_function[i] = 1;
        base_function = np.reshape(base_function, (img_size, img_size));
        hadamard_function = fht.fht2(base_function);
        H[i, :] = np.reshape(hadamard_function, (1,img_size**2));
    return H

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

def Hadamard_stat_completion_comp(Cov,Mean,img, CR):
    img_size, ny = img.shape;
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


    f = np.reshape(img, (img_size**2,1))
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
    def __init__(self, n, M, Mean, Cov, variant=0, H=None, denoi = None):
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
        Var = Cov2Var(Cov)
        Perm = Permutation_Matrix(Var)
        self.Pmatfull = np.dot(Perm,H);
        self.Pmat = self.Pmatfull[:M,:];
        self.Pconv = matrix2conv(self.Pmat);

        #-- Denoising parameters 
        Sigma = np.dot(Perm,np.dot(Cov,np.transpose(Perm)));
        diag_index = np.diag_indices(n**2);
        Sigma = Sigma[diag_index];
        Sigma = n**2/4*Sigma[:M]; #(H = nH donc Cov = n**2 Cov)!
        #Sigma = Sigma[:M];
        Sigma = torch.Tensor(Sigma);
        self.sigma = Sigma.view(1,1,M);
        

        P1 = np.zeros((n**2,1));
        P1[0] = n**2;
        mean = n*np.reshape(Mean,(self.n**2,1))+P1;
        mu = (1/2)*np.dot(Perm, mean);
        #mu = np.dot(Perm, np.reshape(Mean, (n**2,1)))
        mu1 = torch.Tensor(mu[:M]);
        self.mu_1 = mu1.view(1,1,M);

        #-- Measurement preprocessing
        self.Patt = self.Pconv;
        P, _ = split(self.Pconv, 1);
        self.P = P;
#        self.T = T;
        self.P.bias.requires_grad = False;
        self.P.weight.requires_grad = False;
        self.Patt.bias.requires_grad = False;
        self.Patt.weight.requires_grad = False;
#        self.T.weight.requires_grad=False;
#        self.T.weight.requires_grad=False;

        #-- Pseudo-inverse to determine levels of noise.
        Pinv = (1/n**2)*np.transpose(self.Pmat);
        self.Pinv = nn.Linear(M,n**2, False)
        self.Pinv.weight.data=torch.from_numpy(Pinv);
        self.Pinv.weight.data=self.Pinv.weight.data.float();
        self.Pinv.weight.requires_grald=False;
        
        Pnorm = (1/n**2)*self.Pmat;
        self.Pnorm = nn.Linear(M,n**2, False)
        self.Pnorm.weight.data=torch.from_numpy(Pnorm);
        self.Pnorm.weight.data=self.Pnorm.weight.data.float();
        self.Pnorm.weight.requires_grad=False;
        
        # Layers for reconstruction : 
#        
#        self.convnet = nn.Sequential(OrderedDict([
#                ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
#                ('relu1', nn.ReLU()),
#                ('conv2', nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0)),
#                ('relu2', nn.ReLU()),
#                ('conv3', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
#                ]));
#        


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
    
    def forward_maptoimage(self, x, b, c, h, w):
        #- Pre-processing (use batch norm to avoid division by N0 ?)
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M));
        m = torch.reshape(x,(b,1,self.M,1))
        
        var = torch.zeros_like(m).to(m.device);

        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w)
        return x, m, var;
    
    def forward_postprocess(self, x, b, c, h, w, m, var):
        x = self.recon(x,b,c,h,w,m, var)
        x = x.view(b, c, h, w)
        return x
    
    def forward_reconstruct(self, x, b, c, h, w):
        x, m, var = self.forward_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w, m, var)
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
           x = self.forward_maptoimage(x, b, c, h, w)
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
            b,c,h,w = x.shape
            x = self.forward_reconstruct(x, b, c, h, w)
        return x
   
#==============================================================================    
# B. NOISY MEASUREMENTS (NOISE LEVEL IS VARYING)
#==============================================================================
class noiCompNet(compNet):
    def __init__(self, n, M, Mean, Cov, variant, N0, sig = 0.1, H=None,denoi = None):
        super().__init__(n, M, Mean, Cov, variant, H,denoi)
        self.N0 = N0;
        self.sig = sig;
        self.max = nn.MaxPool2d(kernel_size = n);
        print("Varying N0 = {:g} +/- {:g}".format(N0,sig*N0))
        
    def forward_acquire(self, x, b, c, h, w):
        #--Scale input image
        x = (self.N0*(1+self.sig*torch.randn_like(x)))*(x+1)/2;
        #--Acquisition
        x = x.view(b*c, 1, h, w);
        x = self.P(x);
        x = F.relu(x);     # x[:,:,1] = -1/N0 ????
        x = x.view(b*c,1, 2*self.M); # x[:,:,1] < 0??? 
        #--Measurement noise (Gaussian approximation of Poisson)
        x = x + torch.sqrt(x)*torch.randn_like(x);  
        return x
    
    def forward_maptoimage(self, x, b, c, h, w):
        #-- Pre-processing (use batch norm to avoid division by N0 ?)
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = 2/self.N0*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        #--Projection to the image domain
        m = x;

        var = var/(self.N0**2);

        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x, m, var
         

    def forward_Pinv(self, x, b, c, h, w):
        #-- Pre-processing (use batch norm to avoid division by N0 ?)
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = 2/self.N0*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        m = x
        var = var/(self.N0**2);
        #--Projection to the image domain
        x = self.Pinv(x);
        x = x.view(b*c,1,h,w)
        return x, m, var
 
    def forward_N0_Pinv(self, x, b, c, h, w):
        #-- Pre-processing (use batch norm to avoid division by N0 ?)
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        #--Projection to the image domain
        x = self.Pinv(x);
        x = x.view(b*c,1,h,w)
        N0_est = self.max(x);
        N0_est = N0_est.view(b*c,1,1,1);
        N0_est = N0_est.repeat(1,1,h,w);
        x = torch.div(x,N0_est);
        x=2*x-1; 
        return x
     
    def forward_N0_maptoimage(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        
        #-- Pre-processing(Estimating No and normalizing by No) 
        x_est = self.Pinv(x);
        x_est = x_est.view(b*c,1,h,w);
        N0_est = self.max(x_est);
        N0_est = N0_est.view(b*c,1,1);
        N0_est = N0_est.repeat(1,1,self.M);
        x = torch.div(x,N0_est);
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        m = x
        var = torch.div(var, N0_est**2);
        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w)
        return x, m, var
    
    def forward_N0_reconstruct(self, x, b, c, h, w):
        x, m, var = self.forward_N0_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w,m, var)
        return x
 
    def forward_stat_comp(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = x/self.N0;
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 

        m = x
        var = var/(self.N0**2)
        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x,m,var
     
 

#==============================================================================    
# B. NOISY MEASUREMENTS (NOISE LEVEL IS VARYING) + denoising architecture
#==============================================================================
class DenoiCompNet(noiCompNet):
    def __init__(self, n, M, Mean, Cov, variant, N0, sig = 0.1, H=None, mean_denoi=False,denoi = None):
        super().__init__(n, M, Mean, Cov, variant, N0, sig, H,denoi)
        print("Denoised Measurements")
   
    def forward_maptoimage(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = x/self.N0;
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        m = x
        var = var/(self.N0**2);
        
        #--Denoising 
        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma+var), x);
        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x, m, var
    
    def forward_maptoimage_2(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = x/self.N0;
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        
        m = x;
        var = var/(self.N0**2);

        #--Denoising 
        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        mu_1 = self.mu_1.repeat(b*c,1,1).to(x.device);
        x = mu_1 + torch.mul(torch.div(sigma, sigma+var), x-mu_1);
        self.x0 = x
        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x,m, var
     
    def forward_denoised_Pinv(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = x/self.N0;
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        
        m = x;
        var = var/(self.N0**2)

        #--Denoising 
        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma+var), x);

        #--Projection to the image domain
        x = self.Pinv(x);
        x = x.view(b*c,1,h,w) 
        return x, m, var
   
    def forward_reconstruct(self, x, b, c, h, w):
        x, m, var = self.forward_maptoimage(x, b, c, h, w);
        x = self.forward_postprocess(x, b, c, h, w, m, var)
        return x

    def forward_NO_maptoimage(self, x, b, c, h, w):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = x[:,:,self.even_index] + x[:,:,self.uneven_index];
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        
        #-- Pre-processing(Estimating No and normalizing by No) 
        x_est = self.Pinv(x);
        x_est = x_est.view(b*c,1,h,w);
        N0_est = self.max(x_est);
        N0_est = N0_est.view(b*c,1,1);
        N0_est = N0_est.repeat(1,1,self.M);
        x = torch.div(x,N0_est);
         
        var = torch.div(var, N0_est**2);
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
        m = x;

        #--Denoising 
        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma+var), x);

        
        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x, m, var;

    def forward_N0_maptoimage_expe(self, x, b, c, h, w, C, s, g):
        #-- Pre-processing(Recombining positive and negatve values+normalisation) 
        var = g**2*(x[:,:,self.even_index] + x[:,:,self.uneven_index]) - 2*C*g +2*s**2;
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        
        #-- Pre-processing(Estimating No and normalizing by No) 
        x_est = self.Pinv(x);
        x_est = x_est.view(b*c,1,h,w);
        N0_est = self.max(x_est);
        N0_est = N0_est.view(b*c,1,1);
        N0_est = N0_est.repeat(1,1,self.M);
        x = torch.div(x,N0_est);
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b*c,1,self.M)); 
         
        m = x;
        var = torch.div(var, N0_est**2)

        #--Denoising 
        sigma = self.sigma.repeat(b*c,1,1).to(x.device);
        x = torch.mul(torch.div(sigma, sigma+var), x);


        #--Projection to the image domain
        x = self.fc1(x);
        x = x.view(b*c,1,h,w) 
        return x, m, var;    


    
    def forward_N0_reconstruct_expe(self, x, b, c, h, w,C,s,g):
        x, m, var = self.forward_N0_maptoimage_expe(x, b, c, h, w,C,s,g)
        x = self.forward_postprocess(x, b, c, h, w, m, var)
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


