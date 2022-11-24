# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from abc import ABC, abstractmethod
import pywt


########################################################################
# 1. Define Abstract Pattern Class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# All Pattern Classes (Basis, Custom and optimized will)
# inherit this abstract class. The only real abstract 
# method being the set_desired_patterns, and add_
# desired pattern.

class Patterns(ABC):
    def __init__(self,img_size,method ='split',binarized=False, par=2, lvl = 1, dyn = 8):
        super().__init__();
        self.n = img_size;
        self.method = method;
        self.binarized = binarized;
        self.par=par;
        self.lvl =lvl;
        self.dyn = dyn;
        self.Q =nn.Conv2d(1,img_size,kernel_size=img_size, stride=1, padding=0)
        self.T = np.zeros((img_size,2*img_size));
        self.P =nn.Conv2d(1,2*img_size,kernel_size=img_size, stride=1, padding=0)
        self.steps = [0];

    def get_desired_pattern(self):
        return self.Q[self.start,1,:,:]

    def get_all_desired_pattern(self):
        return self.Q

    def get_measurement_matrix(self):
        return self.P, self.T

    def set_measurement_matrix(self):
        (next_P, next_T)=eval(self.method+'(self.Q, self.dyn)');
        self.P = next_P;
        self.T = next_T;

    def save_measurement_matrix(self, root):
        K = self.P.bias.shape[0];
        np.save(root+'T.npy', self.T);
        for i in range(K):
            pat = self.P.weight[i,0,:,:];
            pattern = pat.cpu().detach().numpy();

            cv2.imwrite(root+'pat_{}x{}'.format(i)+'.png', pattern);

    @abstractmethod
    def set_desired_pattern(self,def_matrix):
        pass

    @abstractmethod
    def add_desired_patterns(self, def_matrix):
        pass
 
########################################################################
# 2. Define children of that abstract class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# All Pattern Classes (Basis, Custom and optimized will)
# inherit the abstract class Patterns and implement the two main 
# abstract methods
# 

   
class Basis_patterns(Patterns):
    def __init__(self, img_size, basis, method='split',binarized=False):
        super(Basis_patterns,self).__init__(self, img_size,method='split', binarized=False);
        self.basis = basis;
        self.img_size = img_size;
        self.indexes=np.zeros((img_size, img_size));
        self.cumulated_indexes=np.zeros((img_size, img_size));

    def add_desired_pattern(self, def_matrix):
        temp = np.ones((self.img_size, self.img_size));
        self.indexes += def_matrix;
        self.cumulated += def_matrix +  temp(self.cumulated>0);
        
        I_old = self.Q.bias.shape[0];
        Q = eval(self.basis+'(def_matrix, self.par, self.lvl)');
        I_add = int(np.sum(def_matrix));
        I_new = I_old + I_add;

        next_Q =nn.Conv2d(1,I_new,kernel_size=img_size, stride=1, padding=0);
        next_Q.bias = torch.zeros(I_new);
        next_Q.weight[:I_old, :,:,:] = self.Q.weight;
        next_Q.weight[I_old:,:,:,:] = Q.weight;
        next_Q.bias.requires_grad = False;
        next_Q.weight.requires_grad=False;

        self.Q = next_Q;
        self.steps.append(I_old)

    def set_desired_pattern(self, def_matrix):
        self.indexes = def_matrix;
        self.cumulated = def_matrix ;
        
        next_Q = eval(basis+'(def_matrix, self.par, self.lvl)');

        self.Q = next_Q;
        self.steps = [0];
   
class Custom_patterns(Patterns):
    def __init__(self, img_size,Q, method='split',binarized=False):
        super(Basis_patterns,self).__init__(self, img_size,method='split', binarized=False);
        self.img_size = img_size;
        self.Q = Q;

    def add_desired_pattern(self, Q):
        I_old = self.Q.bias.shape[0];
        I_add = Q.bias.shape[0];
        I_new = I_old + I_add;

        next_Q =nn.Conv2d(1,I_new,kernel_size=img_size, stride=1, padding=0);
        next_Q.bias = torch.zeros(I_new);
        next_Q.weight[:I_old, :,:,:] = self.Q.weight;
        next_Q.weight[I_old:,:,:,:] = Q.weight;
        next_Q.bias.requires_grad = False;
        next_Q.weight.requires_grad=False;

        self.Q = next_Q;
        self.steps.append(I_old)

    def set_desired_pattern(self, Q):
        self.Q =Q;
        self.steps = [0];
   
class Optimized_patterns(Patterns):
    def __init__(self, img_size, basis,M, method='split',binarized=False):
        super(Basis_patterns,self).__init__(self, img_size,method='split', binarized=False);
        self.basis = basis;
        self.img_size = img_size;
        self.M = M;
        
    def add_desired_pattern(self, M_prim):
        
        I_old = self.Q.bias.shape[0];
        Q = eval(self.basis+'_opt(M_prim, self.par, self.lvl)');
        I_add = M_prim;
        I_new = I_old + I_add;

        next_Q =nn.Conv2d(1,I_new,kernel_size=img_size, stride=1, padding=0);
        next_Q.bias = torch.zeros(I_new);
        next_Q.weight[:I_old, :,:,:] = self.Q.weight;
        next_Q.weight[I_old:,:,:,:] = Q.weight;
        next_Q.bias.requires_grad = False;
        next_Q.weight.requires_grad=False;

        self.Q = next_Q;
        self.steps.append(I_old)

    def set_desired_pattern(self, M):
        
        next_Q = eval(basis+'_opt(M, self.par, self.lvl)');

        self.Q = next_Q;
        self.steps = [0];
   


########################################################################
# 3. Define functions for basis pattern
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Return a convolution filter that contains all the basis
# functions of a given transform.
# 


def Fourier(def_matrix, par=0, lvl=1):
    I = int(np.sum(def_matrix));
    (nx, ny) = def_matrix.shape;
    Q =nn.Conv2d(1,2*I,kernel_size=nx, stride=1, padding=0);
    Q.bias.data=torch.zeros(2*I);
    ind = np.nonzero(def_matrix);
    Z=np.zeros((nx, ny));
    for index in range(I):
        i=ind[0][index];
        j=ind[1][index];
        Z[i,j]= 1;
        pat = cv2.dft(np.float32(Z),flags = cv2.DFT_COMPLEX_OUTPUT);
        pat_real = pat[:,:,0];
        pat_img = pat[:,:,1];
        Z[i,j]= 0;
        Q.weight.data[2*index, 0, :,:] = pat_real;
        Q.weight.data[2*index+1, 0, :,:] = pat_img;
    Q.bias.requires_grad = False;
    Q.weight.requires_grad=False;

    return Q

def Hadamard(def_matrix, par=0, lvl=1):
    I = int(np.sum(def_matrix));
    (nx, ny) = def_matrix.shape;

    Q = nn.Conv2d(1,I,kernel_size=nx, stride=1, padding=0)
    Q.bias.data=torch.zeros(I);

    ind = np.nonzero(def_matrix);
    Z=np.zeros((nx, ny));

    for index in range(I):
        i=ind[0][index];
        j=ind[1][index];
        Z[i,j]= 1;
        #pat = torch.from_numpy(fht2(Z));

        Z[i,j]= 0;
        Q.weight.data[index, 0, :,:] = pat;
    Q.bias.requires_grad = False;
    Q.weight.requires_grad=False;

    return Q

def Haar(def_matrix, par=0, lvl=1):
    I = int(np.sum(def_matrix));
    (nx, ny) = def_matrix.shape;
    wave = 'Haar'
    md='periodization'

    Q = nn.Conv2d(1,I,kernel_size=nx, stride=1, padding=0)
    Q.bias.data=torch.zeros(I);

    ind = np.nonzero(def_matrix);
    Z=np.zeros((nx, ny));

    for index in range(I):
        i=ind[0][index];
        j=ind[1][index];
        Z[i,j]= 1;

        pat_coefs = pywt.wavedec2(Z,wave,mode=md,level=lvl)
        [temp, arr] = pywt.coeffs_to_array(pat_coefs);
        
        pat_temp = pywt.array_to_coeffs(Z, arr,output_format='wavedec2');
        pat = pywt.waverec2(pat_temp,wave);
        pat = torch.from_numpy(pat);

        Z[i,j]= 0;
        Q.weight.data[index, 0, :,:] = pat;
    Q.bias.requires_grad = False;
    Q.weight.requires_grad=False;

    return Q


def Daubechies(def_matrix, par=2, lvl = 1):
    I = int(np.sum(def_matrix));
    (nx, ny) = def_matrix.shape;
    wave = 'db'+str(par)
    md='periodization'

    Q = nn.Conv2d(1,I,kernel_size=nx, stride=1, padding=0)
    Q.bias.data=torch.zeros(I);

    ind = np.nonzero(def_matrix);
    Z=np.zeros((nx, ny));

    for index in range(I):
        i=ind[0][index];
        j=ind[1][index];
        Z[i,j]= 1;

        pat_coefs = pywt.wavedec2(Z,wave,mode=md,level=lvl)
        [temp, arr] = pywt.coeffs_to_array(pat_coefs);
        
        pat_temp = pywt.array_to_coeffs(Z, arr,output_format='wavedec2');
        pat = pywt.waverec2(pat_temp,wave);
        pat = torch.from_numpy(pat);

        Z[i,j]= 0;
        Q.weight.data[index, 0, :,:] = pat;
    Q.bias.requires_grad = False;
    Q.weight.requires_grad=False;

    return Q


########################################################################
# 3. Define functions for optimized basis pattern
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Return a convolution filter that contains all the basis
# funtions obtained through a training phase.
# 

def Fourier_opt(M, par=0, lvl=1):
    I = int(np.sum(def_matrix));
    (nx, ny) = def_matrix.shape;
    Q =nn.Conv2d(1,2*I,kernel_size=nx, stride=1, padding=0);
    Q.bias.data=torch.zeros(2*I);
    ind = np.nonzero(def_matrix);
    Z=np.zeros((nx, ny));
    for index in range(I):
        i=ind[0][index];
        j=ind[1][index];
        Z[i,j]= 1;
        pat = cv2.dft(np.float32(Z),flags = cv2.DFT_COMPLEX_OUTPUT);
        pat_real = pat[:,:,0];
        pat_img = pat[:,:,1];
        Z[i,j]= 0;
        Q.weight.data[2*index, 0, :,:] = pat_real;
        Q.weight.data[2*index+1, 0, :,:] = pat_img;
    Q.bias.requires_grad = False;
    Q.weight.requires_grad=False;

    return Q

def Hadamard_opt(M, par=0, lvl=1):
    I = int(np.sum(def_matrix));
    (nx, ny) = def_matrix.shape;

    Q = nn.Conv2d(1,I,kernel_size=nx, stride=1, padding=0)
    Q.bias.data=torch.zeros(I);

    ind = np.nonzero(def_matrix);
    Z=np.zeros((nx, ny));

    for index in range(I):
        i=ind[0][index];
        j=ind[1][index];
        Z[i,j]= 1;
        #pat = torch.from_numpy(fht2(Z));

        Z[i,j]= 0;
        Q.weight.data[index, 0, :,:] = pat;
    Q.bias.requires_grad = False;
    Q.weight.requires_grad=False;

    return Q

def Haar_opt(M, par=0, lvl=1):
    I = int(np.sum(def_matrix));
    (nx, ny) = def_matrix.shape;
    wave = 'Haar'
    md='periodization'

    Q = nn.Conv2d(1,I,kernel_size=nx, stride=1, padding=0)
    Q.bias.data=torch.zeros(I);

    ind = np.nonzero(def_matrix);
    Z=np.zeros((nx, ny));

    for index in range(I):
        i=ind[0][index];
        j=ind[1][index];
        Z[i,j]= 1;

        pat_coefs = pywt.wavedec2(Z,wave,mode=md,level=lvl)
        [temp, arr] = pywt.coeffs_to_array(pat_coefs);
        
        pat_temp = pywt.array_to_coeffs(Z, arr,output_format='wavedec2');
        pat = pywt.waverec2(pat_temp,wave);
        pat = torch.from_numpy(pat);

        Z[i,j]= 0;
        Q.weight.data[index, 0, :,:] = pat;
    Q.bias.requires_grad = False;
    Q.weight.requires_grad=False;

    return Q


def Daubechies_opt(M, par=2, lvl = 1):
    I = int(np.sum(def_matrix));
    (nx, ny) = def_matrix.shape;
    wave = 'db'+str(par)
    md='periodization'

    Q = nn.Conv2d(1,I,kernel_size=nx, stride=1, padding=0)
    Q.bias.data=torch.zeros(I);

    ind = np.nonzero(def_matrix);
    Z=np.zeros((nx, ny));

    for index in range(I):
        i=ind[0][index];
        j=ind[1][index];
        Z[i,j]= 1;

        pat_coefs = pywt.wavedec2(Z,wave,mode=md,level=lvl)
        [temp, arr] = pywt.coeffs_to_array(pat_coefs);
        
        pat_temp = pywt.array_to_coeffs(Z, arr,output_format='wavedec2');
        pat = pywt.waverec2(pat_temp,wave);
        pat = torch.from_numpy(pat);

        Z[i,j]= 0;
        Q.weight.data[index, 0, :,:] = pat;
    Q.bias.requires_grad = False;
    Q.weight.requires_grad=False;

    return Q



 
########################################################################
# 4. Define functions for pattern splitting and shifting
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Implements most used splitting and shifting models
# 


def split(Q, dyn):
    I = Q.bias.shape[0];
    img_size = Q.weight.shape[-1]
    P =  nn.Conv2d(1,2*I,kernel_size=img_size, stride=1, padding=0)
    T_matrix = torch.zeros((I, 2*I))
    T = nn.Linear(2*I, I, bias = False)
    
    for i in range(I):
        pat = Q.weight.data[i,0,:,:];
        pat = pat.cpu().detach().numpy();
        pat_pos = np.zeros((img_size, img_size));
        pat_neg = np.zeros((img_size, img_size));
        pat_pos[pat>0]=pat[pat>0];
        pat_neg[pat<0]= - pat[pat<0];
        
        max_pos = np.max(pat_pos);
        max_neg = np.max(pat_neg);


        pat_pos = (2**dyn-1)*pat_pos/max_pos if max_pos!=0 else pat_pos;
        pat_neg = (2**dyn-1)*pat_neg/max_neg if max_neg!=0 else pat_neg;

        T_matrix[i, 2*i] = max_pos/(2**dyn-1);
        T_matrix[i, 2*i+1] = -max_neg/(2**dyn-1);

        
        P.weight.data[2*i,0,:,:] = torch.from_numpy(pat_pos);
        P.weight.data[2*i+1,0,:,:] = torch.from_numpy(pat_neg);
    P.bias.requires_grad = False;
    P.weight.requires_grad = False;

    T.weight.data = T_matrix;
    T.weight.data = T.weight.data.float();
    T.weight.requires_grad = False;

    return P,T



def shift(Q, dyn):
    I = Q.bias.shape[0];
    img_size = Q.weight.shape[-1];
    P =  nn.Conv2d(1,I+1,kernel_size=img_size, stride=1, padding=0);
    T = np.zeros((I, I+1));
    
    for i in range(I):
        pat = Q.weight.data[i,0,:,:];
        pat = pat.cpu().detach().numpy();
        dc_val = np.min(pat);

        pat_ac = pat - dc_val;

        max_ac = np.max(pat_ac);
        pat_ac = (2**dyn-1)*pat_ac/max_ac;

        T[i,i] = max_ac/(2**dyn-1);
        T[i,I+1] = -max_ac/(2**dyn-1);

        P.weight.data[i,0,:,:] = torch.from_numpy(pat_ac);
    P.weight.data[I,0,:,:] = (2**dyn-1)*torch.ones(img_size, img_size);
    P.bias.requires_grad = False;
    P.weight.requires_grad=False;

    return P,T




def matrix2conv(Matrix):
    """
        Returns Convulution filter che each kernel correponds to a line of
        Matrix, that has been reshaped
    """
    M, N = Matrix.shape;
    img_size =int(round(np.sqrt(N)));

    P = nn.Conv2d(1,M,kernel_size=img_size, stride=1, padding=0);
    P.bias.data=torch.zeros(M);
    for i in range(M):
        pattern = np.reshape(Matrix[i,:],(img_size, img_size));
        P.weight.data[i,0,:,:] = torch.from_numpy(pattern);
    P.bias.requires_grad = False;
    P.weight.requires_grad=False;
    return P

































