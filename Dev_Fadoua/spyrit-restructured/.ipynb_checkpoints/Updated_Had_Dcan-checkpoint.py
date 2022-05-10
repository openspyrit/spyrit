" ==================================================================================
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
import spyrit.misc.walsh_hadamard as wh
from spyrit.misc.statistics import *
import math
from torch import poisson

" ==================================================================================
class Hadamard:
" ==================================================================================
#                         Define Best Hadamard Matrix
" ==================================================================================
    " Inputs:
    "----------------------------------------------------------
    # root:       Path to Hadamard patterns database 
    # (nx,ny):    Image size 
    # M:          Sub-sampling threshold for Hadamard patterns
    " ---------------------------------------------------------
    " Output:
    " ---------------------------------------------------------
    # conv:    Best suited Hadamard pattern Matrix given image size (nx,ny)
" ==============================================================================     
    def __init(self, root, nx, ny, M):
        self.nx = nx
        self.ny = ny
        
        # load Hadamard Matrix
        had_mat = np.load(root+'{}x{}'.format(nx,ny)+'.npy')
        
        # reshape Hadamard matrix cells based on ordinal ranking scores
        had_comp = np.reshape(rankdata(-had_mat, method = 'ordinal'),(nx, ny))
        
        # Sub-sampling
        msk[np.absolute(had_comp) > M] = 0
        self.conv = had_mat(msk) # had_mat au lieu de Hadamard
        
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

    """Generation of a Gray permutation matrix"""


    def conv_list_b2_b10(l): # convert a liste of number in the base 2 to the base 10
        N = len(l)
        for i in range(N):
            l[i] = int(l[i],2)
        return(l)

    def Mat_of_ones_from_list_index(l): # generate a matrix of zero and ones from list of index
        N = len(l)
        M_out = np.zeros((N,N))
        for i in range(N):
            M_out[i,l[i]] = 1
        return(M_out)

    def gray_code_permutation(n): # Generate the N grey code permutation matrix
        N = int(math.log(n, 2))
        graycode = GrayCode(N)
        graycode_list = list(graycode.generate_gray())
        return(Mat_of_ones_from_list_index(conv_list_b2_b10((graycode_list))))

    """Generation of a bit permutation matrix"""

    def bit_reverse_traverse(a): #internet function to generate bit reverse
        n = a.shape[0]
        assert(not n&(n-1) ) # assert that n is a power of 2

        if n == 1:
            yield a[0]
        else:
            even_index = np.arange(n//2)*2
            odd_index = np.arange(n//2)*2 + 1
            for even in bit_reverse_traverse(a[even_index]):
                yield even
            for odd in bit_reverse_traverse(a[odd_index]):
                yield odd

    def get_bit_reversed_list(l): #internet function to generate bit reverse
        n = len(l)

        indexs = np.arange(n)
        b = []
        for i in bit_reverse_traverse(indexs):
            b.append(l[i])

        return b

    def bit_reverse_matrix(n):#internet function to generate bit reverse
        l_br = get_bit_reversed_list([k for k in range(n)])
        Mat_out = np.zeros((n,n))

        for i in range(n):
            Mat_out[i,l_br[i]] = 1
        return(Mat_out)


    def walsh_matrix(n): 
        """Return 1D Walsh-ordered Hadamard transform matrix

        Args:
            n (int): Order of the matrix, a power of two.

        Returns:
            np.ndarray: A n-by-n array

        Examples:
            Walsh-ordered Hadamard matrix of order 8

            >>> print(walsh_matrix(8))
        """
        BR = bit_reverse_matrix(n)
        GRp = gray_code_permutation(n)
        H = hadamard(n)
        return(np.dot(np.dot(GRp,BR),H)) # Apply permutation to the hadmard matrix 


    def walsh2(X,H=None):
        r"""Return 2D Walsh-ordered Hadamard transform of an image :math:`H^\top X H`

        Args:
            X (np.ndarray): image as a 2d array. The size is a power of two.
            H (np.ndarray, optional): 1D Walsh-ordered Hadamard transformation matrix

        Returns:
            np.ndarray: Hadamard transformed image as a 2D array.
        """
        if H is None:
             H = walsh_matrix(len(X))
        return(np.dot(np.dot(H,X),H))


    def iwalsh2(X,H=None):
        """Return 2D inverse Walsh-ordered Hadamard transform of an image

        Args:
            X (np.ndarray): Image as a 2D array. The image is square and its size is a power of two.
            H (np.ndarray, optional): 1D inverse Walsh-ordered Hadamard transformation matrix

        Returns:
            np.ndarray: Inverse Hadamard transformed image as a 2D array.
        """
        if H is None:
             H = walsh_matrix(len(X))
        return(walsh2(X,H)/len(X)**2)

    def walsh2_matrix(n):
        """Return 2D Walsh-ordered Hadamard transformation matrix

        Args:
            n (int): Order of the matrix, which should be a power of two.

        Returns:
            H (np.ndarray): A n*n-by-n*n array
        """
        H = np.zeros((n**2, n**2))
        H1d = walsh_matrix(n)
        for i in range(n**2):
            image = np.zeros((n**2,1));
            image[i] = 1;
            image = np.reshape(image, (n, n));
            hadamard = walsh2(image, H1d);
            H[:, i] = np.reshape(hadamard, (1,n**2));
        return H

    def walsh2_torch(im,H=None):
        """Return 2D Walsh-ordered Hadamard transform of an image

        Args:
            im (torch.Tensor): Image, typically a B-by-C-by-W-by-H Tensor
            H (torch.Tensor, optional): 1D Walsh-ordered Hadamard transformation matrix. A 2-D tensor of size W-by-H.

        Returns:
            torch.Tensor: Hadamard transformed image. Same size as im

        Examples:
            >>> im = torch.randn(256, 1, 64, 64)
            >>> had = walsh2_torch(im)
        """
        if H is None:
             H = torch.from_numpy(walsh_matrix(im.shape[3]).astype('float32'))
        H = H.to(im.device)
        return  torch.matmul(torch.matmul(H,im),H)
    
    # Define Pconv here (imported from pattern_choice.py
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
            # Creates a Tensor from a numpy.ndarray 
            P.weight.data[i,0,:,:] = torch.from_numpy(pattern);
        P.bias.requires_grad = False;
        P.weight.requires_grad=False;
        return P
# 
" ==================================================================================        
class inutile: # ça va aller dans le main
" ==================================================================================
    def __init__(self, H, n, M):
        
        self.H = H        
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
        
        #-- Denoising parameters taken in the preprocess 
        Sigma = np.dot(Perm,np.dot(Cov,np.transpose(Perm)));
        diag_index = np.diag_indices(n**2);
        Sigma = Sigma[diag_index];
        Sigma = n**2/4*Sigma[:M];   # Multiplication by n**2 as H <- nH  leads to Cov <- n**2 Cov 
                                    # Division by 4 to get the covariance of images in [0 1], not [-1, 1]
        Sigma = torch.Tensor(Sigma)
        self.sigma = Sigma.view(1,1,M)
        self.sigma.requires_grad = False  
        
        #-- Measurement preprocessing
        self.Pmat = Pmat
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
        
    def forward_preprocess(self, x, b, c, h, w):
        #- Pre-processing (use batch norm to avoid division by N0 ?)
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index];
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1,h,w).to(x.device)),(b,c,self.M));
        return x   
    
" ==================================================================================
class Forward_operator(nn.Module):
" ==================================================================================
# Faire le produit H*f sans bruit, linear (pytorch) 
    def __init__(self, Pmat):           
        # récupérer la méthode __init__ de la classe parent nn.Module et créer un objet de cette classe
        super().__init__()
        
        # instancier nn.linear        
        # Pmat --> (torch) --> Poids ()
        self.Pmat = nn.linear(self.Pmat.shape[1], self.Pmat.shape[0], False) # False dit que le biais est nul
        self.Pmat.weight.data=torch.from_numpy(Pmat)
        self.Pmat.weight.data=self.Pmat.weight.data.float()
        self.Pmat.weight.requires_grad=False
        
        # adjoint
        self.Pmat_adjoint = nn.linear(self.Pmat.shape[0], self.Pmat.shape[1], False)
        self.Pmat_adjoint.weight.data=torch.from_numpy(Pmat.transpose())
        self.Pmat_adjoint.weight.data = self.Pmat_adjoint.weight.data.float()
        self.Pmat_adjoint.weight.requires_grad = False
               
    def forward(self, x) # --> simule la mesure sous-chantillonnée
        # Pmat*f
        self.x = self.Pmat(x)    
        return
    
    def adjoint(self,x):
         #Pmat.transpose()*f
        self.x = self.Pmat_adjoint(x)            
        return  
    
# renommer Pmat en Hsub --> Hadamard subsampled, et l'adjoint Hsub_T

" ==================================================================================        
class Acquisition(nn.Module):
" ==================================================================================
#                      Forward multiply with Hadamard matrix H * f
" ==================================================================================
# exemple:
# on calcule Pmat --> on écrit H = Forward_operator(Pmat)
# acq = Acquisition(H)
# avec une image x, m = acq.forward(x) (ce qui est équivalent à m = acq(x)) Syntax Pytorch

    def __init__(self, H):
        super().__init__()
        # H is the Hadamard Matrix
        self.H = H
           
    
    def forward(self, x, b, c, h, w):
        
        " x is a torch input image dataset " 
        " b,c,h,w = x.shape  --> b is the number of images, c = 1, h is the number of row pixels, w is the number of column pixels "
        
        #--Scale input image
        x = (x+1)/2; 
        
        #--Acquisition
        x = x.view(b*c, 1, h, w);
        
        x = H(x); # pareil que H.forward(x)
        
        # F importé depuis torch.nn.functional
        x = F.relu(x); ## x[:,:,1] = -1/N0 ????
        
        print("No noise")
        x = x.view(b, c, 2*self.M); 
        
        # x is the product of Hsub-sampled*f ?
        return x
" ==================================================================================
class Bruit_Poisson_approx_Gauss(Acquisition):
" ==================================================================================    
    def __init__(self, alpha, H):
        super().__init__(H)
        self.alpha = alpha
        
    def forward(x, b, c, h, w):

        #--Scale input image      

        x = self.alpha*(x+1)/2;
        
        #--Acquisition
        x = x.view(b*c, 1, h, w);
        x = self.H(x);
        # par rapport aux potentielles faibles valeurs négatives
        x = F.relu(x);     # x[:,:,1] = -1/N0 ????
        x = x.view(b, c, 2*self.M); # x[:,:,1] < 0??? 
        
        #--Measurement noise (Gaussian approximation of Poisson)
        x = x + torch.sqrt(x)*torch.randn_like(x);  
        return x        
" ==================================================================================
class Bruit_Poisson_Pytorch(Acquisition):
" ==================================================================================           
     def __init__(self, alpha, H):
        super().__init__(H)
        self.alpha = alpha
        
    def forward(x, b, c, h, w):

        #--Scale input image      
        x = self.alpha*(x+1)/2;
        
        #--Acquisition
        x = x.view(b*c, 1, h, w);
        x = self.H(x);
        # par rapport aux potentielles faibles valeurs négatives
        x = F.relu(x);     # x[:,:,1] = -1/N0 ????
        x = x.view(b, c, 2*self.M); # x[:,:,1] < 0??? 
        
        #--Measurement noise imported from Pytorch
        x = poisson(x) 
        return x           
    
" ==================================================================================
class Pinv(nn.Module):
" ==================================================================================
    def __init__(self, Pmat):
        super().__init__()
        self.Pmat = Pmat
        #-- Pseudo-inverse to determine levels of noise.
        Pinv = (1/Pmat.shape[1])*np.transpose(self.Pmat)
        self.Pinv = nn.Linear(Pmat.shape[0],Pmat.shape[1], False)
        self.Pinv.weight.data=torch.from_numpy(Pinv)
        self.Pinv.weight.data=self.Pinv.weight.data.float()
        self.Pinv.weight.requires_grad=False
        
    def forward(x):
        # attention à faire un reshape adéquat à l'extérieur de la classe
        # il faut documenter le bon shape de x
        # x = x.view(b*c, 1, self.M)
        x = self.Pinv(x)
        return x
        
" ===========================================================================================
class Generalized_Orthogonal_Tikhonov(nn.Module, preprocess, Pinv):# permet de revenir à l'espace image n*n
" ===========================================================================================   
# 
    def __init__(self, H, n, M, variant):
        nn.Module.__init__(self)
        preprocess.__init__(self, H, n, M)
        Pinv.__init__(self, H, n, M)
        
        #-- Measurement to image domain
        if variant==0:
            #--- Statistical Matrix completion (no mean)
            print("Measurement to image domain: statistical completion (no mean)")
            
            self.fc1 = nn.Linear(M,n**2, False)
            
            W, b, mu1 = stat_completion_matrices(Perm, H, Cov, Mean, M)
            W = (1/n**2)*W

            self.fc1.weight.data=torch.from_numpy(W)
            self.fc1.weight.data=self.fc1.weight.data.float()
            self.fc1.weight.requires_grad=False
        
        if variant==1:
            #--- Statistical Matrix completion  
            print("Measurement to image domain: statistical completion")
            
            self.fc1 = nn.Linear(M,n**2)
            
            W, b, mu1 = stat_completion_matrices(Perm, H, Cov, Mean, M)
            W = (1/n**2)*W
            b = (1/n**2)*b
            b = b - np.dot(W,mu1)
            self.fc1.bias.data=torch.from_numpy(b[:,0])
            self.fc1.bias.data=self.fc1.bias.data.float()
            self.fc1.bias.requires_grad = False
            self.fc1.weight.data=torch.from_numpy(W)
            self.fc1.weight.data=self.fc1.weight.data.float()
            self.fc1.weight.requires_grad=False
        
        elif variant==2:
            #--- Pseudo-inverse
            print("Measurement to image domain: pseudo inverse")           
            self.fc1 = self.Pinv
            
        elif variant==3:
            #--- FC is learnt
            print("Measurement to image domain: free")
            
            self.fc1 = nn.Linear(M,n**2)
            
    def forward(self, x, b, c, h, w):
        #--Projection to the image domain
        x = self.fc1(x)
        x = x.view(b, c, h, w)
    return x     

" ==================================================================================        
class Preprocess(nn.Module): # ça va aller dans le main
" ==================================================================================
    def __init__(self, n, M, N0):
        super().__init()
        self.N0 = N0
        self.M = M
        vector = np.zeros(M)
        vector[0] = n**2
        self.Patt = torch.from_numpy(vector)
        
    def forward(x, b, c, h, w):
        # x = x.view(b*c, 1, 2*self.M)
        x = x[:,:,self.even_index] - x[:,:,self.uneven_index]
        x = x/self.N0
        x = 2*x-torch.reshape(self.Patt(torch.ones(b*c,1, h,w).to(x.device)),(b,c,self.M))   
        return x
    
    def sigma(x):
        # x = x.view(b*c, 1, 2*self.M)
        x = x[:,:,self.even_index] + x[:,:,self.uneven_index]
        x = 4*x/self.N0**2
        return x