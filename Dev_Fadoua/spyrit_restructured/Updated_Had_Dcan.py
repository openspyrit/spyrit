# ==================================================================================
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
# from ..misc.pattern_choice import Hadamard, matrix2conv, split
from collections import OrderedDict
#import cv2
from scipy.stats import rankdata
#from itertools import cycle;
#from pathlib import Path

# from ..misc.disp import *
# import spyrit.misc.walsh_hadamard as wh
# from spyrit.misc.statistics import *
import math
from torch import poisson

# ==================================================================================
class Forward_operator(nn.Module):
# ==================================================================================
# Faire le produit H*f sans bruit, linear (pytorch) 
    def __init__(self, Hsub):           
        # récupérer la méthode __init__ de la classe parent nn.Module et créer un objet de cette classe
        super().__init__()
        
        # instancier nn.linear        
        # Pmat --> (torch) --> Poids ()
        self.M = Hsub.shape[0];
        self.N = Hsub.shape[1] ;
        self.Hsub = nn.Linear(self.N, self.M, False) # False dit que le biais est nul
        self.Hsub.weight.data=torch.from_numpy(Hsub)
        self.Hsub.weight.data=self.Hsub.weight.data.float()
        self.Hsub.weight.requires_grad=False
        
        # adjoint
        self.Hsub_adjoint = nn.Linear(self.M, self.N, False)
        self.Hsub_adjoint.weight.data=torch.from_numpy(Hsub.transpose())
        self.Hsub_adjoint.weight.data = self.Hsub_adjoint.weight.data.float()
        self.Hsub_adjoint.weight.requires_grad = False
               
    def forward(self, x): # --> simule la mesure sous-chantillonnée
        # x.shape[b*c,N]
        x = self.Hsub(x)    
        return x
    
    def adjoint(self,x):
        # x.shape[b*c,M]
        #Pmat.transpose()*f
        x = self.Hsub_adjoint(x)            
        return x

# ==================================================================================        
class Acquisition(nn.Module):
# ==================================================================================
#                      Forward multiply with Hadamard matrix H * f
# ==================================================================================
# exemple:
# on calcule Pmat --> on écrit H = Forward_operator(Pmat)
# acq = Acquisition(H)
# avec une image x, m = acq.forward(x) (ce qui est équivalent à m = acq(x)) Syntax Pytorch

    def __init__(self, FO):
        super().__init__()
        # FO = forward operator
        self.FO = FO
    
    def forward(self, x):
        # input x.shape - [b*c,h*w] - [b*c,N] 
        # output x.shape - [b*c,M] 
        #--Scale input image
        x = (x+1)/2;         
        x = FO(x); 
        # x is the product of Hsub-sampled*f ?
        return x
# ==================================================================================
class Bruit_Poisson_approx_Gauss(Acquisition):
# ==================================================================================    
    def __init__(self, alpha, FO):
        super().__init__(FO)
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
    
# ==================================================================================
class Bruit_Poisson_Pytorch(Acquisition):
# ==================================================================================           
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
    
# ==================================================================================
class Pinv(nn.Module):
# ==================================================================================
    def __init__(self, FO):
        super().__init__()
        # FO = Forward Operator
        #-- Pseudo-inverse to determine levels of noise.
        self.FO = FO;
        
    def forward(self, x):
        # attention à faire un reshape adéquat à l'extérieur de la classe
        # il faut documenter le bon shape de x
        # x = x.view(b*c, M)
        x = (1/self.FO.N)*self.FO.adjoint(x);
        return x
        
# ===========================================================================================
class Generalized_Orthogonal_Tikhonov(nn.Module):# permet de revenir à l'espace image n*n
# ===========================================================================================   
# 
    def __init__(self, H, n, M, variant):
        nn.Module.__init__(self)
        Pinv.__init__(self, H, n, M)
        
        #-- Measurement to image domain
#         if variant==0:
#             #--- Statistical Matrix completion (no mean)
#             print(#Measurement to image domain: statistical completion (no mean)#)
            
#             self.fc1 = nn.Linear(M,n**2, False)
            
#             W, b, mu1 = stat_completion_matrices(Perm, H, Cov, Mean, M)
#             W = (1/n**2)*W

#             self.fc1.weight.data=torch.from_numpy(W)
#             self.fc1.weight.data=self.fc1.weight.data.float()
#             self.fc1.weight.requires_grad=False
        
#         if variant==1:
#             #--- Statistical Matrix completion  
#             print(#Measurement to image domain: statistical completion#)
            
#             self.fc1 = nn.Linear(M,n**2)
            
#             W, b, mu1 = stat_completion_matrices(Perm, H, Cov, Mean, M)
#             W = (1/n**2)*W
#             b = (1/n**2)*b
#             b = b - np.dot(W,mu1)
#             self.fc1.bias.data=torch.from_numpy(b[:,0])
#             self.fc1.bias.data=self.fc1.bias.data.float()
#             self.fc1.bias.requires_grad = False
#             self.fc1.weight.data=torch.from_numpy(W)
#             self.fc1.weight.data=self.fc1.weight.data.float()
#             self.fc1.weight.requires_grad=False
        
#         elif variant==2:
#             #--- Pseudo-inverse
#             print(#Measurement to image domain: pseudo inverse#)           
#             self.fc1 = self.Pinv
            
#         elif variant==3:
#             #--- FC is learnt
#             print(#Measurement to image domain: free#)
            
#             self.fc1 = nn.Linear(M,n**2)
            
#     def forward(self, x, b, c, h, w):
#         #--Projection to the image domain
#         x = self.fc1(x)
#         x = x.view(b, c, h, w)
#     return x     

# ==================================================================================        
class Preprocess(nn.Module): # ça va aller dans le main ??
# ==================================================================================
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
