import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import poisson
from collections import OrderedDict
from spyrit.misc.walsh_hadamard import walsh2_torch, walsh_matrix
from spyrit.core.Forward_Operator import *

# =====================================================================================================================
# Acquisition
# =====================================================================================================================       
class Acquisition(nn.Module):
    r"""
        Simulates acquisition by applying Forward_operator to a scaled image such that :math:`y = H_{sub}\frac{1+x}{2}`
    """
    def __init__(self, FO: Forward_operator):
        super().__init__()
        # FO = forward operator
        self.FO = FO
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Args:
            :math:`x`: Batch of images.
            
        Shape:
            - Input: :math:`(b*c, N)` 
            - Output: :math:`(b*c, M)`
            
        Example:
            >>> dataset = torchvision.datasets.STL10(root=data_root, split='test',download=False, transform=transform)
            >>> dataloader =  torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)
            >>> inputs, _ = next(iter(dataloader))
            >>> b,c,h,w = inputs.shape
            >>> x = inputs.view(b*c,w*h)
            >>> Hsub = walsh_matrix(w*h)
            >>> F0 = Forward_operator(Hsub)
            >>> Acq = Acquisition(FO)
            >>> y = Acq(x)
            >>> print(x.shape)
            >>> print(y.shape)
            torch.Size([10, 1024])
            torch.Size([10, 1024])
            
        """
        # input x.shape - [b*c,h*w] - [b*c,N] 
        # output x.shape - [b*c,M] 
        #--Scale input image
        x = (x+1)/2; 
        x = self.FO.forward(x); 
        # x is the product of Hsub-sampled*f ?
        return x

# ==================================================================================
class Acquisition_Poisson_approx_Gauss(Acquisition):
    r"""
    Acquisition with scaled and noisy image with Gaussian-approximated Poisson noise.
    Args:
        \alpha: Noise level (Image intensity in photons).
        FO: Forward Operator.
        
    Shape:
        - Input1: python scalar.
        - Input2: :math:`(N, 2*M)`.
    """
# ==================================================================================    
    def __init__(self, alpha: float, FO: Forward_operator):
        super().__init__(FO)
        self.alpha = alpha
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Forward propagates image after scaling and simulating Gauss-approximated Poisson noise. See Lorente Mur et. al, A Deep Network for Reconstructing Images from Undersampled Poisson data, [Research Report] Insa Lyon. 2020. `<https://hal.archives-ouvertes.fr/hal-02944869v1>`_
        Args:
            :math:`x`: Batch of images.
            
        Shape:
            - Input: :math:`(b*c, N)`.
            - Output: :math:`(b*c, 2*M)`.
            
        Examples:
            >>> dataset = torchvision.datasets.STL10(root=data_root, split='test',download=False, transform=transform)
            >>> dataloader =  torch.utils.data.DataLoader(testset, batch_size=c, shuffle=False)
            >>> inputs, _ = next(iter(dataloader))
            >>> b,c,h,w = inputs.shape
            >>> x = inputs.view(b*c,w*h)
            >>> Hsub = walsh_matrix(w*h)
            >>> F0 = Forward_operator(Hsub)
            >>> alpha = 9
            >>> Acq_Poisson_approx_Gauss = Acquisition_Poisson_approx_Gauss(alpha, FO)
            >>> y = Acq_Poisson_approx_Gauss(x)
            >>> print(x.shape)
            >>> print(y.shape)
            torch.Size([10, 1024])
            torch.Size([10, 1024])
            
        """
        # Input shape [b*c, N]  
        # Output shape [b*c, 2*M]

        #--Scale input image      
        x = self.alpha*(x+1)/2
        
        #--Acquisition
        x = self.FO(x)
        x = F.relu(x)       # remove small negative values
        
        #--Measurement noise (Gaussian approximation of Poisson)
        x = x + torch.sqrt(x)*torch.randn_like(x);  
        return x  
    
# ==================================================================================
class Acquisition_Poisson_Pytorch(Acquisition):
# ==================================================================================           
    def __init__(self, alpha, H):
        super().__init__(H)
        self.alpha = alpha

    def forward(self, x):
        # Input shape [b*c, N]  
        # Output shape [b*c, 2*M]

        #--Scale input image      
        x = self.alpha*(x+1)/2
        
        #--Acquisition
        x = self.FO(x)
        x = F.relu(x)  
        
        #--Measurement noise imported from Pytorch
        x = poisson(x) 
        return x           
    