import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import poisson
from collections import OrderedDict
from spyrit.misc.walsh_hadamard import walsh2_torch, walsh_matrix
from spyrit.core.Forward_Operator import *
import pdb

# =====================================================================================================================
# Acquisition
# =====================================================================================================================       
class Acquisition(nn.Module):
    r"""
        Simulates acquisition by applying Forward_operator to a scaled image such that :math:`y = H_{sub}\frac{1+x}{2}`.
        
        Args:
            - :math:`FO` : Forward_operator
                
        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> Forward_OP = Forward_operator(Hsub)
            >>> Acq = Acquisition(FO)
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
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = Acq(x)
            >>> print(y.shape)
            torch.Size([10, 400])
            
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
    Acquisition with scaled and noisy image with Gaussian-approximated Poisson noise of level \alpha.
    
    Args:
        - :math:`alpha`: noise level.
        - :math:`FO`: Forward Operator.
        
    Shape:
        - Input1: scalar
        - Input2: inapplicable
        
    Example:
        >>> Hsub = np.array(np.random.random([400,32*32]))
        >>> Forward_OP = Forward_operator(Hsub)
        >>> Acq_Poisson_approx_Gauss = Acquisition_Poisson_approx_Gauss(9, Forward_OP)

    """
# ==================================================================================    
    def __init__(self, alpha: float, FO: Forward_operator):
        super().__init__(FO)
        self.alpha = alpha
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates image acquisition with scaling and Gauss-approximated Poisson noise of level alpha. 
        
        Args:
            :math:`x`: Batch of images.
            
        Shape:
            - Input: :math:`(b*c, N)`.
            - Output: :math:`(b*c, M)`.
            
        Examples:
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = Acq_Poisson_approx_Gauss(x)
            >>> print(y.shape)
            torch.Size([10, 400])
            
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
class Acquisition_Poisson_GaussApprox_sameNoise(Acquisition):
# ==================================================================================    
    r"""same as above except that all images in a batch are corrupted with the same 
        noise sample.
        
        Args:
            - :math:`alpha`: Noise level
            - :math:`FO`: Forward Operator
            
        Shape:
            - Input1: scalar
            - Input2: inapplicable
        
        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> Forward_OP = Forward_operator(Hsub)
            >>> APGA_SN = Acquisition_Poisson_GaussApprox_sameNoise(9, FO)
    """
    def __init__(self, alpha, FO):
        super().__init__(FO)
        self.alpha = alpha
        
    def forward(self, x):
        r""" 
        
        Args:
            - :math:`x`: Batch of images.
            
        Shape:
            - Input: :math:`(bc, N)` where N is the number of pixels per image.
            - Output: :math:`(bc, 2M)` where M is the number of simulated acquisitions.
            
        Example:
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = APGA_SN(x)
            >>> print(y.shape)
            torch.Size([10, 400])
            
        """
        # Input shape [b*c, N]  
        # Output shape [b*c, 2*M]

        #--Scale input image      
        x = self.alpha*(x+1)/2
        
        #--Acquisition
        x = self.FO(x)
        x = F.relu(x)       # remove small negative values
        
        #--Measurement noise (Gaussian approximation of Poisson)
        x = x + torch.sqrt(x)*torch.randn(1,x.shape[1])
        return x  
    
# ==================================================================================
class Acquisition_Poisson(Acquisition):
# ================================================================================== 
    r""" Acquisition with scaled and noisy image with Poisson noise from torch library.
    
    Example:
        >>> Hsub = np.array(np.random.random([400,32*32]))
        >>> FO = Forward_operator(Hsub)
        >>> Acq_Poisson = Acquisition_Poisson(9, FO)
    
    """
    def __init__(self, alpha, FO):
        super().__init__(FO)
        self.alpha = alpha

    def forward(self, x):
        r""" Simulates acquisition of images with scaling and Poisson noise simulation.
        Args:
            - :math:`x`: Batch of images
        
        Shape:
            - Input: :math:`(b*c, N)`
            - Output: :math:`(b*c, M)`
            
        Example:
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = Acq_Poisson(x)
            >>> print(y.shape)
            torch.Size([10, 400])            
        """

        #--Scale input image    
        
        x = self.alpha*(x+1)/2 
        #--Acquisition
        x = self.FO(x)
        x = F.relu(x)  
        
        #--Measurement noise imported from Pytorch
        x = poisson(x) 
        return x           
    