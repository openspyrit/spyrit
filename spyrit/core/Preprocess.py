import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from spyrit.core.Forward_Operator import *
import pdb
# ==================================================================================
# Preprocessing
# ==================================================================================  
class Preprocess_Split_diag_poisson_preprocess(nn.Module):  # Why diag ?
# ==================================================================================
    r"""
        Computes :math:`m = \frac{(m_{+}-m_{-})}{N_0}`
        and also allows to compute :math:`var = \frac{2*Diag(m_{+} + m_{-})}{\alpha^{2}}`
            
        Args:
            - :math:`\alpha`: noise level
            - :math:`M`: number of measurements
            - :math:`N`: number of image pixels
            
        Shape:
            - Input1: scalar
            - Input2: scalar
            - Input3: scalar
            
        Example:
            >>> SPP = Preprocess_Split_diag_poisson_preprocess(10, 400, 32*32)

    """
    def __init__(self, alpha: float, M: int, N: int):
        super().__init__()
        self.alpha = alpha
                
        self.N = N
        self.M = M
        
        self.even_index = range(0,2*M,2)
        self.odd_index  = range(1,2*M,2)
        
        self.max = nn.MaxPool1d(N)

    def forward(self, x: torch.tensor , FO: Forward_operator) -> torch.tensor:
        """ 
        Args:
            - :math:`x`: Batch of images in Hadamard Domain
            - :math:`FO`: Forward Operator
            
        Shape:
            - Input: :math:`(b*c,2*M)`
            - Output: :math:`(b*c,M)`
            
        Example:
            >>> from spyrit.core.Forward_Operator import Forward_operator_Split
            >>> x = torch.tensor(np.random.random([10,2*32*32]), dtype=torch.float)
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO_Split = Forward_operator_Split(Hsub)
            >>> y = SPP(x, FO_Split)
            >>> print(y.shape)
            torch.Size([10, 400])

        """
        # unsplit
        x = x[:,self.even_index] - x[:,self.odd_index]
        # normalize
        x = 2*x/self.alpha - FO.Forward_op(torch.ones(x.shape[0], self.N).to(x.device))
        return x
    
    def sigma(self, x: torch.tensor) -> torch.tensor:
        r"""
        returns variance.
        
        Args:
            - :math:`x`: Batch of images in Hadamard Domain
            
        Shape:
            - Input: :math:`(b*c,2*M)`
            - Output: :math:`(b*c, M)`
            
        Example:
            >>> x = torch.tensor(np.random.random([10,2*32*32]), dtype=torch.float)
            >>> Sig_x = SPP.sigma(x)
            >>> print(Sig_x.shape)
            torch.Size([10, 400])
            
        """
        # Input shape (b*c, 2*M)
        # output shape (b*c, M)
        x = x[:,self.even_index] + x[:,self.odd_index];
        x = 4*x/(self.alpha**2); # Cov is in [-1,1] so *4
        return x
    
    def sigma_expe(self, x, gain=1, mudark=0, sigdark=0, nbin=1):
        r"""
        returns estimated variance of **NOT** normalized measurements
        
        gain in count/electron
        mudark: average dark current in counts
        sigdark: standard deviation or dark current in counts
        nbin: number of raw bin in each spectral channel (if input x results 
        from the sommation/binning of the raw data)
        
        Args:
            - :math:`x`: Batch of images in Hadamard Domain.
            
        Shape:
            - Input: :math:`(b*c,2*M)`
            - Output: :math:`(b*c, M)`
            
        Example:
            >>> x = torch.tensor(np.random.random([10,2*32*32]), dtype=torch.float)
            >>> Sig_exp_x = SPP.sigma_expe(x, gain=1, mudark=0, sigdark=0, nbin=1)
            >>> print(Sig_exp_x.shape)
            torch.Size([10, 400])
           
        """
        # Input shape (b*c, 2*M)
        # output shape (b*c, M)
        x = x[:,self.even_index] + x[:,self.odd_index]
        x = gain*(x - 2*nbin*mudark) + 2*nbin*sigdark**2
        x = 4*x     # to get the cov of an image in [-1,1], not in [0,1]

        return x

    def sigma_from_image(self, x, FO):
        r"""
        
        """
        pdb.set_trace()
        # x - image. Input shape (b*c, N)
        # FO - Forward operator.
        x = FO.Forward_op(x);
        x = x[:,self.even_index] + x[:,self.odd_index]
        x = 4*x/(self.alpha) # here the alpha Contribution is not squared.
        return x
    
    def forward_expe(self, x, FO):
        """ 
            Input shape [b*c,2*M]
            Output shape [b*c,M]
        """
        bc = x.shape[0]
        
        # unsplit
        x = x[:,self.even_index] - x[:,self.odd_index]
        
        # estimate alpha #x_pinv = FO.adjoint(x)
        x_pinv = FO.pinv(x)
        alpha_est = self.max(x_pinv)
        alpha_est = alpha_est.expand(bc,self.M) # shape is (b*c, M)
        
        # normalize
        x = torch.div(x, alpha_est)
        x = 2*x - FO.Forward_op(torch.ones(bc, self.N).to(x.device))
        
        alpha_est = alpha_est[:,0]    # shape is (b*c,)
        
        print(alpha_est)
        
        return x, alpha_est
   
    
    def denormalize_expe(self, x, alpha, h, w):
        """ 
            x has shape (b*c,1,h,w)
            alpha has shape (b*c,)
            
            Output has shape (b*c,1,h,w)
        """
        bc = x.shape[0]
        
        # Denormalization
        alpha = alpha.view(bc,1,1,1)
        alpha = alpha.expand(bc,1,h,w)
        x = (x+1)*alpha/2 
        
        return x

# ==================================================================================
class Preprocess_shift_poisson(nn.Module):      # header needs to be updated!
# ==================================================================================
    r"""Preprocess the measurements acquired using shifted patterns corrupted 
    by Poisson noise
    
    The output value of the layer with input size :math:`(B*C, M+1)` can be 
    described as:

    .. math::
        \text{out}((B*C)_i, M_j}) = 2*\text{input}((B*C)_i, M_{j+1}) -
        \text{input}((B*C)_i, M_0}), \quad 0 \le j \le M-1
 
    The output size of the layer is :math:`(B*C, M)` 

    Note:
        This module ...

    Args:
        in_channels (int): Number of ...
        
    Warning:
        The offset measurement is the 0-th entry of the raw measurements
    """
    
    """
    Computes 
        m = (2 m_shift - m_offset)/N_0
        var = 4*Diag(m_shift + m_offset)/alpha**2
        Warning: dark measurement is assumed to be the 0-th entry of raw measurements
    """
    r"""
    
        Args:
            - :math:`alpha`: noise level
            - :math:`M`: number of measurements
            - :math:`N`: number of image pixels
            
        Shape:
            - Input1: scalar
            - Input2: scalar
            - Input3: scalar
            
        Example:
            >>> PSP = Preprocess_shift_poisson(10, 400, 32*32)
    """
    def __init__(self, alpha, M, N):
        super().__init__()
        self.alpha = alpha
        self.N = N
        self.M = M

    def forward(self, x: torch.tensor, FO: Forward_operator) -> torch.tensor:
        r"""
        Args:
            - :math:`x`: Batch of images in Hadamard domain shifted by 1
            - :maht:`FO`: 
            
        Shape:
            - Input1: :math:`(b*c, M+1)`
            - Output: :math:`(b*c, M)`
            
        Example
            >>>
            >>>
            >>> 
            
        """
        # Input  has shape (b*c, M+1)
        # Output has shape (b*c, M)
        y = self.offset(x)
        x = 2*x[:,1:] - y.expand(x.shape[0],self.M) # Warning: dark measurement is the 0-th entry
        x = x/self.alpha
        x = 2*x - FO.Forward_op(torch.ones(x.shape[0], self.N).to(x.device)) # to shift images in [-1,1]^N
        return x
    
    def sigma(self, x):
        # input x is a set of measurement vectors with shape (b*c, M+1)
        # output is a set of measurement vectors with shape (b*c,M)
        y = self.offset(x)
        x = 4*x[:,1:] + y.expand(x.shape[0],self.M)
        x = x/(self.alpha**2)
        x = 4*x         # to shift images in [-1,1]^N
        return x
    
    def cov(self, x): #return a full matrix ? It is such that Diag(a) + b
        return x

    def sigma_from_image(self, x, FO): # should check this!
        # input x is a set of images with shape (b*c, N)
        # input FO is a Forward_operator
        x = FO.Forward_op(x)
        y = self.offset(x)
        x = x[:,1:] + y.expand(x.shape[0],self.M)
        x = x/(self.alpha)     # here the alpha contribution is not squared.
        return x
    
    def offset(self, x):
        # Input  has shape (b*c, M+1)
        # Output has shape (b*c, 1)
        y = x[:,0,None]
        return y
    
# ==================================================================================
class Preprocess_pos_poisson(nn.Module):  # header needs to be updated!
# ==================================================================================
    r"""Preprocess the measurements acquired using positive (shifted) patterns 
    corrupted by Poisson noise
    
    The output value of the layer with input size :math:`(B*C, M)` can be 
    described as:

    .. math::
        \text{out}((B*C)_i, M_j}) = 2*\text{input}((B*C)_i, M_j}) -
        \sum_{k = 1}^{M-1} \text{input}((B*C)_i, M_k})
 
    The output size of the layer is :math:`(B*C, M)`, which is the imput size 

    Note:
        This module ...

    Args:
        in_channels (int): Number of ...
        
    Warning:
        dark measurement is assumed to be the 0-th entry of raw measurements
    """
    def __init__(self, alpha, M, N):
        super().__init__()
        self.alpha = alpha
        self.N = N
        self.M = M

    def forward(self, x, FO):
        # Input  has shape (b*c, M)
        # Output has shape (b*c, M)
        
        y = self.offset(x)
        print(x.shape)
        print(y.expand(-1,self.M).shape)
        x = 2*x - y.expand(-1,self.M)
        x = x/self.alpha
        x = 2*x - FO.Forward_op(torch.ones(x.shape[0], self.N).to(x.device)) # to shift images in [-1,1]^N
        return x
    
    def offset(self, x):
        # Input  has shape (b*c, M)
        # Output has shape (b*c, 1)
        y = 2/(self.M-2)*x[:,1:].sum(dim=1,keepdim=True)
        return y
