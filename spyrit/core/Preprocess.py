import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from spyrit.core.Forward_Operator import Linear, LinearSplit, LinearRowSplit, HadamSplit
import pdb
from typing import Union

class SplitPoisson(nn.Module):  
    r"""
        Preprocess raw data acquired with a split measurement operator
        
        It computes :math:`m = \frac{y_{+}-y_{-}}{\alpha}` and the variance
        :math:`var = \frac{2(y_{+} + y_{-})}{\alpha^{2}}`, where 
        :math:`y_{+} = H_{+}x` and :math:`y_{-} = H_{-}x` are obtained using a 
        split measurement operator (see :mod:`spyrit.core.LinearSplit`) and 
        :math:`\alpha` is the image intensity.
            
        Args:
            alpha (float): maximun image intensity :math:`\alpha` (in counts)
            
            M (int): number of measurements :math:`M`
            
            N (int): number of pixels in the image :math:`N`
            
        Example:
            >>> split_op = SplitPoisson(1.0, 400, 32*32)

    """
    def __init__(self, alpha: float, M: int, N: int):
        super().__init__()
        self.alpha = alpha
                
        self.N = N
        self.M = M
        
        self.even_index = range(0,2*M,2)
        self.odd_index  = range(1,2*M,2)
        self.max = nn.MaxPool1d(N)

    def forward(self, 
                x: torch.tensor, 
                meas_op: Union[LinearSplit, HadamSplit],
                ) -> torch.tensor:
        """ 
        Args:
            x: batch of images in the Hadamard domain 
            meas_op: measurement operator
            
        Shape:
            x: :math:`(b*c, 2M, N)` with :math:`b` the batch size, :math:`c` the 
            number of channels, :math:`2M` is twice the number of patterns (as 
            it includes both positive and negative components), and :math:`N` 
            is the number of pixels in the image.
            
            meas_op: :math:`meas_op.M` should match :math:`self.M`
            
            Output: :math:`(b*c, M, N)`
            
        Example:
            >>> x = torch.tensor(np.random.random([10,2*400]), dtype=torch.float)
            >>> H = np.random.random([400,32*32])
            >>> forward_op =  LinearSplit(H)
            >>> m = split_op(x, forward_op)
            torch.Size([10, 400])

        """
        # unsplit
        x = x[:,self.even_index] - x[:,self.odd_index]
        # normalize
        x = 2*x/self.alpha - meas_op.H(torch.ones(x.shape[0], self.N).to(x.device))
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
    
    def set_expe(self, gain=1, mudark=0, sigdark=0, nbin=1):
        r"""
        set experimental noise parameters
        
        Args:        
            - gain in count/electron
            - mudark: average dark current in counts
            - sigdark: standard deviation or dark current in counts
            - nbin: number of raw bin in each spectral channel (if input x results from the sommation/binning of the raw data)
                  
        """
        self.gain = gain
        self.mudark = mudark
        self.sigdark = sigdark
        self.nbin = nbin
        
        
    def sigma_expe(self, x: torch.tensor) -> torch.tensor:
        r"""
        returns estimated variance of **NOT** normalized measurements
        
        Args:
            - :math:`x`: Batch of images in Hadamard Domain.
            
        Shape:
            - Input: :math:`(b*c,2*M)`
            - Output: :math:`(b*c, M)`
            
        Example:
            >>> x = torch.tensor(np.random.random([10,2*32*32]), dtype=torch.float)
            >>> Sig_exp_x = SPP.sigma_expe(x)
            >>> print(Sig_exp_x.shape)
            torch.Size([10, 400])
        """
        # Input shape (b*c, 2*M)
        # output shape (b*c, M)
        x = x[:,self.even_index] + x[:,self.odd_index]
        x = self.gain*(x - 2*self.nbin*self.mudark) + 2*self.nbin*self.sigdark**2
        x = 4*x     # to get the cov of an image in [-1,1], not in [0,1]

        return x

    def sigma_from_image(self, x, meas_op):
        r"""
        
        """
        # x - image. Input shape (b*c, N)
        # meas_op - meas_oprward operator.
        
        x = meas_op.H(x);
        x = x[:,self.even_index] + x[:,self.odd_index]
        x = 4*x/(self.alpha) # here the alpha Contribution is not squared.
        return x
    
    def forward_expe(self, x: torch.tensor, meas_op: LinearSplit) -> torch.tensor:
        r""" 
        Args:
            - :math:`x`: Batch of images
            - :math:`meas_op`: Object of the class Forward_operator_Split_ft_had
        
        Shape:
            - Input: :math:`(bc, 2M)`
            - Output: :math:`(bc, M)`
        
        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> Perm = np.random.random([32*32,32*32])
            >>> FO_Split_ft_had = Forward_operator_Split_ft_had(Hsub, Perm, 32, 32)
            >>> xsub = torch.tensor(np.random.random([10, 2*400]), dtype=torch.float)
            >>> y_FE, alpha_est = SPP.forward_expe(xsub, FO_Split_ft_had)
            >>> print(y_FE.shape)
            >>> print(alpha_est)
            torch.Size([10, 400])
            tensor([0.0251, 0.0228, 0.0232, 0.0294, 0.0248, 0.0245, 0.0184, 0.0253, 0.0267,
                    0.0282])

        """
        bc = x.shape[0]
        
        # unsplit
        x = x[:,self.even_index] - x[:,self.odd_index]
        
        # estimate alpha #x_pinv = FO.adjoint(x)
        x_pinv = meas_op.pinv(x)
        alpha_est = self.max(x_pinv)
        alpha_est = alpha_est.expand(bc,self.M) # shape is (b*c, M)
        
        # normalize
        x = torch.div(x, alpha_est)
        x = 2*x - meas_op.H(torch.ones(bc, self.N).to(x.device))
        
        alpha_est = alpha_est[:,0]    # shape is (b*c,)

        return x, alpha_est
   
    
    def denormalize_expe(self, x, norm, h, w):
        r""" 
        Args:
            - :math:`x`: Batch of expanded images.
            - :math:`norm`: normalizarion values.
            - :math:`h, w`: image height and width.
        
        Shape:
            - Input1: :math:`(bc, 1, h, w)`
            - Input2: :math:`(1, bc)`
            - Input3: int
            - Input4: int
            - Output: :math:`(bc, 1, h, w)`
        
        Example:
            >>> x = torch.tensor(np.random.random([10, 32*32]), dtype=torch.float)
            >>> x1 = x.view(10,1,h,w)
            >>> norm = 9*torch.tensor(np.random.random([1,10]))
            >>> y_DE = SPP.denormalize_expe(x1, norm, 32, 32)
            print(y_DE.shape)
            torch.Size([10, 1, 32, 32])
                        
        """
        bc = x.shape[0]
        
        # Denormalization
        norm = norm.view(bc,1,1,1)
        norm = norm.expand(bc,1,h,w)
        x = (x+1)/2*norm
        
        return x
    
class SplitRowPoisson(nn.Module):  
    r"""
        Preprocess raw data acquired with a split measurement operator
        
        It computes :math:`m = \frac{y_{+}-y_{-}}{\alpha}` and the variance
        :math:`var = \frac{2(y_{+} + y_{-})}{\alpha^{2}}`, where 
        `y_{+} = H_{+}x` and `y_{-} = H_{-}x` are obtained using a split
        measurement operator (see :mod:`spyrit.core.LinearRowSplit`) 
            
        Args:
            :math:`\alpha` (float): maximun image intensity (in counts)
            :math:`M` (int): number of measurements
            :math:`h` (int): number of rows in the image, i.e., image height
            
        Example:
            >>> split_op = SplitRawPoisson(2.0, 24, 64)

    """
    def __init__(self, alpha: float, M: int, h: int):
        super().__init__()
        self.alpha = alpha
        self.M = M        
        self.h = h
        
        self.even_index = range(0,2*M,2)
        self.odd_index  = range(1,2*M,2)
        #self.max = nn.MaxPool1d(h)

    def forward(self, 
                x: torch.tensor, 
                meas_op: LinearRowSplit,
                ) -> torch.tensor:
        """ 
        Args:
            x: batch of images that are Hadamard transformed across rows 
            meas_op: measurement operator
            
        Shape:
            x: :math:`(b*c, 2M, w)` with :math:`b` the batch size, :math:`c` the 
            number of channels, :math:`2M` is twice the number of patterns (as 
            it includes both positive and negative components), and :math:`w` 
            is the image width.
            
            meas_op: The number of measurement `meas_op.M` should match `M`,
            while the length of the measurements :math:`meas_op.N` should match
            image height :math:`h`.  
            
            Output: :math:`(b*c,M)`
            
        Example:
            >>> x = torch.rand([10,48,64], dtype=torch.float)
            >>> H_pos = np.random.random([24,64])
            >>> H_neg = np.random.random([24,64])
            >>> forward_op = LinearRowSplit(H_pos, H_neg)
            >>> m = split_op(x, forward_op)
            >>> print(m.shape)
            torch.Size([10, 24, 64])

        """
        # unsplit
        x = x[:,self.even_index] - x[:,self.odd_index]
        # normalize
        e = torch.ones([x.shape[0], meas_op.N, self.h], device=x.device)
        x = 2*x/self.alpha - meas_op.forward_H(e)
        return x
    

# ==================================================================================
class Preprocess_shift_poisson(nn.Module):      # header needs to be updated!
# ==================================================================================
    r"""Preprocess the measurements acquired using shifted patterns corrupted 
    by Poisson noise

        Computes:
        m = (2 m_shift - m_offset)/N_0
        var = 4*Diag(m_shift + m_offset)/alpha**2
        Warning: dark measurement is assumed to be the 0-th entry of raw measurements

        Args:
            - :math:`alpha`: noise level
            - :math:`M`: number of measurements
            - :math:`N`: number of image pixels

        Shape:
            - Input1: scalar
            - Input2: scalar
            - Input3: scalar

        Example:
            >>> PSP = Preprocess_shift_poisson(9, 400, 32*32)
    """
    def __init__(self, alpha, M, N):
        super().__init__()
        self.alpha = alpha
        self.N = N
        self.M = M

    def forward(self, x: torch.tensor, meas_op: Linear) -> torch.tensor:
        r"""  
        
            Warning:
                - The offset measurement is the 0-th entry of the raw measurements.

            Args:
                - :math:`x`: Batch of images in Hadamard domain shifted by 1
                - :math:`meas_op`: Forward_operator

            Shape:
                - Input: :math:`(b*c, M+1)`
                - Output: :math:`(b*c, M)`

            Example:
                >>> Hsub = np.array(np.random.random([400,32*32]))
                >>> FO = Forward_operator(Hsub)
                >>> x = torch.tensor(np.random.random([10, 400+1]), dtype=torch.float)
                >>> y_PSP = PSP(x, FO)
                >>> print(y_PSP.shape)
                torch.Size([10, 400])
                         
        """
        y = self.offset(x)
        x = 2*x[:,1:] - y.expand(x.shape[0],self.M) # Warning: dark measurement is the 0-th entry
        x = x/self.alpha
        x = 2*x - meas_op.H(torch.ones(x.shape[0], self.N).to(x.device)) # to shift images in [-1,1]^N
        return x
    
    def sigma(self, x):
        r"""
            Args:
                - :math:`x`: Batch of images in Hadamard domain shifted by 1

            Shape:
                - Input: :math:`(b*c, M+1)`

            Example:
                >>> x = torch.tensor(np.random.random([10, 400+1]), dtype=torch.float)
                >>> sigma_PSP = PSP.sigma(x)
                >>> print(sigma_PSP.shape)
                torch.Size([10, 400])
        """ 
        # input x is a set of measurement vectors with shape (b*c, M+1)
        # output is a set of measurement vectors with shape (b*c,M)
        y = self.offset(x)
        x = 4*x[:,1:] + y.expand(x.shape[0],self.M)
        x = x/(self.alpha**2)
        x = 4*x         # to shift images in [-1,1]^N
        return x
    
    def cov(self, x): #return a full matrix ? It is such that Diag(a) + b
        return x

    def sigma_from_image(self, x, meas_op): # should check this!
        # input x is a set of images with shape (b*c, N)
        # input meas_op is a Forward_operator
        x = meas_op.H(x)
        y = self.offset(x)
        x = x[:,1:] + y.expand(x.shape[0],self.M)
        x = x/(self.alpha)     # here the alpha contribution is not squared.
        return x
    
    def offset(self, x):
        r""" Get offset component from bach of shifted images.
        
            Args:
                - :math:`x`: Batch of shifted images

            Shape:
                - Input: :math:`(bc, M+1)`
                - Output: :math:`(bc, 1)`

            Example:
                >>> x = torch.tensor(np.random.random([10, 400+1]), dtype=torch.float)
                >>> y = PSP.offset(x)
                >>> print(y.shape)
                torch.Size([10, 1])
        
        """
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


        Warning:
            dark measurement is assumed to be the 0-th entry of raw measurements

        Args:
            - :math:`alpha`: noise level
            - :math:`M`: number of measurements
            - :math:`N`: number of image pixels

        Shape:
            - Input1: scalar
            - Input2: scalar
            - Input3: scalar

        Example:
            >>> PPP = Preprocess_pos_poisson(9, 400, 32*32)
               
    """
    def __init__(self, alpha, M, N):
        super().__init__()
        self.alpha = alpha
        self.N = N
        self.M = M

    def forward(self, x: torch.tensor, meas_op: Linear) -> torch.tensor:
        r"""
        Args:
            - :math:`x`: noise level
            - :math:`meas_op`: Forward_operator

        Shape:
            - Input1: :math:`(bc, M)`
            - Input2: None
            - Output: :math:`(bc, M)`

        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> meas_op = Forward_operator(Hsub)
            >>> x = torch.tensor(np.random.random([10, 400]), dtype=torch.float)
            >>> y = PPP(x, meas_op)
            torch.Size([10, 400])
            
        """
        y = self.offset(x)
        x = 2*x - y.expand(-1,self.M)
        x = x/self.alpha
        x = 2*x - FO.H(torch.ones(x.shape[0], self.N).to(x.device)) # to shift images in [-1,1]^N
        return x
    
    def offset(self, x):
        r""" Get offset component from bach of shifted images.
        
        Args:
            - :math:`x`: Batch of shifted images
        
        Shape:
            - Input: :math:`(bc, M)`
            - Output: :math:`(bc, 1)`
            
        Example:
            >>> x = torch.tensor(np.random.random([10, 400]), dtype=torch.float)
            >>> y = PPP.offset(x)
            >>> print(y.shape)
            torch.Size([10, 1])
        
        """
        y = 2/(self.M-2)*x[:,1:].sum(dim=1,keepdim=True)
        return y
