# -*- coding: utf-8 -*-
"""
Reconstruction methods

Created on Fri Jan 20 11:03:12 2023

@author: ducros
"""
import torch
import torch.nn as nn 
import numpy as np
from spyrit.core.meas import HadamSplit, Linear
import math

# ==================================================================================
class PseudoInverse(nn.Module):
# ==================================================================================
    r""" Moore-Penrose Pseudoinverse
    
    Considering linear measurements :math:`y = Hx`, where :math:`H` is the
    measurement matrix and :math:`x` is a vectorized image, it estimates 
    :math:`x` from :math:`y` by computing :math:`\hat{x} = H^\dagger y`, where 
    :math:`H` is the Moore-Penrose pseudo inverse of :math:`H`.

    Example:
        >>> H = np.random.random([400,32*32])
        >>> Perm = np.random.random([32*32,32*32])
        >>> meas_op =  HadamSplit(H, Perm, 32, 32)
        >>> y = torch.rand([85,400], dtype=torch.float)  
        >>> pinv_op = PseudoInverse()
        >>> x = pinv_op(y, meas_op)
        >>> print(x.shape)
        torch.Size([85, 1024])
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.tensor, meas_op: HadamSplit) -> torch.tensor:
        r""" Compute pseudo-inverse of measurements.
        
        Args:
            :attr:`x`: Batch of measurement vectors.
            
            :attr:`meas_op`: Measurement operator. Any class that 
            implements a :meth:`pinv` method can be used, e.g.,
            :class:`~spyrit.core.forwop.HadamSplit`. 
        Shape:
            
            :attr:`x`: :math:`(*, M)`
            
            :attr:`meas_op`: not applicable
            
            :attr:`output`: :math:`(*, N)`

        Example:
            >>> H = np.random.random([400,32*32])
            >>> Perm = np.random.random([32*32,32*32])
            >>> meas_op =  HadamSplit(H, Perm, 32, 32)
            >>> y = torch.rand([85,400], dtype=torch.float)  
            >>> pinv_op = PseudoInverse()
            >>> x = pinv_op(y, meas_op)
            >>> print(x.shape)
            torch.Size([85, 1024])
        """
        x = meas_op.pinv(x)
        return x
    
# ===========================================================================================
class TikhonovMeasurementPriorDiag(nn.Module): 
# ===========================================================================================   
    r"""
    Tikhonov regularization with prior in the measurement domain
    
    Considering linear measurements :math:`y = Hx`, where :math:`H = GF` is the
    measurement matrix and :math:`x` is a vectorized image, it estimates 
    :math:`x` from :math:`y` by approximately minimizing
    
    .. math::
        \| y - GFx \|^2_{\Sigma^{-1}_\alpha} + \|F(x - x_0)\|^2_{\Sigma^{-1}}
    
    where :math:`x_0` is a mean image prior, :math:`\Sigma` is a covariance 
    prior, and :math:`\Sigma_\alpha` is the measurement noise covariance. 
    
    The class is constructed from :math:`\Sigma`.
    
    Args:
        - :attr:`sigma`:  covariance prior
        - :attr:`M`: number of measurements
    
    Shape:
        - :attr:`sigma`: :math:`(N, N)`
    
    Example:
        >>> sigma = np.random.random([32*32, 32*32])
        >>> recon_op = TikhonovMeasurementPriorDiag(sigma, 400)            
    """
    def __init__(self, sigma: np.array, M: int):
        super().__init__()
        
        N = sigma.shape[0] 
        
        self.comp = nn.Linear(M, N-M, False)
        self.denoise_layer = Denoise_layer(M);
        
        diag_index = np.diag_indices(N);
        var_prior = sigma[diag_index];
        var_prior = var_prior[:M]

        self.denoise_layer.weight.data = torch.from_numpy(np.sqrt(var_prior));
        self.denoise_layer.weight.data = self.denoise_layer.weight.data.float();
        self.denoise_layer.weight.requires_grad = False

        Sigma1 = sigma[:M,:M];
        Sigma21 = sigma[M:,:M];
        W = Sigma21 @ np.linalg.inv(Sigma1);
        
        self.comp.weight.data=torch.from_numpy(W)
        self.comp.weight.data=self.comp.weight.data.float()
        self.comp.weight.requires_grad=False
        
    def forward(self, 
                x: torch.tensor, 
                x_0: torch.tensor, 
                var: torch.tensor, 
                meas_op: HadamSplit) -> torch.tensor:
        r"""
        The exact solution is given by
        
        .. math::
            \hat{x} &= x_0 + F^{-1}\begin{bmatrix}\Sigma_1 \\ \Sigma_{21} \end{bmatrix}
                      [\Sigma_1 + \Sigma_\alpha]^{-1} (y - GF x_0)
            
        where the covariance prior is         
        :math:`\Sigma = \begin{bmatrix} \Sigma_1 & \Sigma_{21}^\top \\ \Sigma_{21} & \Sigma_2\end{bmatrix}`
        
        To accelerate the computation of the exact solution, which is dominated
        by a matrix inversion that cannot be precomputed when the measurement
        noise covariance is not known in advance, we compute
        
        .. math::
            \hat{x} = x_0 + F^{-1} \begin{bmatrix} y_1 \\ y_2\end{bmatrix}
        
        with :math:`y_1 = D_1(D_1 + \Sigma_\alpha)^{-1} (y - GF x_0)` and 
        :math:`y_2 = \Sigma_1 \Sigma_{21}^{-1} y_1`, where we choose 
        :math:`D_1 =\textrm{Diag}(\Sigma_1)`. Assuming :math:`\Sigma_\alpha`  
        is diagonal, the inversion is straigtforward.
        
        Args:
            - :attr:`x`: A batch of measurement vectors :math:`y`
            - :attr:`x_0`: A batch of prior images :math:`x_0`
            - :attr:`var`: A batch of measurement noise variances :math:`\Sigma_\alpha`
            - :attr:`meas_op`: A measurement operator that provides :math:`GF` and :math:`F^{-1}`
            
        Shape:
            - :attr:`x`: :math:`(*, M)`
            - :attr:`x_0`: :math:`(*, N)`
            - :attr:`var` :math:`(*, M)`
            - Output: :math:`(*, N)`
            
        Example:
            >>> sigma = np.random.random([32*32, 32*32])
            >>> recon_op = TikhonovMeasurementPriorDiag(sigma, 400)
            >>> H = np.random.random([400,32*32])
            >>> Perm = np.random.random([32*32,32*32])         
            >>> meas_op =  HadamSplit(H, Perm, 32, 32)
            >>> y = torch.rand([85,400], dtype=torch.float) 
            >>> x_0 = torch.zeros((85, 32*32), dtype=torch.float)
            >>> var = torch.zeros((85, 400), dtype=torch.float)
            >>> x = recon_op(y, x_0, var, meas_op)
            >>> print(x.shape)
            torch.Size([85, 1024])       
        """
        x = x - meas_op.forward_H(x_0)
        y1 = torch.mul(self.denoise_layer(var),x)
        y2 = self.comp(y1)

        y = torch.cat((y1,y2),-1)
        x = x_0 + meas_op.inverse(y) 
        return x

# ===========================================================================================
class Denoise_layer(nn.Module):
# ===========================================================================================
    r"""Applies a transformation to the incoming data: :math:`y = A^2/(A^2+x) `

    Args:
        in_features: size of each input sample

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{in})`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, 1)`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`

    Examples::
        >>> m = Denoise_layer(30)
        >>> input = torch.randn(128, 30)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, in_features):
        super(Denoise_layer, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, 0, 2/math.sqrt(self.in_features))

    def forward(self, inputs):
        return self.tikho(inputs, self.weight)

    def extra_repr(self):
        return 'in_features={}'.format(self.in_features)

    @staticmethod
    def tikho(inputs, weight):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Applies a transformation to the incoming data: :math:`y = A^2/(A^2+x)`.
    
        Shape:
            - Input: :math:`(N, *, in\_features)` where `*` means any number of
              additional dimensions - Variance of measurements
            - Weight: :math:`(in\_features)` - corresponds to the standard deviation
              of our prior.
            - Output: :math:`(N, in\_features)`
        """
        sigma = weight**2; # prefer to square it, because when leant, it can got to the 
        #negative, which we do not want to happen.
        # TO BE Potentially done : square inputs.
        den = sigma + inputs;
        ret = sigma/den;
        return ret

#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  RECONSTRUCTION NETWORKS
#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ===========================================================================================
class DC_Net(nn.Module):
# ===========================================================================================
    def __init__(self, Acq, PreP, DC_layer, Denoi):
        super().__init__()
        self.Acq = Acq  # must be a split operator for now
        self.PreP = PreP
        self.DC_layer = DC_layer
        self.Denoi = Denoi

    def forward(self, x):
        # x - of shape [b,c,h,w]
        b,c,h,w = x.shape

        # Acquisition
        x = x.view(b*c,h*w)    # shape x = [b*c,h*w] = [b*c,N]
        x_0 = torch.zeros_like(x).to(x.device)
        x = self.Acq(x) #  shape x = [b*c, 2*M]

        # Preprocessing
        sigma_noi = self.PreP.sigma(x)
        x = self.PreP(x, self.Acq.FO) # shape x = [b*c, M]

        # Data consistency layer
        # measurements to the image domain 
        x = self.DC_layer(x, x_0, sigma_noi, self.Acq.FO) # shape x = [b*c, N]

        # Image-to-image mapping via convolutional networks 
        # Image domain denoising 
        x = x.view(b*c,1,h,w)
        x = self.Denoi(x); # shape stays the same

        x = x.view(b,c,h,w)
        return x;

    def forward_mmse(self, x):
        # x - of shape [b,c,h,w]
        b,c,h,w = x.shape;

        # Acquisition
        x = x.view(b*c,h*w); # shape x = [b*c,h*w] = [b*c,N]
        x_0 = torch.zeros_like(x).to(x.device);
        x = self.Acq(x); #  shape x = [b*c, 2*M]

        # Preprocessing
        sigma_noi = self.PreP.sigma(x);
        x = self.PreP(x, self.Acq.FO); # shape x = [b*c, M]

        # Data consistency layer
        # measurements to the image domain 
        x = self.DC_layer(x, x_0, sigma_noi, self.Acq.FO); # shape x = [b*c, N]

        # Image-to-image mapping via convolutional networks 
        # Image domain denoising 
        x = x.view(b*c,1,h,w);
        return x;
    
    def reconstruct(self, x, h = None, w = None):
        # x - of shape [b,c, 2M]
        b, c, M2 = x.shape;
        
        if h is None :
            h = int(np.sqrt(self.Acq.FO.N))         
        
        if w is None :
            w = int(np.sqrt(self.Acq.FO.N))       
        
        # MMSE reconstruction    
        x = self.reconstruct_meas2im(x, h, w)
        
        # Image-to-image mapping via convolutional networks 
        # Image domain denoising 
        x = x.view(b*c,1,h,w)
        x = self.Denoi(x)   # shape stays the same
        x = x.view(b,c,h,w)
        return x;
    
    def reconstruct2(self, x):
        """
        input x is of shape [b*c, 2M]
        """
        # Measurement to image domain mapping
        x = self.reconstruct_meas2im2(x)         # shape x = [b*c,1,h,w]
        
        # Image domain denoising
        x = self.Denoi(x)                       # shape stays the same
        return x
    
    def reconstruct_meas2im2(self, x):
        # x of shape [b*c, 2M]
        bc, _ = x.shape
    
        # Preprocessing
        sigma_noi = self.PreP.sigma(x)
        x = self.PreP(x, self.Acq.FO) # shape x = [b*c, M]
        x_0 = torch.zeros((bc, self.Acq.FO.N)).to(x.device)
        #x_0 = torch.zeros_like(x).to(x.device)
        
        # measurements to image domain processing
        x = self.DC_layer(x, x_0, sigma_noi, self.Acq.FO) # shape x = [b*c, N]
        x = x.view(bc,1,self.Acq.FO.h, self.Acq.FO.w)   # shape x = [b*c,1,h,w]
        
        return x
    
    def reconstruct_meas2im(self, x, h = None, w = None):
        # x - of shape [b,c, 2M]
        b, c, M2 = x.shape;
        
        if h is None :
            h = int(np.sqrt(self.Acq.FO.N))   
        
        if w is None :
            w = int(np.sqrt(self.Acq.FO.N))
        
        x = x.view(b*c, M2)
        x_0 = torch.zeros((b*c, self.Acq.FO.N)).to(x.device)

        # Preprocessing
        sigma_noi = self.PreP.sigma(x)
        x = self.PreP(x, self.Acq.FO) # shape x = [b*c, M]

        # Data consistency layer
        # measurements to the image domain 
        x = self.DC_layer(x, x_0, sigma_noi, self.Acq.FO) # shape x = [b*c, N]

        x = x.view(b,c,h,w)
        return x
    
    def reconstruct_expe(self, x, h = None, w = None):
        # x - of shape [b,c, 2M]
        bc, _ = x.shape;
        
        if h is None:
            h = int(np.sqrt(self.Acq.FO.N))   
        
        if w is None:
            w = int(np.sqrt(self.Acq.FO.N))
        
        #x = x.view(bc, M2)
        x_0 = torch.zeros((bc, self.Acq.FO.N)).to(x.device)

        # Preprocessing experimental data
        #var_noi = self.PreP.sigma(x)
        #x, N0_est = self.PreP.forward_expe(x, self.Acq.FO) # shape x = [b*c, M]
        
        var_noi = self.PreP.sigma(x)
        x, N0_est = self.PreP.forward_expe(x, self.Acq.FO) # shape x = [b*c, M]
        #var_noi = torch.div(var_noi, N0_est**2)
        
        # Measurement to image domain 
        x = self.DC_layer(x, x_0, var_noi, self.Acq.FO) # shape x = [b*c, N]
        #x = x.view(b,c,h,w)
        
        # Image domain denoising 
        x = x.view(bc,1,h,w)
        x = self.Denoi(x)   # shape stays the same
        #x = x.view(b,c,h,w)
        
        # Denormalization
        N0_est = N0_est.view(bc,1,1,1)
        N0_est = N0_est.expand(bc,1,h,w)
        x = (x+1)*N0_est/2  
        
        return x
    
    
    def reconstruct_expe2(self, x, h = None, w = None):
        # x - of shape [b,c, 2M]
        bc, _ = x.shape;
        
        if h is None:
            h = int(np.sqrt(self.Acq.FO.N))   
        
        if w is None:
            w = int(np.sqrt(self.Acq.FO.N))
        
        #x = x.view(bc, M2)
        x_0 = torch.zeros((bc, self.Acq.FO.N)).to(x.device)

        # Preprocessing experimental data
        #var_noi = self.PreP.sigma(x)
        #x, N0_est = self.PreP.forward_expe(x, self.Acq.FO) # shape x = [b*c, M]
        
        var_noi = self.PreP.sigma_expe(x, gain=1, mudark=700, sigdark=17)
        x, N0_est = self.PreP.forward_expe(x, self.Acq.FO) # shape x = [b*c, M]
        N0_div = N0_est.view(bc,1).expand(bc,self.Acq.FO.M)
        var_noi = torch.div(var_noi, N0_div**2)
        
        # Measurement to image domain 
        x = self.DC_layer(x, x_0, var_noi, self.Acq.FO) # shape x = [b*c, N]
        #x = x.view(b,c,h,w)
        
        # Image domain denoising 
        x = x.view(bc,1,h,w)
        x = self.Denoi(x)   # shape stays the same
        #x = x.view(b,c,h,w)
        
        # Denormalization
        N0_est = N0_est.view(bc,1,1,1)
        N0_est = N0_est.expand(bc,1,h,w)
        x = (x+1)*N0_est/2  
        
        return x


# ===========================================================================================
class Pinv_Net(nn.Module):
# ===========================================================================================
    def __init__(self, Acq, PreP, DC_layer, Denoi):
        super().__init__()
        self.Acq = Acq; # must be a split operator for now
        self.PreP = PreP;
        self.DC_layer = DC_layer; # must be Pinv
        self.Denoi = Denoi;

    def forward(self, x):
        # x is of shape [b,c,h,w]
        b,c,_,_ = x.shape

        # Acquisition
        x = x.view(b,c,self.Acq.FO.N) 
        x = x.view(b*c,self.Acq.FO.N)       # shape x = [b*c,h*w] = [b*c,N]
        x = self.Acq(x)                     # shape x = [b*c, 2*M]

        # Reconstruction 
        x = self.reconstruct(x)             # shape x = [bc, 1, h,w]
        x = x.view(b,c,self.Acq.FO.h, self.Acq.FO.w)
        
        return x

    def forward_meas2im(self, x):
        # x is of shape [b,c,h,w]
        b,c,_,_ = x.shape

        # Acquisition
        x = x.view(b,c,self.Acq.FO.N) 
        x = x.view(b*c,self.Acq.FO.N)           # shape x = [b*c,h*w] = [b*c,N]
        x = self.Acq(x)                         # shape x = [b*c, 2*M]

        # Reconstruction 
        x = self.reconstruct_meas2im(x)         # shape x = [bc,1,h,w]
        x = x.view(b,c,self.Acq.FO.h, self.Acq.FO.w)
        
        return x

    def reconstruct(self, x):
        """
        input x is of shape [b*c, 2M]
        """
        # Measurement to image domain mapping
        x = self.reconstruct_meas2im(x)         # shape x = [b*c,1,h,w]
        
        # Image domain denoising
        x = self.Denoi(x)                       # shape stays the same
        
        return x

    def reconstruct_meas2im(self, x):
        # x of shape [b*c, 2M]
        bc, _ = x.shape
    
        # Preprocessing
        x = self.PreP(x, self.Acq.FO) # shape x = [b*c, M]
    
        # measurements to image domain processing
        x = self.DC_layer(x, self.Acq.FO)               # shape x = [b*c,N]
        x = x.view(bc,1,self.Acq.FO.h, self.Acq.FO.w)   # shape x = [b*c,1,h,w]
        
        return x


    def reconstruct_expe(self, x):
        """
        output image is denormalized with units of photon counts
        """
        # x of shape [b*c, 2M]
        bc, _ = x.shape
    
        # Preprocessing
        x, N0_est = self.PreP.forward_expe(x, self.Acq.FO) # shape x = [b*c, M]
        
        print(N0_est)
    
        # measurements to image domain processing
        x = self.DC_layer(x, self.Acq.FO)               # shape x = [b*c,N]
        x = x.view(bc,1,self.Acq.FO.h, self.Acq.FO.w)   # shape x = [b*c,1,h,w]

        # Image domain denoising
        x = self.Denoi(x)                               # shape x = [b*c,1,h,w]
        
        print(x.max())
        
        # Denormalization 
        x = self.PreP.denormalize_expe(x, N0_est, self.Acq.FO.h, self.Acq.FO.w)
        
        return x
    
# ===========================================================================================
class DC2_Net(Pinv_Net):
# ===========================================================================================
    def __init__(self, Acq, PreP, DC_layer, Denoi):
        super().__init__(Acq, PreP, DC_layer, Denoi)

    def reconstruct_meas2im(self, x):
        # x of shape [b*c, 2M]
        bc, _ = x.shape
    
        # Preprocessing
        var_noi = self.PreP.sigma(x)
        x = self.PreP(x, self.Acq.FO) # shape x = [b*c, M]
    
        # measurements to image domain processing
        x_0 = torch.zeros((bc, self.Acq.FO.N)).to(x.device)
        x = self.DC_layer(x, x_0, var_noi, self.Acq.FO)
        x = x.view(bc,1,self.Acq.FO.h, self.Acq.FO.w)   # shape x = [b*c,1,h,w]
        
        return x        
        
    def reconstruct_expe(self, x):
        """
        The output images are denormalized, i.e., they have units of photon counts. 
        The estimated image intensity N0 is used for both normalizing the raw 
        data and computing the variance of the normalized data.
        """
        # x of shape [b*c, 2M]
        bc, _ = x.shape
        
        # Preprocessing expe
        var_noi = self.PreP.sigma_expe(x)
        x, N0_est = self.PreP.forward_expe(x, self.Acq.FO) # x <- x/N0_est
        x = x/self.PreP.gain
        norm = self.PreP.gain*N0_est
        
        # variance of preprocessed measurements
        var_noi = torch.div(var_noi, (norm.view(-1,1).expand(bc,self.Acq.FO.M))**2)
    
        # measurements to image domain processing
        x_0 = torch.zeros((bc, self.Acq.FO.N)).to(x.device)
        x = self.DC_layer(x, x_0, var_noi, self.Acq.FO)
        x = x.view(bc,1,self.Acq.FO.h, self.Acq.FO.w)       # shape x = [b*c,1,h,w]

        # Image domain denoising
        x = self.Denoi(x)                                  # shape x = [b*c,1,h,w]
        
        # Denormalization 
        x = self.PreP.denormalize_expe(x, norm, self.Acq.FO.h, self.Acq.FO.w)
        
        return x