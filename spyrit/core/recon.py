# -*- coding: utf-8 -*-
"""
Reconstruction methods

Created on Fri Jan 20 11:03:12 2023

@author: ducros
"""
import torch
import torch.nn as nn 
import numpy as np
from spyrit.core.meas import HadamSplit, LinearRowSplit, Linear
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
        
    def forward(self, x: torch.tensor, meas_op) -> torch.tensor:
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
        - :attr:`sigma`:  covariance prior with shape :math:`(N, N)`
        - :attr:`M`: number of measurements
    
        
    Attributes:
        :attr:`comp`: The learnable completion layer initialized as 
        :math:`\Sigma_1 \Sigma_{21}^{-1}`. This layer is a :class:`nn.Linear`
        
        :attr:`denoi`: The learnable denoising layer initialized from 
        :math:`\Sigma_1`.
    
    Example:
        >>> sigma = np.random.random([32*32, 32*32])
        >>> recon_op = TikhonovMeasurementPriorDiag(sigma, 400)            
    """
    def __init__(self, sigma: np.array, M: int):
        super().__init__()
        
        N = sigma.shape[0] 
        
        self.comp = nn.Linear(M, N-M, False)
        self.denoi = Denoise_layer(M)
        
        diag_index = np.diag_indices(N)
        var_prior = sigma[diag_index]
        var_prior = var_prior[:M]

        self.denoi.weight.data = torch.from_numpy(np.sqrt(var_prior));
        self.denoi.weight.data = self.denoi.weight.data.float();
        self.denoi.weight.requires_grad = False

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
        
        We approximate the solution as
        
        .. math::
            \hat{x} = x_0 + F^{-1} \begin{bmatrix} y_1 \\ y_2\end{bmatrix}
        
        with :math:`y_1 = D_1(D_1 + \Sigma_\alpha)^{-1} (y - GF x_0)` and 
        :math:`y_2 = \Sigma_1 \Sigma_{21}^{-1} y_1`, where 
        :math:`\Sigma = \begin{bmatrix} \Sigma_1 & \Sigma_{21}^\top \\ \Sigma_{21} & \Sigma_2\end{bmatrix}`
        and  :math:`D_1 =\textrm{Diag}(\Sigma_1)`. Assuming the noise 
        covariance :math:`\Sigma_\alpha` is diagonal, the matrix inversion 
        involded in the computation of :math:`y_1` is straigtforward.
        
        This is an approximation to the exact solution
        
        .. math::
            \hat{x} &= x_0 + F^{-1}\begin{bmatrix}\Sigma_1 \\ \Sigma_{21} \end{bmatrix}
                      [\Sigma_1 + \Sigma_\alpha]^{-1} (y - GF x_0)
            
        
        See Lemma B.0.5 of the PhD dissertation of A. Lorente Mur (2021): 
        https://theses.hal.science/tel-03670825v1/file/these.pdf
        
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
            >>> B, H, M = 85, 32, 512
            >>> sigma = np.random.random([H**2, H**2])
            >>> recon_op = TikhonovMeasurementPriorDiag(sigma, M)
            >>> Ord = np.ones((H,H))
            >> meas = HadamSplit(M, H, Ord)
            >>> y = torch.rand([B,M], dtype=torch.float)  
            >>> x_0 = torch.zeros((B, H**2), dtype=torch.float)
            >>> var = torch.zeros((B, M), dtype=torch.float)
            >>> x = recon_op(y, x_0, var, meas)
            torch.Size([85, 1024])       
        """
        x = x - meas_op.forward_H(x_0)
        y1 = torch.mul(self.denoi(var),x)
        y2 = self.comp(y1)

        y = torch.cat((y1,y2),-1)
        x = x_0 + meas_op.inverse(y) 
        return x

# ===========================================================================================
class Denoise_layer(nn.Module):
# ===========================================================================================
    r""" Wiener filter that assumes additive white Gaussian noise
    
    .. math::
        y = \sigma_\text{prior}^2/(\sigma^2_\text{prior} + \sigma^2_\text{meas}) x, 
    where :math:`\sigma^2_\text{prior}` is the variance prior and 
    :math:`\sigma^2_\text{meas}` is the variance of the measurement,
    x is the input vector and y is the output vector.

    Args:
        :attr:`M`: size of incoming vector

    Shape:
        - Input: :math:`(*, M)`.
        - Output: :math:`(*, M)`.

    Attributes:
        :attr:`sigma`: 
            the learnable standard deviation prior 
            :math:`\sigma_\text{prior}` of shape :math:`(M, 1)`. The 
            values are initialized from 
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where 
            :math:`k = 1/M`.

    Example:
        >>> m = Denoise_layer(30)
        >>> input = torch.randn(128, 30)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, M):
        super(Denoise_layer, self).__init__()
        self.in_features = M
        self.weight = nn.Parameter(torch.Tensor(M))
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
        var = weight**2 # prefer to square it, because when leant, it can got to the 
        #negative, which we do not want to happen.
        # TO BE Potentially done : square inputs.
        den = var + inputs
        ret = var/den
        return ret

#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  RECONSTRUCTION NETWORKS
#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# =============================================================================
class PinvNet(nn.Module):
# =============================================================================
    r""" Pseudo inverse reconstruction network
    
    .. math:
        
        
    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`) 
        
        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)
        
        :attr:`denoi` (optional): Image denoising operator 
        (see :class:`~spyrit.core.nnet`). 
        Default :class:`~spyrit.core.nnet.Identity`
    
    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(B,C,H,W)` 
        
        :attr:`output`: Reconstructed images with shape :math:`(B,C,H,W)`
    
    Attributes:
        :attr:`Acq`: Acquisition operator initialized as :attr:`noise`
        
        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`
        
        :attr:`pinv`: Analytical reconstruction operator initialized as 
        :class:`~spyrit.core.recon.PseudoInverse()`
        
        :attr:`Denoi`: Image denoising operator initialized as :attr:`denoi`

    
    Example:
        >>> B, C, H, M = 10, 1, 64, 64**2
        >>> Ord = np.ones((H,H))
        >>> meas = HadamSplit(M, H, Ord)
        >>> noise = NoNoise(meas)
        >>> prep = SplitPoisson(1.0, M, H*H)
        >>> recnet = PinvNet(noise, prep)
        >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
        >>> z = recnet(x)
        >>> print(z.shape)
        >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
        torch.Size([10, 1, 64, 64])
        tensor(5.8912e-06)
    """
    def __init__(self, noise, prep, denoi=nn.Identity()):
        super().__init__()
        self.acqu = noise 
        self.prep = prep
        self.pinv = PseudoInverse()
        self.denoi = denoi

    def forward(self, x):
        r""" Full pipeline of reconstrcution network
            
        Args:
            :attr:`x`: ground-truth images
        
        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`
            
            :attr:`output`: reconstructed images with shape :math:`(B,C,H,W)`
        
        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet(x)
            >>> print(z.shape)
            >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
            torch.Size([10, 1, 64, 64])
            tensor(5.8912e-06)
        """
        
        b,c,_,_ = x.shape

        # Acquisition
        x = x.view(b*c,self.acqu.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        x = self.acqu(x)                     # shape x = [b*c, 2*M]

        # Reconstruction 
        x = self.reconstruct(x)             # shape x = [bc, 1, h,w]
        x = x.view(b,c,self.acqu.meas_op.h, self.acqu.meas_op.w)
        
        return x
    
    def acquire(self, x):
        r""" Simulate data acquisition
            
        Args:
            :attr:`x`: ground-truth images
        
        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`
            
            :attr:`output`: measurement vectors with shape :math:`(BC,2M)`
        
        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet.acquire(x)
            >>> print(z.shape)
            torch.Size([10, 8192])
        """
        
        b,c,_,_ = x.shape

        # Acquisition
        x = x.view(b*c,self.acqu.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        x = self.acqu(x)                     # shape x = [b*c, 2*M]
        
        return x
    
    def meas2img(self, y):
        """Return images from raw measurement vectors

        Args:
            :attr:`x`: raw measurement vectors
        
        Shape:
            :attr:`x`: :math:`(BC,2M)`
            
            :attr:`output`: :math:`(BC,1,H,W)`
        
        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H**2)
            >>> recnet = PinvNet(noise, prep) 
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        m = self.prep(y)
        m = torch.nn.functional.pad(m, (0, self.acqu.meas_op.N-self.acqu.meas_op.M))
        z = m @ self.acqu.meas_op.Perm.weight.data.T
        z = z.view(-1,1,self.acqu.meas_op.h, self.acqu.meas_op.w)
        
        return z

    def reconstruct(self, x):
        r""" Reconstruction step of a reconstruction network
            
        Args:
            :attr:`x`: raw measurement vectors
        
        Shape:
            :attr:`x`: :math:`(BC,2M)`
            
            :attr:`output`: :math:`(BC,1,H,W)`
        
        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H**2)
            >>> recnet = PinvNet(noise, prep) 
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        # Measurement to image domain mapping
        bc, _ = x.shape
    
        # Preprocessing in the measurement domain
        x = self.prep(x) # shape x = [b*c, M]
    
        # measurements to image-domain processing
        x = self.pinv(x, self.acqu.meas_op)               # shape x = [b*c,N]
                
        # Image-domain denoising
        x = x.view(bc,1,self.acqu.meas_op.h, self.acqu.meas_op.w)   # shape x = [b*c,1,h,w]
        x = self.denoi(x)                       
        
        return x
    
    def reconstruct_pinv(self, x):
        r""" Reconstruction step of a reconstruction network
            
        Args:
            :attr:`x`: raw measurement vectors
        
        Shape:
            :attr:`x`: :math:`(BC,2M)`
            
            :attr:`output`: :math:`(BC,1,H,W)`
        
        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H**2)
            >>> recnet = PinvNet(noise, prep) 
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct_pinv(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        # Measurement to image domain mapping
        bc, _ = x.shape
    
        # Preprocessing in the measurement domain
        x = self.prep(x)#, self.acqu.meas_op) # shape x = [b*c, M]
    
        # measurements to image-domain processing
        x = self.pinv(x, self.acqu.meas_op)               # shape x = [b*c,N]
                
        # Image-domain denoising
        x = x.view(bc,1,self.acqu.meas_op.h, self.acqu.meas_op.w)   # shape x = [b*c,1,h,w]                       
        
        return x


    def reconstruct_expe(self, x):
        r""" Reconstruction step of a reconstruction network
        
        Same as :meth:`reconstruct` reconstruct except that:
            
        1. The preprocessing step estimates the image intensity for normalization
        
        2. The output images are "denormalized", i.e., have units of photon counts
            
        Args:
            :attr:`x`: raw measurement vectors
        
        Shape:
            :attr:`x`: :math:`(BC,2M)`
            
            :attr:`output`: :math:`(BC,1,H,W)`

        """
        # x of shape [b*c, 2M]
        bc, _ = x.shape
    
        # Preprocessing
        x, N0_est = self.prep.forward_expe(x, self.acqu.meas_op) # shape x = [b*c, M]
        print(N0_est)
    
        # measurements to image domain processing
        x = self.pinv(x, self.acqu.meas_op)               # shape x = [b*c,N]
        
        # Image domain denoising
        x = x.view(bc,1,self.acqu.meas_op.h, self.acqu.meas_op.w)   # shape x = [b*c,1,h,w]
        x = self.denoi(x)                               # shape x = [b*c,1,h,w]
        print(x.max())
        
        # Denormalization 
        x = self.prep.denormalize_expe(x, N0_est, self.acqu.meas_op.h, 
                                                  self.acqu.meas_op.w)
        return x
    
#%%===========================================================================================
class DCNet(nn.Module):
# ===========================================================================================
    r""" Denoised completion reconstruction network
    
    .. math:
        
    
    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`) 
        
        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)
        
        :attr:`sigma`: UPDATE!! Tikhonov reconstruction operator of type 
        :class:`~spyrit.core.recon.TikhonovMeasurementPriorDiag()`
        
        :attr:`denoi` (optional): Image denoising operator 
        (see :class:`~spyrit.core.nnet`). 
        Default :class:`~spyrit.core.nnet.Identity`
    
    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(B,C,H,W)` 
        
        :attr:`output`: Reconstructed images with shape :math:`(B,C,H,W)`
    
    Attributes:
        :attr:`Acq`: Acquisition operator initialized as :attr:`noise`
        
        :attr:`PreP`: Preprocessing operator initialized as :attr:`prep`
        
        :attr:`DC_Layer`: Data consistency layer initialized as :attr:`tikho`
        
        :attr:`Denoi`: Image denoising operator initialized as :attr:`denoi`

    
    Example:
        >>> B, C, H, M = 10, 1, 64, 64**2
        >>> Ord = np.ones((H,H))
        >>> meas = HadamSplit(M, H, Ord)
        >>> noise = NoNoise(meas)
        >>> prep = SplitPoisson(1.0, M, H*H)
        >>> sigma = np.random.random([H**2, H**2])
        >>> recnet = DCNet(noise,prep,sigma)
        >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
        >>> z = recnet(x)
        >>> print(z.shape)
        torch.Size([10, 1, 64, 64])
    """
    def __init__(self, 
                 noise, 
                 prep, 
                 sigma,
                 denoi = nn.Identity()):
        
        super().__init__()
        self.Acq = noise 
        self.prep = prep
        Perm = noise.meas_op.Perm.weight.data.cpu().numpy().T
        sigma_perm = Perm @ sigma @ Perm.T
        self.tikho = TikhonovMeasurementPriorDiag(sigma_perm, noise.meas_op.M)
        self.denoi = denoi
        
    def forward(self, x):
        r""" Full pipeline of the reconstruction network
            
        Args:
            :attr:`x`: ground-truth images
        
        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`
            
            :attr:`output`: reconstructed images with shape :math:`(B,C,H,W)`
        
        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> sigma = np.random.random([H**2, H**2])
            >>> recnet = DCNet(noise,prep,sigma)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        
        b,c,_,_ = x.shape

        # Acquisition
        x = x.view(b*c,self.Acq.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        x = self.Acq(x)                     # shape x = [b*c, 2*M]

        # Reconstruction 
        x = self.reconstruct(x)             # shape x = [bc, 1, h,w]
        x = x.view(b,c,self.Acq.meas_op.h, self.Acq.meas_op.w)
        
        return x
    
    def acquire(self, x):
        r""" Simulate data acquisition
            
        Args:
            :attr:`x`: ground-truth images
        
        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`
            
            :attr:`output`: measurement vectors with shape :math:`(BC,2M)`
        
        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> sigma = np.random.random([H**2, H**2])
            >>> recnet = DCNet(noise,prep,sigma)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet.acquire(x)
            >>> print(z.shape)
            torch.Size([10, 8192])
        """
        
        b,c,_,_ = x.shape

        # Acquisition
        x = x.view(b*c,self.Acq.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        x = self.Acq(x)                     # shape x = [b*c, 2*M]
        
        return x

    def reconstruct(self, x):
        r""" Reconstruction step of a reconstruction network
            
        Args:
            :attr:`x`: raw measurement vectors
        
        Shape:
            :attr:`x`: raw measurement vectors with shape :math:`(BC,2M)`
            
            :attr:`output`: reconstructed images with shape :math:`(BC,1,H,W)`
        
        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = np.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> sigma = np.random.random([H**2, H**2])
            >>> recnet = DCNet(noise,prep,sigma)
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        # x of shape [b*c, 2M]
        bc, _ = x.shape
    
        # Preprocessing
        var_noi = self.prep.sigma(x)
        x = self.prep(x) # shape x = [b*c, M]
    
        # measurements to image domain processing
        x_0 = torch.zeros((bc, self.Acq.meas_op.N), device=x.device)
        x = self.tikho(x, x_0, var_noi, self.Acq.meas_op)
        x = x.view(bc,1,self.Acq.meas_op.h, self.Acq.meas_op.w)   # shape x = [b*c,1,h,w]
        
        # Image domain denoising
        x = self.denoi(x)               
        
        return x        
        
    def reconstruct_expe(self, x):
        r""" Reconstruction step of a reconstruction network
        
        Same as :meth:`reconstruct` reconstruct except that:
            
            1. The preprocessing step estimates the image intensity. The 
            estimated intensity is used for both normalizing the raw 
            data and computing the variance of the normalized data.
            
            2. The output images are "denormalized", i.e., have units of photon 
            counts
            
        Args:
            :attr:`x`: raw measurement vectors
        
        Shape:
            :attr:`x`: :math:`(BC,2M)`
            
            :attr:`output`: :math:`(BC,1,H,W)`

        """
        # x of shape [b*c, 2M]
        bc, _ = x.shape
        
        # Preprocessing expe
        var_noi = self.prep.sigma_expe(x)
        x, N0_est = self.prep.forward_expe(x, self.Acq.meas_op) # x <- x/N0_est
        x = x/self.prep.gain
        norm = self.prep.gain*N0_est
        
        # variance of preprocessed measurements
        var_noi = torch.div(var_noi, (norm.view(-1,1).expand(bc,self.Acq.meas_op.M))**2)
    
        # measurements to image domain processing
        x_0 = torch.zeros((bc, self.Acq.meas_op.N), device=x.device)
        x = self.tikho(x, x_0, var_noi, self.Acq.meas_op)
        x = x.view(bc,1,self.Acq.meas_op.h, self.Acq.meas_op.w)       # shape x = [b*c,1,h,w]

        # Image domain denoising
        x = self.denoi(x)                                  # shape x = [b*c,1,h,w]
        
        # Denormalization 
        x = self.prep.denormalize_expe(x, norm, self.Acq.meas_op.h, self.Acq.meas_op.w)
        
        return x