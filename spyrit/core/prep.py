import torch
import torch.nn as nn
from spyrit.core.meas import Linear, LinearSplit, LinearRowSplit, HadamSplit
from typing import Union, Tuple
import math
  
#==============================================================================
class DirectPoisson(nn.Module):
#==============================================================================
    r"""
    
    Input data should be dark substracted. Probably bugged/Use with caution.
    
    Args:
        :attr:`alpha`: maximun image intensity :math:`\alpha` (in counts)
        
        :attr:`meas_op`: measurement operator (see :mod:`~spyrit.core.meas`)
        
        
    Example:
        >>> H = np.random.random([400,32*32])
        >>> meas_op =  Linear(H)
        >>> prep_op = DirectPoisson(1.0, meas_op)

    """
    def __init__(self, alpha: float, meas_op):
        super().__init__()
        self.alpha = alpha
        self.N = meas_op.N
        self.M = meas_op.M
        
        self.max = nn.MaxPool1d(self.N)
        self.register_buffer('H_ones', meas_op(torch.ones((1,self.N))))
            
    def forward(self, x: torch.tensor) -> torch.tensor:
        r""" 
        Preprocess measurements to compensate for the affine image normalization
        
        It computes :math:`\frac{1}{\alpha}x - H1`, where 1 represents the 
        all-ones vector.
        
        Args:
            :attr:`x`: batch of measurement vectors 
            
        Shape:
            x: :math:`(B, M)` where :math:`B` is the batch dimension
            
            meas_op: the number of measurements :attr:`meas_op.M` should match
            :math:`M`.
            
            Output: :math:`(B, M)`
            
        Example:
            >>> x = torch.rand([10,400], dtype=torch.float)
            >>> H = np.random.random([400,32*32])
            >>> meas_op =  Linear(H)
            >>> prep_op = DirectPoisson(1.0, meas_op)
            >>> m = prep_op(x)
            >>> print(m.shape)
            torch.Size([10, 400])
        """
        # normalize
        s = x.shape[:-1] + torch.Size([self.M])     # torch.Size([*,M])
        H_ones = self.H_ones.expand(s)
        x = 2*x/self.alpha - H_ones
        return x
    
    def forward_expe2(self, x: torch.tensor, 
                            dummy,
                            dim = -1) -> torch.tensor:
        
        # # estimate intensity x gain (in counts)
        # mu = torch.sum(x, dim, keepdim=True)
        
        # # All rows of an image have the same normalization
        # alpha = torch.amax(mu, -2, keepdim=True)
        
        alpha = torch.tensor([11000], device=x.device)
        
        s = x.shape[:-1] + torch.Size([self.M])     # torch.Size([*,M])
        H_ones = self.H_ones.expand(s)
        x = 2*x/self.alpha - H_ones
        return x, alpha 
    
    def denormalize_expe(self, x, beta):
        r""" 
        Denormalize images from the range [-1;1] to the range [0; :math:`\beta`]
        
        It computes :math:`m = \frac{\beta}{2}(x+1)`, where 
        :math:`\beta` is the normalization factor. 
        
        Args:
            :attr:`x`: Batch of images
            
            :attr:`beta`: Normalizarion factor
            
            :attr:`h`: Image height
            
            :attr:`w`: Image width
        
        Shape:
            :attr:`x`: :math:`(*, 1, h, w)`
            
            :attr:`beta`: :math:`(*)` or :math:`(*, 1)` 
            
            :attr:`h`: int
            
            :attr:`w`: int
            
            :attr:`Output`: :math:`(*, 1, h, w)`
        
        Example:
            >>> x = torch.rand([10, 1, 32,32], dtype=torch.float)
            >>> beta = 9*torch.rand([10])
            >>> y = prep_op.denormalize_expe(x, beta, 32, 32)
            >>> print(y.shape)
            torch.Size([10, 1, 32, 32])
                        
        """
        x = (x+1)/2*beta
        
        return x

    def set_expe(self, gain=1.0, mudark=0.0, sigdark=0.0, nbin=1.0):
        r"""
        Sets experimental parameters of the sensor
        
        Args:        
            - :attr:`gain` (float): gain (in count/electron)
            - :attr:`mudark` (float): average dark current (in counts)
            - :attr:`sigdark` (float): standard deviation or dark current (in counts)
            - :attr:`nbin` (float): number of raw bin in each spectral channel (if input x results from the sommation/binning of the raw data)
        
        Example:
            >>> split_op.set_expe(gain=1.6)
            >>> print(split_op.gain)
            1.6
        """
        self.gain = gain
        self.mudark = mudark
        self.sigdark = sigdark
        self.nbin = nbin
    
#==============================================================================
class SplitPoisson(nn.Module):
#==============================================================================
    r"""
    Preprocess the raw data acquired with a plit measurement operator assuming 
    Poisson noise.  It also compensates for the affine transformation applied 
    to the images to get positive intensities.
    
    It computes :math:`m = \frac{y_{+}-y_{-}}{\alpha} - H1` and the variance
    :math:`var = \frac{2(y_{+} + y_{-})}{\alpha^{2}}`, where 
    :math:`y_{+} = H_{+}x` and :math:`y_{-} = H_{-}x` are obtained using a 
    split measurement operator (see :mod:`spyrit.core.LinearSplit`), 
    :math:`\alpha` is the image intensity, and 1 is the all-ones vector.
    
    Args:
        alpha (float): maximun image intensity :math:`\alpha` (in counts)
        
        :attr:`meas_op`: measurement operator (see :mod:`~spyrit.core.meas`)
        
        
    Example:
        >>> H = np.random.random([400,32*32])
        >>> meas_op =  LinearSplit(H)
        >>> split_op = SplitPoisson(10, meas_op)

    Example 2:
        >>> Perm = np.random.random([32,32])
        >>> meas_op = HadamSplit(400, 32,  Perm)
        >>> split_op = SplitPoisson(10, meas_op)

    """
    def __init__(self, alpha: float, meas_op):
        super().__init__()
        self.alpha = alpha
        self.N = meas_op.N
        self.M = meas_op.M
        
        self.even_index = range(0,2*self.M,2)
        self.odd_index  = range(1,2*self.M,2)
        self.max = nn.MaxPool1d(self.N)
        
        self.register_buffer('H_ones', meas_op.H(torch.ones((1,self.N))))

    def forward(self, x: torch.tensor) -> torch.tensor:
        r""" 
        Preprocess to compensates for image normalization and splitting of the 
        measurement operator.
        
        It computes :math:`\frac{x[0::2]-x[1::2]}{\alpha} - H1`
        
        Args:
            :attr:`x`: batch of measurement vectors 
            
        Shape:
            x: :math:`(*, 2M)` where :math:`*` indicates one or more dimensions
            
            meas_op: the number of measurements :attr:`meas_op.M` should match
            :math:`M`.
            
            Output: :math:`(*, M)`
            
        Example:
            >>> x = torch.rand([10,2*400], dtype=torch.float)
            >>> H = np.random.random([400,32*32])
            >>> meas_op =  LinearSplit(H)
            >>> split_op = SplitPoisson(10, meas_op)
            >>> m = split_op(x)
            >>> print(m.shape)
            torch.Size([10, 400])
            
        Example 2:
            >>> x = torch.rand([10,2*400], dtype=torch.float)
            >>> Perm = np.random.random([32,32])
            >>> meas_op = HadamSplit(400, 32,  Perm)
            >>> split_op = SplitPoisson(10, meas_op)
            >>> m = split_op(x)
            >>> print(m.shape)
            torch.Size([10, 400])
        """
        s = x.shape[:-1] + torch.Size([self.M]) # torch.Size([*,M])
        H_ones = self.H_ones.expand(s)
        
        # unsplit
        x = x[...,self.even_index] - x[...,self.odd_index]
        # normalize
        x = 2*x/self.alpha - H_ones
        return x
    
    def forward_expe(self, 
                     x: torch.tensor, 
                     meas_op: Union[LinearSplit, HadamSplit],
                     dim = -1
                     ) -> Tuple[torch.tensor, torch.tensor]:
        r""" 
        [Update] Preprocess to compensate for image normalization and splitting of the 
        measurement operator.
        
        It computes :math:`m = \frac{x[0::2]-x[1::2]}{\alpha}`, where 
        :math:`\alpha = \max H^\dagger (x[0::2]-x[1::2])`. 
        
        Contrary to :meth:`~forward`, the image intensity :math:`\alpha` 
        is estimated from the pseudoinverse of the unsplit measurements. This 
        method is typically called for the reconstruction of experimental 
        measurements, while :meth:`~forward` is called in simulations. 
        
        The method returns a tuple containing both :math:`m` and :math:`\alpha`
        
        Args:
            :attr:`x`: batch of measurement vectors
            
            :attr:`meas_op`: measurement operator (required to estimate 
            :math:`\alpha`)
                
            Output (:math:`m`, :math:`\alpha`): preprocess measurement and estimated 
            intensities.
            
            :attr:`meas_op`: dimensions across which the maximum is computed
        
        Shape:
            x: :math:`(*, 2M)` where :math:`B` is the batch dimension
            
            meas_op: the number of measurements :attr:`meas_op.M` should match
            :math:`M`.
            
            :math:`m`: :math:`(*, M)`
            
            :math:`\alpha`: :math:`(*, 1)` 
        
        Example:
            >>> x = torch.rand([10,2*400], dtype=torch.float)
            >>> Perm = np.random.random([32,32])
            >>> meas_op = HadamSplit(400, 32,  Perm)
            >>> split_op = SplitPoisson(10, meas_op)
            >>> m, alpha = split_op.forward_expe(x, meas_op)
            >>> print(m.shape)
            >>> print(alpha.shape)
            torch.Size([10, 400])
            torch.Size([10, 1])
        """
               
        # unsplit
        x = x[..., self.even_index] - x[..., self.odd_index]
        
        # estimate intensity x gain (in counts)
        x_pinv = meas_op.pinv(x)
        alpha = torch.amax(x_pinv, dim=dim, keepdim=True) - torch.amin(x_pinv, dim=dim, keepdim=True)
        #alpha = alpha/2 # does it make sense ??? NO
        x = x/alpha
        x = 2*x - self.H_ones

        return x, alpha
    
    def forward_expe2(self, 
                     x: torch.tensor, 
                     meas_op: Union[LinearSplit, HadamSplit],
                     dim = -1
                     ) -> Tuple[torch.tensor, torch.tensor]:

        # estimate intensity (in counts)
        z = x[..., self.even_index] + x[..., self.odd_index]
        mu = torch.mean(z, dim, keepdim=True)
        alpha = (2/self.N)*(mu - 2*self.mudark)/self.gain
        
        # alternative based on the variance
        #var = torch.var(z, dim, keepdim=True)
        #alpha_2 = (2/self.N)*(var - 2*self.sigdark**2)/self.gain**2
        
        #gain = (var - 2*self.sigdark**2)/(mu - 2*self.mudark)
        
        # Alternative where all rows of an image have the same normalization
        alpha = torch.amax(alpha, -2, keepdim=True)
        
        # intensity x gain (in counts)
        norm = alpha*self.gain
        
        # unsplit
        x = x[..., self.even_index] - x[..., self.odd_index]
        
        # normalize
        x = x / norm
        x = 2*x - self.H_ones

        return x, norm  # or alpha? Double check.
    
    def sigma(self, x: torch.tensor) -> torch.tensor:
        r""" Estimates the variance of the preprocessed measurements 
        
        The variance is estimated as :math:`\frac{4}{\alpha^2} H(x[0::2]+x[1::2])`
        
        Args:
            :attr:`x`: batch of images in the Hadamard domain
            
        Shape:
            - Input: :math:`(*,2*M)` where :math:`*` indicates one or more dimensions
            - Output: :math:`(*, M)`
            
            
        Example:
            >>> x = torch.rand([10,2*400], dtype=torch.float)
            >>> v = split_op.sigma(x)
            >>> print(v.shape)
            torch.Size([10, 400])
            
        """
        x = x[...,self.even_index] + x[...,self.odd_index]
        x = 4*x/(self.alpha**2) # Cov is in [-1,1] so *4
        return x
    
    def set_expe(self, gain=1.0, mudark=0.0, sigdark=0.0, nbin=1.0):
        r"""
        Sets experimental parameters of the sensor
        
        Args:        
            - :attr:`gain` (float): gain (in count/electron)
            - :attr:`mudark` (float): average dark current (in counts)
            - :attr:`sigdark` (float): standard deviation or dark current (in counts)
            - :attr:`nbin` (float): number of raw bin in each spectral channel (if input x results from the sommation/binning of the raw data)
        
        Example:
            >>> split_op.set_expe(gain=1.6)
            >>> print(split_op.gain)
            1.6
        """
        self.gain = gain
        self.mudark = mudark
        self.sigdark = sigdark
        self.nbin = nbin
        
        
    def sigma_expe(self, x: torch.tensor) -> torch.tensor:
        r"""
        Estimates the variance of the measurements that are compensated for 
        splitting but **NOT** for image normalization
        
        
        Args:
            :attr:`x`: Batch of images in the Hadamard domain.
            
        Shape:
            Input: :math:`(*,2*M)` where :math:`*` indicates one or more dimensions
            
            Output: :math:`(*, M)`
            
        Example:
            >>> x = torch.rand([10,2*32*32], dtype=torch.float)
            >>> split_op.set_expe(gain=1.6)
            >>> v = split_op.sigma_expe(x)
            >>> print(v.shape)
            torch.Size([10, 400])
        """
        # Input shape (b*c, 2*M)
        # output shape (b*c, M)
        x = x[...,self.even_index] + x[...,self.odd_index]
        x = self.gain*(x - 2*self.nbin*self.mudark) + 2*self.nbin*self.sigdark**2
        x = 4*x     # to get the cov of an image in [-1,1], not in [0,1]

        return x

    def sigma_from_image(self, 
                         x: torch.tensor, 
                         meas_op: Union[LinearSplit, HadamSplit]
                         )-> torch.tensor:
        r"""
        Estimates the variance of the preprocessed measurements corresponding
        to images through a measurement operator
        
        The variance is estimated as 
        :math:`\frac{4}{\alpha} \{(Px)[0::2] + (Px)[1::2]\}`
        
        Args:
            :attr:`x`: Batch of images
            
            :attr:`meas_op`: Measurement operator 
            
        Shape:
            :attr:`x`: :math:`(*,N)`
            
            :attr:`meas_op`: An operator such that :attr:`meas_op.N` :math:`=N` 
            and :attr:`meas_op.M` :math:`=M`
            
            Output: :math:`(*, M)`
            
        Example:
            >>> x = torch.rand([10,2*400], dtype=torch.float)
            >>> Perm = np.random.random([32,32])
            >>> meas_op = HadamSplit(400, 32,  Perm)
            >>> split_op = SplitPoisson(10, meas_op)
            >>> v = split_op.sigma_from_image(x, meas_op)
            >>> print(v.shape)
            torch.Size([10, 400])
        
        """        
        x = meas_op(x);
        x = x[:,self.even_index] + x[:,self.odd_index]
        x = 4*x/self.alpha  # here alpha should not be squared
        return x
   
    
    def denormalize_expe(self, x, beta):
        r""" 
        Denormalize images from the range [-1;1] to the range [0; :math:`\beta`]
        
        It computes :math:`m = \frac{\beta}{2}(x+1)`, where 
        :math:`\beta` is the normalization factor. 
        
        Args:
            - :attr:`x`: Batch of images
            - :attr:`beta`: Normalizarion factor
            - :attr:`h`: Image height
            - :attr:`w`: Image width
        
        Shape:
            - :attr:`x`: :math:`(*, 1, h, w)`
            - :attr:`beta`: :math:`(*)` or :math:`(*, 1)` 
            - :attr:`h`: int
            - :attr:`w`: int
            - Output: :math:`(*, 1, h, w)`
        
        Example:
            >>> x = torch.rand([10, 1, 32,32], dtype=torch.float)
            >>> beta = 9*torch.rand_like(x)
            >>> y = split_op.denormalize_expe(x, beta, 32, 32)
            >>> print(y.shape)
            torch.Size([10, 1, 32, 32])
                        
        """
        x = (x+1)/2*beta
        
        return x
#==============================================================================    
class SplitRowPoisson(nn.Module):
#==============================================================================
    r"""
        Preprocess raw data acquired with a split measurement operator
        
        It computes :math:`m = \frac{y_{+}-y_{-}}{\alpha}` and the variance
        :math:`\sigma^2 = \frac{2(y_{+} + y_{-})}{\alpha^{2}}`, where 
        :math:`y_{+} = H_{+}x` and :math:`y_{-} = H_{-}x` are obtained using 
        a split measurement operator such as :class:`spyrit.core.LinearRowSplit`.
            
        Args:
            - :math:`\alpha` (float): maximun image intensity (in counts)
            - :math:`M` (int): number of measurements
            - :math:`h` (int): number of rows in the image, i.e., image height
            
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
            >>> meas_op = LinearRowSplit(H_pos, H_neg)
            >>> m = split_op(x, meas_op)
            >>> print(m.shape)
            torch.Size([10, 24, 64])

        """
        # unsplit
        x = x[:,self.even_index] - x[:,self.odd_index]
        # normalize
        e = torch.ones([x.shape[0], meas_op.N, self.h], device=x.device)
        x = 2*x/self.alpha - meas_op.forward_H(e)
        return x