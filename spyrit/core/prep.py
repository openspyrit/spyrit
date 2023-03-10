import torch
import torch.nn as nn
from spyrit.core.meas import Linear, LinearSplit, LinearRowSplit, HadamSplit
from typing import Union, Tuple

#==============================================================================
class SplitPoisson(nn.Module):
#==============================================================================
    r"""
    Preprocess raw data acquired with a split measurement operator
    
    It computes :math:`m = \frac{y_{+}-y_{-}}{\alpha}` and the variance
    :math:`var = \frac{2(y_{+} + y_{-})}{\alpha^{2}}`, where 
    :math:`y_{+} = H_{+}x` and :math:`y_{-} = H_{-}x` are obtained using a 
    split measurement operator (see :mod:`spyrit.core.LinearSplit`) and 
    :math:`\alpha` is the image intensity.
    
    It also compensates for the affine transformation applied to x to get 
    positive intensities.

    
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
        r""" 
        Preprocess to compensates for image normalization and splitting of the 
        measurement operator.
        
        It computes :math:`\frac{x[0::2]-x[1::2]}{\alpha}`
        
        Args:
            :attr:`x`: batch of images in the Hadamard domain 
            
            :attr:`meas_op`: measurement operator
            
        Shape:
            x: :math:`(B, 2M)` where :math:`B` is the batch dimension
            
            meas_op: the number of measurements :attr:`meas_op.M` should match
            :math:`M`.
            
            Output: :math:`(B, M)`
            
        Example:
            >>> x = torch.rand([10,2*400], dtype=torch.float)
            >>> H = np.random.random([400,32*32])
            >>> meas_op =  LinearSplit(H)
            >>> m = split_op(x, meas_op)
            torch.Size([10, 400])
            
        Example 2:
            >>> Perm = np.random.random([32*32,32*32])
            >>> meas_op = HadamSplit(H, Perm, 32, 32)
            >>> m = split_op(x, meas_op)
            >>> print(m.shape)
            torch.Size([10, 400])
        """
        # unsplit
        x = x[:,self.even_index] - x[:,self.odd_index]
        # normalize
        x = 2*x/self.alpha - meas_op.H(torch.ones(x.shape[0], self.N).to(x.device))
        return x
    
    def forward_expe(self, 
                     x: torch.tensor, 
                     meas_op: Union[LinearSplit, HadamSplit]
                     ) -> Tuple[torch.tensor, torch.tensor]:
        r""" 
        Preprocess to compensate for image normalization and splitting of the 
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
        
        Shape:
            x: :math:`(B, 2M)` where :math:`B` is the batch dimension
            
            meas_op: the number of measurements :attr:`meas_op.M` should match
            :math:`M`.
            
            :math:`m`: :math:`(B, M)`
            
            :math:`\alpha`: :math:`(B)` 
        
        Example:
            >>> Perm = np.random.random([32*32,32*32])
            >>> meas_op = HadamSplit(H, Perm, 32, 32)
            >>> m, alpha = split_op.forward_expe(x, meas_op)
            >>> print(m.shape)
            >>> print(alpha.shape)
            torch.Size([10, 400])
            torch.Size([10])
        """
        bc = x.shape[0]
        
        # unsplit
        x = x[:,self.even_index] - x[:,self.odd_index]
        
        # estimate alpha
        x_pinv = meas_op.pinv(x)
        alpha = self.max(x_pinv)
        alpha = alpha.expand(bc,self.M) # shape is (b*c, M)
        
        # normalize
        x = torch.div(x, alpha)
        x = 2*x - meas_op.H(torch.ones(bc, self.N).to(x.device))
        
        alpha = alpha[:,0]    # shape is (b*c,)

        return x, alpha
    
    def sigma(self, x: torch.tensor) -> torch.tensor:
        r""" Estimates the variance of the preprocessed measurements 
        
        The variance is estimated as :math:`\frac{4}{\alpha^2} H(x[0::2]+x[1::2])`
        
        Args:
            :attr:`x`: batch of images in the Hadamard domain
            
        Shape:
            - Input: :math:`(B,2*M)` where :math:`B` is the batch dimension
            - Output: :math:`(B, M)`
            
        Example:
            >>> x = torch.rand([10,2*400], dtype=torch.float)
            >>> v = split_op.sigma(x)
            >>> print(v.shape)
            torch.Size([10, 400])
            
        """
        x = x[:,self.even_index] + x[:,self.odd_index]
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
            Input: :math:`(B,2*M)` where :math:`B` is the batch dimension
            
            Output: :math:`(B, M)`
            
        Example:
            >>> x = torch.rand([10,2*32*32], dtype=torch.float)
            >>> split_op.set_expe(gain=1.6)
            >>> v = split_op.sigma_expe(x)
            >>> print(v.shape)
            torch.Size([10, 400])
        """
        # Input shape (b*c, 2*M)
        # output shape (b*c, M)
        x = x[:,self.even_index] + x[:,self.odd_index]
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
            >>> x = torch.rand([10,32*32], dtype=torch.float)
            >>> Perm = np.random.random([32*32,32*32])
            >>> meas_op = HadamSplit(H, Perm, 32, 32)
            >>> v = split_op.sigma_from_image(x, meas_op)
            >>> print(v.shape)
            torch.Size([10, 400])
        
        """        
        x = meas_op(x);
        x = x[:,self.even_index] + x[:,self.odd_index]
        x = 4*x/self.alpha  # here alpha should not be squared
        return x
   
    
    def denormalize_expe(self, x, beta, h, w):
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
            >>> beta = 9*torch.rand([10])
            >>> y = split_op.denormalize_expe(x, beta, 32, 32)
            >>> print(y.shape)
            torch.Size([10, 1, 32, 32])
                        
        """
        bc = x.shape[0]
        
        # Denormalization
        beta = beta.view(bc,1,1,1)
        beta = beta.expand(bc,1,h,w)
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
        x = 2*x - meas_op.H(torch.ones(x.shape[0], self.N).to(x.device)) # to shift images in [-1,1]^N
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
