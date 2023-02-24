import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import poisson
from spyrit.core.meas import Linear, LinearSplit, LinearRowSplit, HadamSplit
from typing import Union

# =====================================================================================================================
# NoNoise
# =====================================================================================================================       
class NoNoise(nn.Module):
    r"""
    Simulates measurements from images in the range [0;1] by computing  
    :math:`y = \frac{1}{2} H(1+x)`.
    
    .. note::
        Assumes that the incoming images :math:`x` are in the range [-1;1]
        
    The class is constructed from a measuremznt operator (see the
    :mod:`~spyrit.core.meas` submodule)
    
    Args:
        :attr:`meas_op` : Measurement operator (see the
        :mod:`~spyrit.core.meas` submodule)
            
    Example 1: Using a :class:`~spyrit.core.meas.Linear` measurement operator
        >>> H =np.random.random([400,32*32])
        >>> linear_op = Linear(H)
        >>> linear_acq = NoNoise(linear_op)
        
    Example 2: Using a :class:`~spyrit.core.meas.HadamSplit` measurement operator
        >>> H = np.random.random([400,32*32])
        >>> Perm = np.random.random([32*32,32*32])
        >>> split_op = HadamSplit(H, Perm, 32, 32)
        >>> split_acq = NoNoise(split_op)
    """
    def __init__(self, meas_op: Union[Linear, 
                                      LinearSplit, 
                                      HadamSplit, 
                                      LinearRowSplit]):
        super().__init__()
        self.meas_op = meas_op
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates measurements
        
        
        Args:
            :attr:`x`: Batch of images
            
        Shape:
            - :attr:`x`: :math:`(*, N)` 
            - :attr:`Output`: :math:`(*, M)`
            
        Example 1: Using a :class:`~spyrit.core.meas.Linear` measurement operator
            >>> x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
            >>> y = linear_acq(x)
            >>> print(y.shape)
            torch.Size([10, 400])
            
        Example 2: Using a :class:`~spyrit.core.meas.HadamSplit` measurement operator
            >>> x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
            >>> y = split_acq(x)
            >>> print(y.shape)
            torch.Size([10, 800])  
        """
        x = (x+1)/2; 
        x = self.meas_op(x)
        return x

# ==================================================================================
class Poisson(NoNoise):
# ================================================================================== 
    r""" 
    Simulates measurements corrupted by Poisson noise
    
    Assuming incoming images :math:`x` in the range [-1;1], measurements are 
    first simulated for images in the range [0; :math:`\alpha`]. Then, Poisson
    noise is applied: :math:`y = \mathcal{P}(\frac{\alpha}{2} H(1+x))`.
    
    .. note::
        Assumes that the incoming images :math:`x` are in the range [-1;1]
                
    The class is constructed from a measurement operator and an image 
    intensity :math:`\alpha` that controls the noise level.
    
    Args:
        :attr:`meas_op`: Measurement operator :math:`H` (see the :mod:`~spyrit.core.meas` submodule)
        
        :attr:`alpha` (float): Image intensity (in photoelectrons) 
            
    Example 1: Using a :class:`~spyrit.core.meas.Linear` measurement operator
        >>> H =np.random.random([400,32*32])
        >>> linear_op = Linear(H)
        >>> linear_acq = Poisson(linear_op, 10.0)
        
    Example 2: Using a :class:`~spyrit.core.meas.HadamSplit` measurement operator
        >>> H = np.random.random([400,32*32])
        >>> Perm = np.random.random([32*32,32*32])
        >>> split_op = HadamSplit(H, Perm, 32, 32)
        >>> split_acq = Poisson(split_op, 200.0)
        
    Example 3: Using a :class:`~spyrit.core.meas.LinearRowSplit` measurement operator
        >>> H_pos = np.random.rand(24,64)
        >>> H_neg = np.random.rand(24,64)
        >>> split_row_op = LinearRowSplit(H_pos,H_neg)
        >>> split_acq = Poisson(split_row_op, 50.0)
    """
    def __init__(self, meas_op: Union[Linear, 
                                      LinearSplit, 
                                      HadamSplit, 
                                      LinearRowSplit], alpha = 50.0):
        super().__init__(meas_op)
        self.alpha = alpha

    def forward(self, x):
        r"""
        Simulates measurements corrupted by Poisson noise
        
        Args:
            :attr:`x`: Batch of images
            
        Shape:
            - :attr:`x`: :math:`(*, N)` 
            - :attr:`Output`: :math:`(*, M)`
            
        Example 1: Two noisy measurement vectors from a :class:`~spyrit.core.meas.Linear` measurement operator
            >>> H = np.random.random([400,32*32])
            >>> meas_op = Linear(H)
            >>> noise_op = Poisson(meas_op, 10.0)
            >>> x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
            >>> y = noise_op(x)
            >>> print(y.shape)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            >>> y = noise_op(x)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            torch.Size([10, 400])
            Measurements in (2249.00 , 2896.00)
            Measurements in (2237.00 , 2880.00)
            
        Example 2: Two noisy measurement vectors from a :class:`~spyrit.core.meas.HadamSplit` operator
            >>> Perm = np.random.random([32*32,32*32])
            >>> meas_op = HadamSplit(H, Perm, 32, 32)
            >>> noise_op = Poisson(meas_op, 200.0)
            >>> x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
            >>> y = noise_op(x)
            >>> print(y.shape)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            >>> y = noise_op(x)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            torch.Size([10, 800])
            Measurements in (0.00 , 55338.00)
            Measurements in (0.00 , 55077.00)
            
        Example 3: Two noisy measurement vectors from a :class:`~spyrit.core.meas.LinearRowSplit` operator
            >>> H_pos = np.random.rand(24,64)
            >>> H_neg = np.random.rand(24,64)
            >>> meas_op = LinearRowSplit(H_pos,H_neg)
            >>> noise_op = Poisson(meas_op, 50.0)
            >>> x = torch.FloatTensor(10, 64, 92).uniform_(-1, 1)
            >>> y = noise_op(x)
            >>> print(y.shape)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            >>> y = noise_op(x)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            torch.Size([10, 48, 92])
            Measurements in (500.00 , 1134.00)
            Measurements in (465.00 , 1140.00)
        """
        #x = self.alpha*(x+1)/2 
        #x = self.meas_op(x)
        x = super().forward(x)  # NoNoise forward
        x = self.alpha*x
        x = F.relu(x)           # troncate negative values to zero
        x = poisson(x) 
        return x           
    
# ==================================================================================
class PoissonApproxGauss(NoNoise):
# ================================================================================== 
    r""" 
    Simulates measurements corrupted by Poisson noise. To accelerate the 
    computation, we consider a Gaussian approximation to the Poisson 
    distribution.
    
    Assuming incoming images :math:`x` in the range [-1;1], measurements are 
    first simulated for images in the range [0; :math:`\alpha`]: 
    :math:`y = \frac{\alpha}{2} P(1+x)`. Then, Gaussian noise 
    is added: :math:`y + \sqrt{y} \cdot \mathcal{G}(\mu=0,\sigma^2=1)`.
    
    The class is constructed from a measurement operator :math:`P` and 
    an image intensity :math:`\alpha` that controls the noise level.
    
    .. warning::
        Assumes that the incoming images :math:`x` are in the range [-1;1]
    
    Args:
        :attr:`meas_op`: Measurement operator :math:`H` (see the :mod:`~spyrit.core.meas` submodule)
        
        :attr:`alpha` (float): Image intensity (in photoelectrons) 
    
    Example 1: Using a :class:`~spyrit.core.meas.Linear` measurement operator
        >>> H = np.random.random([400,32*32])
        >>> meas_op = Linear(H)
        >>> noise_op = PoissonApproxGauss(meas_op, 10.0)
        
    Example 2: Using a :class:`~spyrit.core.meas.HadamSplit` operator    
        >>> Perm = np.random.random([32*32,32*32])
        >>> meas_op = HadamSplit(H, Perm, 32, 32)
        >>> noise_op = PoissonApproxGauss(meas_op, 200.0)
        
    Example 3: Using a :class:`~spyrit.core.meas.LinearRowSplit` operator
        >>> H_pos = np.random.rand(24,64)
        >>> H_neg = np.random.rand(24,64)
        >>> meas_op = LinearRowSplit(H_pos,H_neg)
        >>> noise_op = PoissonApproxGauss(meas_op, 50.0)            
    """   
    def __init__(self, meas_op: Union[Linear, 
                                      LinearSplit, 
                                      HadamSplit, 
                                      LinearRowSplit], alpha: float):
        super().__init__(meas_op)
        self.alpha = alpha
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates measurements corrupted by Poisson noise        
        
        Args:
            :attr:`x`: Batch of images
            
        Shape:
            - :attr:`x`: :math:`(*, N)` 
            - :attr:`Output`: :math:`(*, M)`
            
        Example 1: Two noisy measurement vectors from a :class:`~spyrit.core.meas.Linear` measurement operator
            >>> H = np.random.random([400,32*32])
            >>> meas_op = Linear(H)
            >>> noise_op = PoissonApproxGauss(meas_op, 10.0)
            >>> x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
            >>> y = noise_op(x)
            >>> print(y.shape)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            >>> y = noise_op(x)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            torch.Size([10, 400])
            Measurements in (2255.57 , 2911.18)
            Measurements in (2226.49 , 2934.42)
            
        Example 2: Two noisy measurement vectors from a :class:`~spyrit.core.meas.HadamSplit` operator    
            >>> Perm = np.random.random([32*32,32*32])
            >>> meas_op = HadamSplit(H, Perm, 32, 32)
            >>> noise_op = PoissonApproxGauss(meas_op, 200.0)
            >>> x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
            >>> y = noise_op(x)
            >>> print(y.shape)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            >>> y = noise_op(x)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            torch.Size([10, 800])
            Measurements in (0.00 , 55951.41)
            Measurements in (0.00 , 56216.86)
            
        Example 3: Two noisy measurement vectors from a :class:`~spyrit.core.meas.LinearRowSplit` operator
            >>> H_pos = np.random.rand(24,64)
            >>> H_neg = np.random.rand(24,64)
            >>> meas_op = LinearRowSplit(H_pos,H_neg)
            >>> noise_op = PoissonApproxGauss(meas_op, 50.0)
            >>> x = torch.FloatTensor(10, 64, 92).uniform_(-1, 1)
            >>> y = noise_op(x)
            >>> print(y.shape)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            >>> y = noise_op(x)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            torch.Size([10, 48, 92])
            Measurements in (460.43 , 1216.94)
            Measurements in (441.85 , 1230.43)
        """
        x = super().forward(x)  # NoNoise forward
        x = self.alpha*x
        x = F.relu(x)           # remove small negative values
        x = x + torch.sqrt(x)*torch.randn_like(x);  
        return x  
# ==================================================================================
class PoissonApproxGaussSameNoise(NoNoise):
# ================================================================================== 
    r""" 
    Simulates measurements corrupted by Poisson noise. To accelerate the 
    computation, we consider a Gaussian approximation to the Poisson 
    distribution. Contrary to :class:`~spyrit.core.noise.PoissonApproxGauss`, 
    all measurements in a batch are corrupted with the same noise sample.
    
    Assuming incoming images :math:`x` in the range [-1;1], measurements are 
    first simulated for images in the range [0; :math:`\alpha`]: 
    :math:`y = \frac{\alpha}{2} P(1+x)`. Then, Gaussian noise 
    is added: :math:`y + \sqrt{y} \cdot \mathcal{G}(\mu=0,\sigma^2=1)`.
    
    The class is constructed from a measurement operator :math:`P` and 
    an image intensity :math:`\alpha` that controls the noise level.
    
    .. warning::
        Assumes that the incoming images :math:`x` are in the range [-1;1]
    
    Args:
        :attr:`meas_op`: Measurement operator :math:`H` (see the :mod:`~spyrit.core.meas` submodule)
        
        :attr:`alpha` (float): Image intensity (in photoelectrons) 
    
    Example 1: Using a :class:`~spyrit.core.meas.Linear` measurement operator
        >>> H = np.random.random([400,32*32])
        >>> meas_op = Linear(H)
        >>> noise_op = PoissonApproxGaussSameNoise(meas_op, 10.0)
        
    Example 2: Using a :class:`~spyrit.core.meas.HadamSplit` operator    
        >>> Perm = np.random.random([32*32,32*32])
        >>> meas_op = HadamSplit(H, Perm, 32, 32)
        >>> noise_op = PoissonApproxGaussSameNoise(meas_op, 200.0)
    """   
    def __init__(self, meas_op: Union[Linear, 
                                      LinearSplit, 
                                      HadamSplit], alpha: float):
        super().__init__(meas_op)
        self.alpha = alpha
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates measurements corrupted by Poisson noise        
        
        Args:
            :attr:`x`: Batch of images
            
        Shape:
            - :attr:`x`: :math:`(*, N)` 
            - :attr:`Output`: :math:`(*, M)`
            
        Example 1: Two noisy measurement vectors from a :class:`~spyrit.core.meas.Linear` measurement operator
            >>> H = np.random.random([400,32*32])
            >>> meas_op = Linear(H)
            >>> noise_op = PoissonApproxGaussSameNoise(meas_op, 10.0)
            >>> x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
            >>> y = noise_op(x)
            >>> print(y.shape)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            >>> y = noise_op(x)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            torch.Size([10, 400])
            Measurements in (2255.57 , 2911.18)
            Measurements in (2226.49 , 2934.42)
            
        Example 2: Two noisy measurement vectors from a :class:`~spyrit.core.meas.HadamSplit` operator    
            >>> Perm = np.random.random([32*32,32*32])
            >>> meas_op = HadamSplit(H, Perm, 32, 32)
            >>> noise_op = PoissonApproxGaussSameNoise(meas_op, 200.0)
            >>> x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
            >>> y = noise_op(x)
            >>> print(y.shape)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            >>> y = noise_op(x)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            torch.Size([10, 800])
            Measurements in (0.00 , 55951.41)
            Measurements in (0.00 , 56216.86)
        """
        x = super().forward(x)  # NoNoise forward
        x = self.alpha*x
        x = F.relu(x)           # remove small negative values
        x = x + torch.sqrt(x)*torch.randn(1, x.shape[1])
        return x