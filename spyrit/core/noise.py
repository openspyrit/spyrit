"""
Noise models

.. math::

    y \sim \mathcal{N}\left(z;\theta\right),
    
where :math:`\mathcal{N}` the noise distribution, :math:`z` represents the 
noiseless measurements, and :math:`\theta` represents the parameters
of the noise distribution.

There are two main classes in this module, which simulate Gaussian and Poisson
noise.
"""

import torch
import torch.nn as nn
# import torch.nn.functional as F

# ==============================================================================
class Gaussian(nn.Module):
    r"""
    Simulate measurements corrupted by additive Gaussian noise

    .. math::

        y \sim \mathcal{N}\left(\mu = z, \sigma^2\right),

    where :math:`\mathcal{N}(\mu, \sigma^2)` is a Gaussian distribution
    with mean :math:`\mu` and variance :math:`\sigma^2` and :math:`z` are
    the noiseless measurements.

    The class is constructed from the standard deviation of the noise :math:`\sigma`.

    Args:
        :attr:`sigma` (:class:`float`): Standard deviation of the noise :math:`\sigma`

    Example:
        >>> noise = Gaussian(1.0)
        >>> z = torch.tensor([1, 3, 6])
        >>> y = noise(z)
        >>> print(y)
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, z):
        r"""Simulates measurements corrupted by additive Gaussian noise

        .. math::

            y \sim \mathcal{N}\left(\mu = z, \sigma^2\right),

        where :math:`\mathcal{N}(\mu, \sigma^2)` is a Gaussian distribution
        with mean :math:`\mu` and variance :math:`\sigma^2` and :math:`z` are
        the noiseless measurements.

        Args:
            :attr:`z` (:class:`torch.tensor`): Noiseless measurements :math:`z` with arbitrary shape.

        Output:
            :class:`torch.tensor`: Noisy measurement :math:`y` with the same shape as :attr:`z`.

        Example: 
            Two different noisy measurement vectors
            
            >>> import spyrit.core.noise as sn
            >>> import torch
            >>> noise = sn.Gaussian(0.1)
            >>> z = torch.empty(10, 4).uniform_(1, 2)
            >>> y = noise(z)
            >>> print(y.shape)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            torch.Size([10, 4])
            Measurements in (0.86 , 2.09)
            
            >>> y = noise(z)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Measurements in (1.01 , 1.98)
        """
        return z + self.sigma * torch.randn(z.shape)


# # =============================================================================
# class PoissonApproxGauss(NoNoise):
#     # =========================================================================
#     r"""
#     Simulates measurements corrupted by gaussian-approximated Poisson noise.

#     To accelerate the computation, we consider a Gaussian approximation to the Poisson
#     distribution.

#     Assuming incoming images :math:`x` in the range [-1;1], measurements are
#     first simulated for images in the range [0; :math:`\alpha`]:
#     :math:`y = \frac{\alpha}{2} P(1+x)`. Then, Gaussian noise
#     is added: :math:`y + \sqrt{y} \cdot \mathcal{G}(\mu=0,\sigma^2=1)`.

#     The class is constructed from a measurement operator :math:`P` and
#     an image intensity :math:`\alpha` that controls the noise level.

#     .. warning::
#         Assumes that the incoming images :math:`x` are in the range [-1;1]

#     Args:
#         :attr:`meas_op`: Measurement operator :math:`H` (see the :mod:`~spyrit.core.meas` submodule)

#         :attr:`alpha` (float): Image intensity (in photoelectrons)

#     Example 1: Using a :class:`~spyrit.core.meas.Linear` measurement operator
#         >>> H = torch.rand([400,32*32])
#         >>> meas_op = Linear(H)
#         >>> noise_op = PoissonApproxGauss(meas_op, 10.0)

#     Example 2: Using a :class:`~spyrit.core.meas.HadamSplit` operator
#         >>> Perm = torch.rand([32*32,32*32])
#         >>> meas_op = HadamSplit(H, Perm, 32, 32)
#         >>> noise_op = PoissonApproxGauss(meas_op, 200.0)

#     Example 3: Using a :class:`~spyrit.core.meas.LinearSplit` operator
#         >>> H = torch.rand(24,64)
#         >>> meas_op = LinearSplit(H)
#         >>> noise_op = PoissonApproxGauss(meas_op, 50.0)
#     """

#     def __init__(
#         self,
#         meas_op: Union[Linear, LinearSplit, HadamSplit],
#         alpha: float,
#     ):
#         super().__init__(meas_op)
#         self.alpha = alpha

#     def forward(self, x: torch.tensor) -> torch.tensor:
#         r"""
#         Simulates measurements corrupted by Poisson noise

#         Args:
#             :attr:`x`: Batch of images. The input is directly passed to the
#             measurement operator, so its shape depends on the type of the
#             measurement operator.

#         Shape:
#             :attr:`x`: :math:`(*, h, w)` if `self.meas_op` is a static
#             measurement operator, :math:`(*, t, c, h, w)` if it is a dynamic
#             measurement operator.

#             :attr:`Output`: :math:`(*, M)` (static measurements) or `(*, c, M)`
#             (dynamic measurements)

#         Example 1: Two noisy measurement vectors from a :class:`~spyrit.core.meas.Linear` measurement operator
#             >>> H = torch.rand([400,32*32])
#             >>> meas_op = Linear(H)
#             >>> noise_op = PoissonApproxGauss(meas_op, 10.0)
#             >>> x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
#             >>> y = noise_op(x)
#             >>> print(y.shape)
#             >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
#             >>> y = noise_op(x)
#             >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
#             torch.Size([10, 400])
#             Measurements in (2255.57 , 2911.18)
#             Measurements in (2226.49 , 2934.42)

#         Example 2: Two noisy measurement vectors from a :class:`~spyrit.core.meas.HadamSplit` operator
#             >>> Perm = torch.rand([32*32,32*32])
#             >>> meas_op = HadamSplit(H, Perm, 32, 32)
#             >>> noise_op = PoissonApproxGauss(meas_op, 200.0)
#             >>> x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
#             >>> y = noise_op(x)
#             >>> print(y.shape)
#             >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
#             >>> y = noise_op(x)
#             >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
#             torch.Size([10, 800])
#             Measurements in (0.00 , 55951.41)
#             Measurements in (0.00 , 56216.86)

#         Example 3: Two noisy measurement vectors from a :class:`~spyrit.core.meas.LinearSplit` operator
#             >>> H = torch.rand(24,64)
#             >>> meas_op = LinearSplit(H)
#             >>> noise_op = PoissonApproxGauss(meas_op, 50.0)
#             >>> x = torch.FloatTensor(10, 64, 92).uniform_(-1, 1)
#             >>> y = noise_op(x)
#             >>> print(y.shape)
#             >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
#             >>> y = noise_op(x)
#             >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
#             torch.Size([10, 48, 92])
#             Measurements in (460.43 , 1216.94)
#             Measurements in (441.85 , 1230.43)
#         """
#         x = super().forward(x)  # NoNoise forward, scaling to [0, 1]
#         x *= self.alpha
#         x = F.relu(x)  # remove small negative values
#         x = x + torch.sqrt(x) * torch.randn_like(x)
#         return x


# # =============================================================================
# class PoissonApproxGaussSameNoise(NoNoise):
#     # =========================================================================
#     r"""
#     Simulates measurements corrupted by identical Gaussian-approximated Poisson noise.

#     To accelerate the
#     computation, we consider a Gaussian approximation to the Poisson
#     distribution. Contrary to :class:`~spyrit.core.noise.PoissonApproxGauss`,
#     all measurements in a batch are corrupted with the same noise sample, i.e.
#     the noise depends only on the measurement number (and order).

#     Assuming incoming images :math:`x` in the range [-1;1], measurements are
#     first simulated for images in the range [0; :math:`\alpha`]:
#     :math:`y = \frac{\alpha}{2} P(1+x)`. Then, Gaussian noise
#     is added: :math:`y + \sqrt{y} \cdot \mathcal{G}(\mu=0,\sigma^2=1)`.

#     The class is constructed from a measurement operator :math:`P` and
#     an image intensity :math:`\alpha` that controls the noise level.

#     .. warning::
#         Assumes that the incoming images :math:`x` are in the range [-1;1]

#     Args:
#         :attr:`meas_op`: Measurement operator :math:`H` (see the :mod:`~spyrit.core.meas` submodule)

#         :attr:`alpha` (float): Image intensity (in photoelectrons)

#     Example 1: Using a :class:`~spyrit.core.meas.Linear` measurement operator
#         >>> H = torch.rand([400,32*32])
#         >>> meas_op = Linear(H)
#         >>> noise_op = PoissonApproxGaussSameNoise(meas_op, 10.0)

#     Example 2: Using a :class:`~spyrit.core.meas.HadamSplit` operator
#         >>> Perm = torch.rand([32*32,32*32])
#         >>> meas_op = HadamSplit(H, Perm, 32, 32)
#         >>> noise_op = PoissonApproxGaussSameNoise(meas_op, 200.0)
#     """

#     def __init__(self, meas_op: Union[Linear, LinearSplit, HadamSplit], alpha: float):
#         super().__init__(meas_op)
#         self.alpha = alpha

#     def forward(self, x: torch.tensor) -> torch.tensor:
#         r"""
#         Simulates measurements corrupted by Poisson noise

#         Args:
#             :attr:`x`: Batch of images. The input is directly passed to the
#             measurement operator, so its shape depends on the type of the
#             measurement operator.

#         Shape:
#             :attr:`x`: :math:`(*, h, w)` if `self.meas_op` is a static
#             measurement operator, :math:`(*, t, c, h, w)` if it is a dynamic
#             measurement operator.

#             :attr:`Output`: :math:`(*, M)` (static measurements) or `(*, c, M)`
#             (dynamic measurements)

#         Example 1: Two noisy measurement vectors from a :class:`~spyrit.core.meas.Linear` measurement operator
#             >>> H = torch.rand([400,32*32])
#             >>> meas_op = Linear(H)
#             >>> noise_op = PoissonApproxGaussSameNoise(meas_op, 10.0)
#             >>> x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
#             >>> y = noise_op(x)
#             >>> print(y.shape)
#             >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
#             >>> y = noise_op(x)
#             >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
#             torch.Size([10, 400])
#             Measurements in (2255.57 , 2911.18)
#             Measurements in (2226.49 , 2934.42)

#         Example 2: Two noisy measurement vectors from a :class:`~spyrit.core.meas.HadamSplit` operator
#             >>> Perm = torch.rand([32*32,32*32])
#             >>> meas_op = HadamSplit(H, Perm, 32, 32)
#             >>> noise_op = PoissonApproxGaussSameNoise(meas_op, 200.0)
#             >>> x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
#             >>> y = noise_op(x)
#             >>> print(y.shape)
#             >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
#             >>> y = noise_op(x)
#             >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
#             torch.Size([10, 800])
#             Measurements in (0.00 , 55951.41)
#             Measurements in (0.00 , 56216.86)
#         """
#         x = super().forward(x)  # NoNoise forward
#         x *= self.alpha
#         x = F.relu(x)  # remove small negative values
#         x = x + torch.sqrt(x) * torch.randn((*[1] * (x.ndim - 1), x.shape[-1]))
#         return x


# =============================================================================
class Poisson(nn.Module):
    r"""Simulate measurements corrupted by Poisson noise

    .. math::
        y \sim \mathcal{P}\left(\alpha z\right), \quad \text{with }z\ge 0
        
    where :math:`\mathcal{P}` is the Poisson distribution and :math:`\alpha` represents the intensity of the noiseless measurements :math:`z`.
    
    The class is constructed from the intensity :math:`\alpha`.

    Args:
        :attr:`alpha` (:class:`float`): The intensity of the measurements.

    Attributes:
        :attr:`alpha` (:class:`float`): Intensity of the measurements.
        
    Example:
        >>> noise = Poisson(10.0)
        >>> z = torch.tensor([1, 3, 6])
        >>> y = noise(z)
        >>> print(y)
        tensor([11., 32., 57.])
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, z: torch.tensor) -> torch.tensor:
        r"""Corrupt measurement by Poisson noise
        
        .. math::
            y \sim \mathcal{P}\left(\alpha z\right).

        Args:
            :attr:`z` (:class:`torch.tensor`): Measurements :math:`z` with arbitrary shape.

        Returns:
            :class:`torch.tensor`: Noisy measurement :math:`y` with the same shape as :attr:`z`.
            
        Example: 
            Two different noisy measurement vectors
            
            >>> import spyrit.core.noise as sn
            >>> import torch
            >>> noise = sn.Poisson(100)
            >>> z = torch.empty(10, 4).uniform_(0, 1)
            >>> y = noise(z)
            >>> print(y.shape)
            >>> print(f"Noiseless measurements in ({torch.min(z):.2f} , {torch.max(z):.2f})")
            >>> print(f"Noisy measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            torch.Size([10, 4])
            Noiseless measurements in (0.03 , 0.97)
            Noisy measurements in (3.00 , 96.00)
            
            >>> y = noise(z)
            >>> print(f"Noisy measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Noisy measurements in (2.00 , 124.00)
        """
        return torch.poisson(self.alpha*z)


# =============================================================================
class PoissonApproxGauss(Poisson):
    r"""Gaussian approximation of Poisson noise

    .. math::
        y \sim  \alpha z  + \sqrt{\alpha z} \cdot \mathcal{N}(0, 1), \quad \text{with }z\ge 0

    where  :math:`\alpha` represents the intensity of the noiseless     
    measurements :math:`z`, and :math:`\mathcal{N}(0, 1)` is a Gaussian
    distribution with zero mean and unit variance.
    
    This is an approximation of :math:`y \sim \mathcal{P}\left(\alpha z\right)`, where :math:`\mathcal{P}` is the Poisson distribution. Computing the Gaussian approximation is faster than the original Poisson model. 

    Args:
        :attr:`alpha` (:class:`float`): The intensity of the measurements.

    Attributes:
        :attr:`alpha` (:class:`float`): Intensity of the measurements.

    """

    def __init__(self, alpha: float):
        super().__init__(alpha)

    def forward(self, z: torch.tensor) -> torch.tensor:
        r"""Corrupt measurement by Gaussian approximation of Poisson noise

        .. math::
            y \sim  \alpha z  + \sqrt{\alpha z} \cdot \mathcal{N}(0, 1) \quad \text{with }z\ge 0

        Args:
            :attr:`z` (:class:`torch.tensor`): Measurements :math:`z` with 
            arbitrary shape.

        Returns:
            :class:`torch.tensor`: Noisy measurement :math:`y` with the same 
            shape as :attr:`z`.
            
        Raises:
             RuntimeError: If there are negative values in the input tensor.
             
        Example: 
            Two different noisy measurement vectors
            
            >>> import spyrit.core.noise as sn
            >>> import torch
            >>> noise = sn.PoissonApproxGauss(100)
            >>> z = torch.empty(10, 4).uniform_(0, 1)
            >>> y = noise(z)
            >>> print(y.shape)
            >>> print(f"Noiseless measurements in ({torch.min(z):.2f} , {torch.max(z):.2f})")
            >>> print(f"Noisy measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            torch.Size([10, 4])
            Noiseless measurements in (0.06 , 0.96)
            Noisy measurements in (3.63 , 116.96)
            
            >>> y = noise(z)
            >>> print(f"Noisy measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Noisy measurements in (3.25 , 110.16)
        """
        if torch.any(z < 0):
            raise RuntimeError("Input tensor contains negative values.")
        
        return self.alpha*z + torch.sqrt(self.alpha*z) * torch.randn_like(z)


# =============================================================================
class PoissonApproxGaussSameNoise(Poisson):
    r"""Gaussian approximation of Poisson noise

    .. math::
        y \sim  \alpha z  + \sqrt{\alpha z} \cdot \mathcal{N}(0, 1), \quad \text{with }z\ge 0

    where  :math:`\alpha` represents the intensity of the noiseless     
    measurements :math:`z`, and :math:`\mathcal{N}(0, 1)` is a Gaussian
    distribution with zero mean and unit variance.
    
    This is an approximation of :math:`y \sim \mathcal{P}\left(\alpha z\right)`, where :math:`\mathcal{P}` is the Poisson distribution. Computing the Gaussian approximation is faster than the original Poisson model. 
    
    .. important::
        Contrary to :class:`~spyrit.core.noise.PoissonApproxGauss`, 
        different noise realisations apply only to the last dimension of 
        the input tensor. The same noise realisations are repeated to the 
        first dimensions of the input tensor.

    Args:
        :attr:`alpha` (:class:`float`): The intensity of the measurements.

    Attributes:
        :attr:`alpha` (:class:`float`): Intensity of the measurements.
    """

    def __init__(self, alpha: float):
        super().__init__(alpha)

    def forward(self, z: torch.tensor) -> torch.tensor:
        r"""Corrupt measurement by Gaussian approximation of Poisson noise

        .. math::
            y \sim  \alpha z  + \sqrt{\alpha z} \cdot \mathcal{N}(0, 1) \quad \text{with }z\ge 0

        Args:
            :attr:`z` (:class:`torch.tensor`): Measurements :math:`z` with 
            arbitrary shape.

        Returns:
            :class:`torch.tensor`: Noisy measurement :math:`y` with the same 
            shape as :attr:`z`.
            
        .. important::
            Contrary to :class:`~spyrit.core.noise.PoissonApproxGauss`, 
            different noise realisations apply only to the last dimension of 
            the input tensor. The same noise realisations are repeated to the 
            first dimensions of the input tensor.
            
        Raises:
             RuntimeError: If there are negative values in the input tensor.
             
        Example: 
            Two different noisy measurement vectors
            
            >>> import spyrit.core.noise as sn
            >>> import torch
            >>> noise = sn.PoissonApproxGaussSameNoise(100)
            >>> z = torch.empty(10, 4).uniform_(0, 1)
            >>> y = noise(z)
            >>> print(y.shape)
            >>> print(f"Noiseless measurements in ({torch.min(z):.2f} , {torch.max(z):.2f})")
            >>> print(f"Noisy measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            torch.Size([10, 4])
            Noiseless measurements in (0.13 , 0.98)
            Noisy measurements in (10.74 , 108.50)
            
            >>> y = noise(z)
            >>> print(f"Noisy measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Noisy measurements in (9.95 , 103.54)
        """
        if torch.any(z < 0):
            raise RuntimeError("Input tensor contains negative values.")
        y = torch.sqrt(self.alpha*z)
        y = y*torch.randn((*[1]*(z.ndim - 1),z.shape[-1]))
        y = y + self.alpha*z
        return y
