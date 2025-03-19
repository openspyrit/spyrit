r"""Noise models

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
        >>> z = torch.tensor([1., 3., 6.])
        >>> y = noise(z)
        >>> print(y)
        tensor([...])
    """

    def __init__(self, sigma: float = 0.1):
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
            torch.Size([10, 4])
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Measurements in (...)

            >>> y = noise(z)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Measurements in (...)
        """
        return torch.normal(z, self.sigma)
        # z + self.sigma * torch.randn(z.shape, device=z.device)


# =============================================================================
class Poisson(nn.Module):
    r"""Simulate measurements corrupted by Poisson noise

    .. math::
        y \sim \mathcal{P}\left(\alpha z\right), \quad \text{with }z\ge 0

    where :math:`\mathcal{P}` is the Poisson distribution and :math:`\alpha` represents the intensity of the noiseless measurements :math:`z`.

    The class is constructed from the intensity :math:`\alpha`.

    Args:
        :attr:`alpha` (:class:`float`): The intensity of the measurements. Defaults to 10.

    Attributes:
        :attr:`alpha` (:class:`float`): Intensity of the measurements.

    Example:
        >>> noise = Poisson(10.0)
        >>> z = torch.tensor([1, 3, 6])
        >>> y = noise(z)
        >>> print(y)
        tensor([...])
    """

    def __init__(self, alpha: float = 10):
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
            torch.Size([10, 4])
            >>> print(f"Noiseless measurements in ({torch.min(z):.2f} , {torch.max(z):.2f})")
            Noiseless measurements in (...)
            >>> print(torch.all((z >= 0) & (z <= 1)))
            tensor(True)
            >>> print(f"Noisy measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Noisy measurements in (...)

            >>> y = noise(z)
            >>> print(f"Noisy measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Noisy measurements in (...)
        """
        return torch.poisson(self.alpha * z)


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
        :attr:`alpha` (:class:`float`): The intensity of the measurements. Defaults to 10.

    Attributes:
        :attr:`alpha` (:class:`float`): Intensity of the measurements.

    """

    def __init__(self, alpha: float = 10):
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
            torch.Size([10, 4])
            >>> print(f"Noiseless measurements in ({torch.min(z):.2f} , {torch.max(z):.2f})")
            Noiseless measurements in (...)
            >>> print(f"Noisy measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Noisy measurements in (...)

            >>> y = noise(z)
            >>> print(f"Noisy measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Noisy measurements in (...)
        """
        if torch.any(z < 0):
            raise RuntimeError("Input tensor contains negative values.")

        return self.alpha * z + torch.sqrt(self.alpha * z) * torch.randn_like(z)


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
        :attr:`alpha` (:class:`float`): The intensity of the measurements. Defaults to 10.

    Attributes:
        :attr:`alpha` (:class:`float`): Intensity of the measurements.
    """

    def __init__(self, alpha: float = 10):
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
            torch.Size([10, 4])
            >>> print(f"Noiseless measurements in ({torch.min(z):.2f} , {torch.max(z):.2f})")
            Noiseless measurements in (...)
            >>> print(f"Noisy measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Noisy measurements in (...)

            >>> y = noise(z)
            >>> print(f"Noisy measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Noisy measurements in (...)
        """
        if torch.any(z < 0):
            raise RuntimeError("Input tensor contains negative values.")
        y = torch.sqrt(self.alpha * z)
        y = y * torch.randn((*[1] * (z.ndim - 1), z.shape[-1]))
        y = y + self.alpha * z
        return y
