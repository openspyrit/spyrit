"""
Noise models for simulating measurements in imaging.

There are four classes in this module, that each simulate a different type of
noise in the measurements. The classes simulate the following types of noise:

- NoNoise: Simulates measurements with no noise

- Poisson: Simulates measurements corrupted by Poisson noise (each pixel
    receives a number of photons that follows a Poisson distribution)

- PoissonApproxGauss: Simulates measurements corrupted by Poisson noise, but
    approximates the Poisson distribution with a Gaussian distribution

- PoissonApproxGaussSameNoise: Simulates measurements corrupted by Poisson
    noise, but all measurements in a batch are corrupted with the same noise
    sample (approximated by a Gaussian distribution)
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import poisson

from spyrit.core.meas import Linear, LinearSplit, HadamSplit  # , LinearRowSplit


# =============================================================================
class NoNoise(nn.Module):
    # =========================================================================
    r"""
    Simulates measurements from images in the range [0;1] by computing
    :math:`y = \frac{1}{2} H(1+x)`.

    .. note::
        Assumes that the incoming images :math:`x` are in the range [-1;1]

    The class is constructed from a measurement operator (see the
    :mod:`~spyrit.core.meas` submodule)

    Args:
        :attr:`meas_op` : Measurement operator (see the
        :mod:`~spyrit.core.meas` submodule)

    Example 1: Using a :class:`~spyrit.core.meas.Linear` measurement operator
        >>> H = torch.rand([400,32*32])
        >>> linear_op = Linear(H)
        >>> linear_acq = NoNoise(linear_op)

    Example 2: Using a :class:`~spyrit.core.meas.HadamSplit` measurement operator
        >>> H = torch.rand([400,32*32])
        >>> Perm = torch.rand([32*32,32*32])
        >>> split_op = HadamSplit(H, Perm, 32, 32)
        >>> split_acq = NoNoise(split_op)
    """

    def __init__(self, meas_op: Union[Linear, LinearSplit, HadamSplit]):
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
        x = (x + 1) / 2
        x = self.meas_op(x)
        return x

    def sort_by_indices(self, 
                        x: torch.tensor, 
                        axis: str='rows',
                        inverse_permutation: bool=False
                        ) -> torch.tensor:
        """Reorder the rows or columns of a tensor according to the indices.
        
        The indices are stored in the attribute :attr:`self.meas_op.indices`
        and are used to reorder the rows or columns of the input tensor
        :math:`x`. The indices give the order in which the rows or columns
        should be reordered.

        ..note::
            This method is identical to the function
            :func:`~spyrit.misc.sampling.sort_by_indices`.

        Args:
            x (torch.tensor): 
                Input tensor to be reordered. The tensor must have the same
                number of rows or columns as the number of elements in the 
                attribute :attr:`self.indices`.
                
            axis (str, optional): 
                Axis along which to order the tensor. Must be either "rows" or
                "cols". Defaults to "rows".
            
            inverse_permutation (bool, optional): *
                If True, the permutation matrix is transposed before being used.
                Defaults to False.

        Raises:
            ValueError: 
                If axis is not "rows" or "cols".
            
            ValueError: 
                If the number of rows or columns in x is not equal to the length
                of the indices.

        Returns:
            torch.tensor: 
                Tensor x with reordered rows or columns according to the indices.
        """
        return self.meas_op.sort_by_indices(x, axis, inverse_permutation)


# =============================================================================
class Poisson(NoNoise):
    # =========================================================================
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
        >>> H = torch.rand([400,32*32])
        >>> linear_op = Linear(H)
        >>> linear_acq = Poisson(linear_op, 10.0)

    Example 2: Using a :class:`~spyrit.core.meas.HadamSplit` measurement operator
        >>> H = torch.rand([400,32*32])
        >>> Perm = torch.rand([32*32,32*32])
        >>> split_op = HadamSplit(H, Perm, 32, 32)
        >>> split_acq = Poisson(split_op, 200.0)

    Example 3: Using a :class:`~spyrit.core.meas.LinearSplit` measurement operator
        >>> H = torch.rand(24,64)
        >>> split_row_op = LinearSplit(H)
        >>> split_acq = Poisson(split_row_op, 50.0)
    """

    def __init__(
        self,
        meas_op: Union[Linear, LinearSplit, HadamSplit],
        alpha=50.0,
    ):
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
            >>> H = torch.rand([400,32*32])
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
            >>> Perm = torch.rand([32*32,32*32])
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

        Example 3: Two noisy measurement vectors from a :class:`~spyrit.core.meas.LinearSplit` operator
            >>> H = torch.rand(24,64)
            >>> meas_op = LinearSplit(H)
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
        # x = self.alpha*(x+1)/2
        # x = self.meas_op(x)
        x = super().forward(x)  # NoNoise forward
        x = self.alpha * x
        x = F.relu(x)  # troncate negative values to zero
        x = poisson(x)
        return x


# =============================================================================
class PoissonApproxGauss(NoNoise):
    # =========================================================================
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
        >>> H = torch.rand([400,32*32])
        >>> meas_op = Linear(H)
        >>> noise_op = PoissonApproxGauss(meas_op, 10.0)

    Example 2: Using a :class:`~spyrit.core.meas.HadamSplit` operator
        >>> Perm = torch.rand([32*32,32*32])
        >>> meas_op = HadamSplit(H, Perm, 32, 32)
        >>> noise_op = PoissonApproxGauss(meas_op, 200.0)

    Example 3: Using a :class:`~spyrit.core.meas.LinearSplit` operator
        >>> H = torch.rand(24,64)
        >>> meas_op = LinearSplit(H)
        >>> noise_op = PoissonApproxGauss(meas_op, 50.0)
    """

    def __init__(
        self,
        meas_op: Union[Linear, LinearSplit, HadamSplit],
        alpha: float,
    ):
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
            >>> H = torch.rand([400,32*32])
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
            >>> Perm = torch.rand([32*32,32*32])
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

        Example 3: Two noisy measurement vectors from a :class:`~spyrit.core.meas.LinearSplit` operator
            >>> H = torch.rand(24,64)
            >>> meas_op = LinearSplit(H)
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
        x = self.alpha * x
        x = F.relu(x)  # remove small negative values
        x = x + torch.sqrt(x) * torch.randn_like(x)
        return x


# =============================================================================
class PoissonApproxGaussSameNoise(NoNoise):
    # =========================================================================
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
        >>> H = torch.rand([400,32*32])
        >>> meas_op = Linear(H)
        >>> noise_op = PoissonApproxGaussSameNoise(meas_op, 10.0)

    Example 2: Using a :class:`~spyrit.core.meas.HadamSplit` operator
        >>> Perm = torch.rand([32*32,32*32])
        >>> meas_op = HadamSplit(H, Perm, 32, 32)
        >>> noise_op = PoissonApproxGaussSameNoise(meas_op, 200.0)
    """

    def __init__(self, meas_op: Union[Linear, LinearSplit, HadamSplit], alpha: float):
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
            >>> H = torch.rand([400,32*32])
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
            >>> Perm = torch.rand([32*32,32*32])
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
        x = self.alpha * x
        x = F.relu(x)  # remove small negative values
        x = x + torch.sqrt(x) * torch.randn(1, x.shape[1])
        return x
