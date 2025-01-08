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

from spyrit.core.meas import Linear, LinearSplit, HadamSplit  # , LinearRowSplit


# =============================================================================
class NoNoise(nn.Module):
    # =========================================================================
    r"""
    Simulates measurements not corrupted by noise.

    Assuming incoming images :math:`x` in the range [-1;1], measurements are
    first simulated from images in the range [0;1] by computing
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

    @property
    def device(self) -> torch.device:
        return self.meas_op.device

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates measurements

        Args:
            :attr:`x`: Batch of images. The input is directly passed to the
            measurement operator, so its shape depends on the type of the
            measurement operator.

        Output:
            The batch of measurements. Its shape depends on the input shape.

        Shape:
            :attr:`x`: :math:`(*, h, w)` if `self.meas_op` is a static
            measurement operator, :math:`(*, t, c, h, w)` if it is a dynamic
            measurement operator.
            :attr:`Output`: :math:`(*, M)` (static measurements) or `(*, c, M)`
            (dynamic measurements)

        Example 1: Using a :class:`~spyrit.core.meas.Linear` measurement operator
            >>> x = torch.FloatTensor(10, 3, 32, 32).uniform_(-1, 1)
            >>> linear_acq = NoNoise(linear_op)
            >>> y = linear_acq(x)
            >>> print(y.shape)
            torch.Size([10, 3, 400])

        Example 2: Using a :class:`~spyrit.core.meas.DynamicLinear` measurement operator
            >>> x = torch.FloatTensor(10, 400, 3, 32, 32).uniform_(-1, 1)
            >>> dyn_acq = DynamicLinear(torch.rand(400, 32*32))
            >>> noise_acq = NoNoise(dyn_acq)
            >>> y = split_acq(x)
            >>> print(y.shape)
            torch.Size([10, 3, 400])
        """
        x = (x + 1) / 2
        x = self.meas_op(x)
        return x

    def reindex(
        self, x: torch.tensor, axis: str = "rows", inverse_permutation: bool = False
    ) -> torch.tensor:
        """Sorts a tensor along a specified axis using the indices tensor.

        The indices tensor is contained in the attribute :attr:`self.meas_op.indices`.
        This is equivalent to calling :math:`self.meas_op.reindex` with the same
        arguments.

        The indices tensor contains the new indices of the elements in the values
        tensor. `values[0]` will be placed at the index `indices[0]`, `values[1]`
        at `indices[1]`, and so on.

        Using the inverse permutation allows to revert the permutation: in this
        case, it is the element at index `indices[0]` that will be placed at the
        index `0`, the element at index `indices[1]` that will be placed at the
        index `1`, and so on.

        .. note::
            See :func:`~spyrit.core.torch.reindex()` for more details.

        Args:
            values (torch.tensor): The tensor to sort. Can be 1D, 2D, or any
            multi-dimensional batch of 2D tensors.

            axis (str, optional): The axis to sort along. Must be either 'rows' or
            'cols'. If `values` is 1D, `axis` is not used. Default is 'rows'.

            inverse_permutation (bool, optional): Whether to apply the permutation
            inverse. Default is False.

        Raises:
            ValueError: If `axis` is not 'rows' or 'cols'.

        Returns:
            torch.tensor: The sorted tensor by the given indices along the
            specified axis.
        """
        return self.meas_op.reindex(x, axis, inverse_permutation)


# ==============================================================================
class Gaussian(nn.Module):
    r"""
    Simulates measurements corrupted by additive Gaussian noise

    .. math::

        y \sim \mathcal{N}\left(\mu = Hx, \sigma^2\right),

    where :math:`\mathcal{N}(\mu, \sigma^2)` is a Gaussian distribution with mean :math:`\mu` and
    variance :math:`\sigma^2`, :math:`H` is the measurement operator and :math:`x` is the input signal/image.

    The class is constructed from a measurement operator :math:`H` and the
    standard deviation of the noise :math:`\sigma`.

    Args:
        - :attr:`meas_op` (:mod:`~spyrit.core.meas`): Measurement operator :math:`H`

        - :attr:`sigma` (float): Standard deviation of the noise :math:`\sigma`

    Example 1: Using a :class:`~spyrit.core.meas.Linear` measurement operator
        >>> H = torch.rand([400,32*32])
        >>> linear_op = Linear(H)
        >>> linear_acq = Gaussian(linear_op, 10.0)

    """

    def __init__(
        self,
        meas_op: Union[Linear, LinearSplit, HadamSplit],
        sigma=1.0,
    ):
        super().__init__()
        self.meas_op = meas_op
        self.sigma = sigma

    def forward(self, x):
        r"""Simulates measurements corrupted by additive Gaussian noise

        .. math::

            y \sim \mathcal{N}\left(\mu = Hx, \sigma^2\right),

        where :math:`\mathcal{N}(\mu, \sigma^2)` is a Gaussian distribution
        with mean :math:`\mu` and variance :math:`\sigma^2`, :math:`H` is the
        measurement operator and :math:`x` is the input signal/image.

        Args:
            :attr:`x`: Batch of images :math:`x` with shape :math:`(*, h, w)`
            if :attr:`self.meas_op` is a static measurement operator, or
            :math:`(*, t, c, h, w)` if :attr:`self.meas_op` is a dynamic
            measurement operator.

        Output:
            Batch of measurements :math:`y` with shape :math:`(*, M)` if
            :attr:`self.meas_op` is a static measurement operator or
            :math:`(*, c, M)` if :attr:`self.meas_op` is a dynamic
            measurement operator.

        Example 1: Two different noisy measurement vectors from a :class:`~spyrit.core.meas.Linear` measurement operator
            >>> H = torch.rand([400,32*32])
            >>> meas_op = Linear(H)
            >>> noise_op = Gaussian(meas_op, 10.0)
            >>> x = torch.FloatTensor(10, 32, 32).uniform_(-1, 1)
            >>> y = noise_op(x)
            >>> print(y.shape)
            torch.Size([10, 400])
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Measurements in (-42.06 , 61.63)
            >>> y = noise_op(x)
            >>> print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            Measurements in (-43.89 , 47.60)
        """
        x = self.meas_op(x)
        x = x + self.sigma * torch.randn(x.shape)
        return x


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
            :attr:`x`: Batch of images. The input is directly passed to the
            measurement operator, so its shape depends on the type of the
            measurement operator.

        Output:
            :attr:`y` The batch of measurements. Its shape depends on the input
            shape.

        Shape:
            :attr:`x`: :math:`(*, h, w)` if `self.meas_op` is a static
            measurement operator, :math:`(*, t, c, h, w)` if it is a dynamic
            measurement operator.

            :attr:`Output`: :math:`(*, M)` (static measurements) or `(*, c, M)`
            (dynamic measurements)

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
        x = super().forward(x)  # NoNoise forward (scaling to [0, 1])
        x *= self.alpha
        x = F.relu(x)  # remove small negative values
        x = torch.poisson(x)
        return x


# =============================================================================
class PoissonApproxGauss(NoNoise):
    # =========================================================================
    r"""
    Simulates measurements corrupted by gaussian-approximated Poisson noise.

    To accelerate the computation, we consider a Gaussian approximation to the Poisson
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
            :attr:`x`: Batch of images. The input is directly passed to the
            measurement operator, so its shape depends on the type of the
            measurement operator.

        Shape:
            :attr:`x`: :math:`(*, h, w)` if `self.meas_op` is a static
            measurement operator, :math:`(*, t, c, h, w)` if it is a dynamic
            measurement operator.

            :attr:`Output`: :math:`(*, M)` (static measurements) or `(*, c, M)`
            (dynamic measurements)

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
        x = super().forward(x)  # NoNoise forward, scaling to [0, 1]
        x *= self.alpha
        x = F.relu(x)  # remove small negative values
        x = x + torch.sqrt(x) * torch.randn_like(x)
        return x


# =============================================================================
class PoissonApproxGaussSameNoise(NoNoise):
    # =========================================================================
    r"""
    Simulates measurements corrupted by identical Gaussian-approximated Poisson noise.

    To accelerate the
    computation, we consider a Gaussian approximation to the Poisson
    distribution. Contrary to :class:`~spyrit.core.noise.PoissonApproxGauss`,
    all measurements in a batch are corrupted with the same noise sample, i.e.
    the noise depends only on the measurement number (and order).

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
            :attr:`x`: Batch of images. The input is directly passed to the
            measurement operator, so its shape depends on the type of the
            measurement operator.

        Shape:
            :attr:`x`: :math:`(*, h, w)` if `self.meas_op` is a static
            measurement operator, :math:`(*, t, c, h, w)` if it is a dynamic
            measurement operator.

            :attr:`Output`: :math:`(*, M)` (static measurements) or `(*, c, M)`
            (dynamic measurements)

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
        x *= self.alpha
        x = F.relu(x)  # remove small negative values
        x = x + torch.sqrt(x) * torch.randn((*[1] * (x.ndim - 1), x.shape[-1]))
        return x
