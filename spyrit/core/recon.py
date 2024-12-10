"""
Reconstruction methods and networks.
"""

import warnings
from typing import Union

import math
import torch
import torch.nn as nn

import spyrit.core.meas as meas
from spyrit.core.noise import NoNoise
from spyrit.core.prep import DirectPoisson, SplitPoisson

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")


# =============================================================================
class PositiveParameters(nn.Module):
    def __init__(self, params, requires_grad=True):
        super(PositiveParameters, self).__init__()
        self.params = torch.tensor(params, requires_grad=requires_grad)

    def forward(self):
        return torch.abs(self.params)


# =============================================================================
class PseudoInverse(nn.Module):
    r"""Moore-Penrose pseudoinverse.

    Considering linear measurements :math:`y = Hx`, where :math:`H` is the
    measurement matrix and :math:`x` is a vectorized image, it estimates
    :math:`x` from :math:`y` by computing :math:`\hat{x} = H^\dagger y`, where
    :math:`H` is the Moore-Penrose pseudo inverse of :math:`H`.

    Example:
        >>> H = torch.rand([400,32*32])
        >>> Perm = torch.rand([32*32,32*32])
        >>> meas_op =  HadamSplit(H, Perm, 32, 32)
        >>> y = torch.rand([85,400], dtype=torch.float)
        >>> pinv_op = PseudoInverse()
        >>> x = pinv_op(y, meas_op)
        >>> print(x.shape)
        torch.Size([85, 1024])
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.tensor,
        meas_op: Union[meas.Linear, meas.DynamicLinear],
        **kwargs,
    ) -> torch.tensor:
        r"""Computes pseudo-inverse of measurements.

        Args:
            :attr:`x`: Batch of measurement vectors.

            :attr:`meas_op`: Measurement operator. Any class that
            implements a :meth:`pinv` method can be used, e.g.,
            :class:`~spyrit.core.meas.HadamSplit`.

            :attr:`kwargs`: Additional keyword arguments that are passed to
            the :meth:`pinv` method of the measurement operator. Can be used
            to specify a regularization parameter.

        Shape:

            :attr:`x`: :math:`(*, M)`

            :attr:`meas_op`: not applicable

            :attr:`output`: :math:`(*, N)`

        Example:
            >>> H = torch.rand([400,32*32])
            >>> Perm = torch.rand([32*32,32*32])
            >>> meas_op =  HadamSplit(H, Perm, 32, 32)
            >>> y = torch.rand([85,400], dtype=torch.float)
            >>> pinv_op = PseudoInverse()
            >>> x = pinv_op(y, meas_op)
            >>> print(x.shape)
            torch.Size([85, 1024])
        """
        return meas_op.pinv(x, **kwargs)


# =============================================================================
class TikhonovMeasurementPriorDiag(nn.Module):
    r"""
    Tikhonov regularisation with prior in the measurement domain.

    Considering linear measurements :math:`m = Hx \in\mathbb{R}^M`, where
    :math:`H = GF` is the measurement matrix and :math:`x\in\mathbb{R}^N` is a
    vectorized image, it estimates :math:`x` from :math:`m` by approximately
    minimizing

    .. math::
        \|m - GFx \|^2_{\Sigma^{-1}_\alpha} + \|F(x - x_0)\|^2_{\Sigma^{-1}}

    where :math:`x_0\in\mathbb{R}^N` is a mean image prior,
    :math:`\Sigma\in\mathbb{R}^{N\times N}` is a covariance prior, and
    :math:`\Sigma_\alpha\in\mathbb{R}^{M\times M}` is the measurement noise
    covariance. The matrix :math:`G\in\mathbb{R}^{M\times N}` is a
    subsampling matrix.

    .. note::
        The class is instantiated from :math:`\Sigma`, which represents the
        covariance of :math:`Fx`.

    Args:
        - :attr:`sigma`:  covariance prior with shape :math:`N` x :math:`N`
        - :attr:`M`: number of measurements :math:`M`

    Attributes:
        :attr:`comp`: The learnable completion layer initialized as
        :math:`\Sigma_1 \Sigma_{21}^{-1}`. This layer is a :class:`nn.Linear`

        :attr:`denoi`: The learnable denoising layer initialized from
        :math:`\Sigma_1`.

    Example:
        >>> sigma = torch.rand([32*32, 32*32])
        >>> recon_op = TikhonovMeasurementPriorDiag(sigma, 400)
    """

    def __init__(self, sigma: torch.tensor, M: int):
        super().__init__()

        # N = sigma.shape[0]

        var_prior = sigma.diag()[:M]

        # self.denoi = Denoise_layer(M)
        # self.denoi.weight.data = torch.sqrt(var_prior)
        # self.denoi.weight.data = self.denoi.weight.data.float()
        # self.denoi.weight.requires_grad = False

        self.denoise_weights = nn.Parameter(torch.sqrt(var_prior), requires_grad=False)

        Sigma1 = sigma[:M, :M]
        Sigma21 = sigma[M:, :M]
        # W = Sigma21 @ torch.linalg.inv(Sigma1)
        W = torch.linalg.solve(Sigma1.T, Sigma21.T).T

        self.comp = nn.Parameter(W, requires_grad=False)
        # self.comp = nn.Linear(M, N - M, False)
        # self.comp.weight.data = W
        # self.comp.weight.data = self.comp.weight.data.float()
        # self.comp.weight.requires_grad = False

    def wiener_denoise(self, x: torch.tensor, var: torch.tensor) -> torch.tensor:
        """Returns a denoised version of the input tensor using the variance prior.

        This uses the attribute self.denoise_weights, which is a learnable
        parameter.

        Inputs:
            x (torch.tensor): The input tensor to be denoised.

            var (torch.tensor): The variance prior.

        Returns:
            torch.tensor: The denoised tensor.
        """

        weights_squared = self.denoise_weights**2
        return torch.mul((weights_squared / (weights_squared + var)), x)

    def forward(
        self,
        x: torch.tensor,
        x_0: torch.tensor,
        var: torch.tensor,
        meas_op: meas.HadamSplit,
    ) -> torch.tensor:
        r"""
        Computes the Tikhonov regularization with prior in the measurement domain.

        We approximate the solution as:

        .. math::
            \hat{x} = x_0 + F^{-1} \begin{bmatrix} m_1 \\ m_2\end{bmatrix}

        with :math:`m_1 = D_1(D_1 + \Sigma_\alpha)^{-1} (m - GF x_0)` and
        :math:`m_2 = \Sigma_1 \Sigma_{21}^{-1} m_1`, where
        :math:`\Sigma = \begin{bmatrix} \Sigma_1 & \Sigma_{21}^\top \\ \Sigma_{21} & \Sigma_2\end{bmatrix}`
        and  :math:`D_1 =\textrm{Diag}(\Sigma_1)`. Assuming the noise
        covariance :math:`\Sigma_\alpha` is diagonal, the matrix inversion
        involved in the computation of :math:`m_1` is straightforward.

        This is an approximation to the exact solution

        .. math::
            \hat{x} &= x_0 + F^{-1}\begin{bmatrix}\Sigma_1 \\ \Sigma_{21} \end{bmatrix}
                      [\Sigma_1 + \Sigma_\alpha]^{-1} (m - GF x_0)

        See Lemma B.0.5 of the PhD dissertation of A. Lorente Mur (2021):
        https://theses.hal.science/tel-03670825v1/file/these.pdf

        Args:
            - :attr:`x`: A batch of measurement vectors :math:`m`
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
            >>> sigma = torch.rand([H**2, H**2])
            >>> recon_op = TikhonovMeasurementPriorDiag(sigma, M)
            >>> Ord = torch.ones((H,H))
            >> meas = HadamSplit(M, H, Ord)
            >>> y = torch.rand([B,M], dtype=torch.float)
            >>> x_0 = torch.zeros((B, H**2), dtype=torch.float)
            >>> var = torch.zeros((B, M), dtype=torch.float)
            >>> x = recon_op(y, x_0, var, meas)
            torch.Size([85, 1024])
        """
        x = x - meas_op.forward_H(x_0)
        y1 = self.wiener_denoise(x, var)
        y2 = y1 @ self.comp.T

        y = torch.cat((y1, y2), -1)
        x = x_0 + meas_op.inverse(y)
        return x


# =============================================================================
class Tikhonov(nn.Module):
    r"""Tikhonov regularization (aka as ridge regression).

    It estimates the signal :math:`x\in\mathbb{R}^{N}` the from linear
    measurements :math:`y = Ax\in\mathbb{R}^{M}` corrupted by noise by solving

    .. math::
        \| y - Ax \|^2_{\Gamma{-1}} + \|x\|^2_{\Sigma^{-1}},

    where :math:`\Gamma` is covariance of the noise, and :math:`\Sigma` is the
    signal covariance. In the case :math:`M\le N`, the solution can be computed
    as

    .. math::
        \hat{x} = \Sigma A^\top (A \Sigma A^\top + \Gamma)^{-1} y,

    where we assume that both covariance matrices are positive definite. The
    class is constructed from :math:`A` and :math:`\Sigma`, while
    :math:`\Gamma` is passed as an argument to :meth:`forward()`. Passing
    :math:`\Gamma` to :meth:`forward()` is useful in the presence of signal-
    dependent noise.

    .. note::
        * :math:`x` can be a 1d signal or a vectorized image/volume. This can
          be specified by setting the :attr:`meas_shape` attribute of the
          measurement operator.

        * The above formulation assumes that the signal :math:`x` has zero mean.


    Args:
        :attr:`meas_op` (spyrit.core.meas): Measurement operator that gives acces the
        measurement matrix :math:`A` with shape :math:`(M, N)`.

        :attr:`Sigma` (torch.tensor): Covariance prior with shape :math:`(N, N)`.

        :attr:`approx` (bool): Compute (faster) approximate
        solutions. Defaults to `False`. See :meth:`forward()` for details.

    Attributes:
        :attr:`B`: (torch.tensor): Matrix initialised as :math:`B = \Sigma A^\top`.

        :attr:`C`: (torch.tensor): Matrix initialised as
        :math:`C = A \Sigma A^\top`. This corresponds to the covariance prior
        in the transformed domain.

    Example:
        >>> B, H, M, N = 85, 17, 32, 64
        >>> sigma = torch.rand(N, N)
        >>> gamma = torch.rand(M, M)
        >>> A = torch.rand([M,N])
        >>> meas  = Linear(A, meas_shape=(1,N))
        >>> recon = Tikhonov(meas, sigma)
        >>> y = torch.rand(B,H,M)
        >>> x = recon(y, gamma)
        >>> print(y.shape)
        >>> print(x.shape)
        torch.Size([85, 17, 32])
        torch.Size([85, 17, 64])
    """

    def __init__(self, meas_op, sigma: torch.tensor, approx=False):
        """
        meas_op is a measurement operator
        sigma is the image covariance prior
        """
        super().__init__()

        dtype = sigma.dtype
        A = meas_op.H  # .to(sigma.dtype)
        sigma = sigma.to(A.dtype)

        if isinstance(meas_op, meas.DynamicLinearSplit):
            # the measurement covariance prior is assumed to be for
            # Hadamard-matrix measurements (i.e. no splitting)
            A = A[::2, :] - A[1::2, :]

        if approx:
            # if we use the diagonal approximation, then we assume that the
            # A @ sigma @ A.T is diagonal
            sigma_meas = torch.diag(A @ sigma @ A.T).to(dtype)
        else:
            sigma_meas = (A @ sigma @ A.T).to(dtype)

        # estimation of the missing measurements
        sigma_A_T = torch.mm(sigma, A.mT).to(dtype)

        self.register_buffer("sigma_meas", sigma_meas)
        self.register_buffer("sigma_A_T", sigma_A_T)
        self.meas_op = meas_op
        self.img_shape = meas_op.img_shape
        self.approx = approx

    def divide(self, y: torch.tensor, gamma: torch.tensor) -> torch.tensor:

        if self.approx:
            return y / (self.sigma_meas + torch.diagonal(gamma, dim1=-2, dim2=-1))
        else:
            # we need to expand the matrices for the solve/ matmul
            batch_shape = y.shape[:-1]
            expand_shape = batch_shape + (self.sigma_meas.shape)
            y = y.unsqueeze(-1)  # add a dimension to y for batch matrix multiplications
            y = torch.linalg.solve((self.sigma_meas + gamma).expand(expand_shape), y)
            return y.squeeze(-1)

    def forward(
        self, y: torch.tensor, gamma: torch.tensor  # x_0: torch.tensor,
    ) -> torch.tensor:
        r"""
        The Tikhonov solution is computed as

        .. math::
            \hat{x} = B^\top (C + \Gamma)^{-1} y

        with :math:`B = \Sigma A^\top` and :math:`C = A \Sigma A^\top`. When
        :attr:`self.approx` is True, it is approximated as

        .. math::
            \hat{x} = B^\top  \frac{y}{\text{diag}(C + \Gamma)}

        Args:
            :attr:`y` (torch.tensor):  A batch of measurement vectors :math:`y`

            :attr:`gamma` (torch.tensor): A batch of noise covariance :math:`\Gamma`

        Shape:
            :attr:`y` (torch.tensor): :math:`(*, M)`

            :attr:`gamma` (torch.tensor): :math:`(*, M, M)`

            Output (torch.tensor): :math:`(*, N)`
        """
        y = self.divide(y, gamma)
        y = torch.matmul(self.sigma_A_T, y.unsqueeze(-1)).squeeze(-1)
        # y = y.reshape(*y.shape[:-1], *self.img_shape)

        return y


# =============================================================================
class Denoise_layer(nn.Module):
    r"""Defines a learnable Wiener filter that assumes additive white Gaussian noise.

    The filter is pre-defined upon initialization with the standard deviation prior
    (if known), or with an integer representing the size of the input vector.
    In the second case, the standard deviation prior is initialized at random
    from a uniform (0,2/size) distribution.

    Using the foward method (the implicit call method), the filter is fully
    defined:

    .. math::
        \sigma_\text{prior}^2/(\sigma^2_\text{prior} + \sigma^2_\text{meas})

    where :math:`\sigma^2_\text{prior}` is the variance prior defined at
    initialization and :math:`\sigma^2_\text{meas}` is the measurement variance
    defined using the forward method. The value given by the equation above
    can then be multiplied by the measurement vector to obtain the denoised
    measurement vector.

    .. note::
        The weight (defined at initialization or accessible through the
        attribute :attr:`weight`) should not be squared (as it is squared when
        the forward method is called).

    Args:
        :attr:`std_dev_or_size` (torch.tensor or int): 1D tensor representing
        the standard deviation prior or an integer defining the size of the
        randomly-initialized standard deviation prior. If an array is passed
        and it is not 1D, it is flattened. It is stored internally as a
        :class:`nn.Parameter`, whose :attr:`data` attribute is accessed through
        the :attr:`sigma` attribute, and whose :attr:`requires_grad` attribute
        is accessed through the :attr:`requires_grad` attribute.

    Shape for forward call:
        - Input: :math:`(*, in\_features)` measurement variance.
        - Output: :math:`(*, in\_features)` fully defined Wiener filter.

    Attributes:
        :attr:`weight`:
        The learnable standard deviation prior :math:`\sigma_\text{prior}` of
        shape :math:`(in\_features, 1)`. The values are initialized from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k = 1/in\_features`.

        :attr:`sigma`:
        The learnable standard deviation prior :math:`\sigma_\text{prior}` of shape
        :math:`(, in\_features)`. If the input is an integer, the standard deviation prior
        is initialized at random from  :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`,
        where :math:`k = 1/in\_features`.

        :attr:`in_features`:
        The number of input features.

        :attr:`requires_grad`:
        A boolean indicating whether the autograd should record operations on
        the standard deviation tensor. Default is True.

    Example:
        >>> m = Denoise_layer(30)
        >>> input = torch.randn(128, 30)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(
        self, std_dev_prior_or_size: Union[torch.tensor, int], requires_grad=True
    ):
        super(Denoise_layer, self).__init__()

        warnings.warn(
            "This class is deprecated and will be removed in a future release. "
            "Please use the `TikhonovMeasurementPriorDiag` class instead.",
            DeprecationWarning,
        )

        if isinstance(std_dev_prior_or_size, int):
            self.weight = nn.Parameter(
                torch.Tensor(std_dev_prior_or_size), requires_grad=requires_grad
            )
            self.reset_parameters()

        else:
            if not isinstance(std_dev_prior_or_size, torch.Tensor):
                raise TypeError(
                    "std_dev_or_size should be an integer or a torch.Tensor"
                )
            self.weight = nn.Parameter(
                std_dev_prior_or_size.reshape(-1), requires_grad=requires_grad
            )

    @property
    def in_features(self):
        return self.weight.data.numel()

    def reset_parameters(self):
        r"""
        Resets the standard deviation prior :math:`\sigma_\text{prior}`.

        The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`,
        where :math:`k = 1/in\_features`. They are stored in the :attr:`weight`
        attribute.
        """
        nn.init.uniform_(self.weight, 0, 2 / math.sqrt(self.in_features))

    def forward(self, sigma_meas_squared: torch.tensor) -> torch.tensor:
        r"""
        Fully defines the Wiener filter with the measurement variance.

        This outputs :math:`\sigma_\text{prior}^2/(\sigma_\text{prior}^2 + \sigma^2_\text{meas})`,
        where :math:`\sigma^2_\text{meas}` is the measurement variance (see :attr:`sigma_meas_squared`) and
        :math:`\sigma_\text{prior}` is the standard deviation prior defined
        upon construction of the class (see :attr:`self.weight`).

        .. note::
            The measurement variance should be squared before being passed to
            this method, unlike the standard deviation prior (defined at construction).

        Args:
            :attr:`sigma_meas_squared` (torch.tensor): input tensor :math:`\sigma^2_\text{meas}`
            of shape :math:`(*, in\_features)`

        Returns:
            torch.tensor: The multiplicative filter of shape
            :math:`(*, in\_features)`

        Shape:
            - Input: :math:`(*, in\_features)`
            - Output: :math:`(*, in\_features)`
        """
        if sigma_meas_squared.shape[-1] != self.in_features:
            raise ValueError(
                "The last dimension of the input tensor "
                + f"({sigma_meas_squared.shape[-1]})should be equal to the number of "
                + f"input features ({self.in_features})."
            )
        return self.tikho(sigma_meas_squared, self.weight)

    def extra_repr(self):
        return "in_features={}".format(self.in_features)

    @staticmethod
    def tikho(inputs: torch.tensor, weight: torch.tensor) -> torch.tensor:
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Applies a transformation to the incoming data: :math:`y = \sigma_\text{prior}^2/(\sigma_\text{prior}^2+x)`.

        :math:`x` is the input tensor (see :attr:`inputs`) and :math:`\sigma_\text{prior}` is the
        standard deviation prior (see :attr:`weight`).

        Args:
            :attr:`inputs` (torch.tensor): input tensor :math:`x` of shape
            :math:`(N, *, in\_features)`

            :attr:`weight` (torch.tensor): standard deviation prior :math:`\sigma_\text{prior}` of
            shape :math:`(in\_features)`

        Returns:
            torch.tensor: The transformed data :math:`y` of shape
            :math:`(N, in\_features)`

        Shape:
            - :attr:`inputs`: :math:`(N, *, in\_features)` where `*` means any number of
              additional dimensions - Variance of measurements
            - :attr:`weight`: :math:`(in\_features)` - corresponds to the standard deviation
              of our prior.
            - :attr:`output`: :math:`(N, in\_features)`
        """
        a = weight**2  # prefer to square it, because when learnt, it can go to the
        # negative, which we do not want to happen.
        # TO BE Potentially done : square inputs.
        b = a + inputs
        return a / b


# -----------------------------------------------------------------------------
# |                      RECONSTRUCTION NETWORKS                              |
# -----------------------------------------------------------------------------


# =============================================================================
class PinvNet(nn.Module):
    r"""Pseudo inverse reconstruction network

    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`)

        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)

        :attr:`denoi` (optional): Image denoising operator
        (see :class:`~spyrit.core.nnet`).
        Default :class:`~spyrit.core.nnet.Identity`

    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(B,C,H,W)`
        corresponding to the batch size, number of channels, height, and width.

        :attr:`output`: Reconstructed images with shape :math:`(B,C,H,W)`
        corresponding to the batch size, number of channels, height, and width.

    Attributes:
        :attr:`Acq`: Acquisition operator initialized as :attr:`noise`

        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`

        :attr:`pinv`: Analytical reconstruction operator initialized as
        :class:`~spyrit.core.recon.PseudoInverse()`

        :attr:`Denoi`: Image denoising operator initialized as :attr:`denoi`


    Example:
        >>> B, C, H, M = 10, 1, 64, 64**2
        >>> Ord = torch.ones((H,H))
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

    @property
    def device(self):
        return self.acqu.device

    def forward(self, x):
        r"""Full pipeline of reconstrcution network

        Args:
            :attr:`x`: ground-truth images

        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`

            :attr:`output`: reconstructed images with shape :math:`(B,C,H,W)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
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
        x = self.acquire(x)
        x = self.reconstruct(x)
        return x

    def acquire(self, x):
        r"""Simulates data acquisition

        Args:
            :attr:`x`: ground-truth images

        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`

            :attr:`output`: measurement vectors with shape :math:`(BC,2M)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet.acquire(x)
            >>> print(z.shape)
            torch.Size([10, 8192])
        """
        # b, c, _, _ = x.shape
        # Acquisition
        # x = x.reshape(b * c, self.acqu.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        return self.acqu(x)  # shape x = [b*c, 2*M]

    def meas2img(self, y):
        """Returns images from raw measurement vectors

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: :math:`(*,2M)`

            :attr:`output`: :math:`(*,H,W)`

        Example:
            >>> B, C, H, M = 10, 3, 64, 64**2
            >>> Ord = torch.ones(H,H)
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H**2)
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.rand((B,C,2*M), dtype=torch.float32)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
            torch.Size([10, 3, 64, 64])
        """
        m = self.prep(y)
        m = torch.nn.functional.pad(m, (0, self.acqu.meas_op.N - self.acqu.meas_op.M))

        # reindex the measurements
        z = self.acqu.meas_op.reindex(m, "cols", False)
        return z.reshape(*z.shape[:-1], self.acqu.meas_op.h, self.acqu.meas_op.w)

    def reconstruct(self, x):
        r"""Preprocesses, reconstructs, and denoises raw measurement vectors.

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: :math:`(BC,2M)`

            :attr:`output`: :math:`(BC,1,H,W)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H**2)
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        return self.denoi(self.reconstruct_pinv(x))

    def reconstruct_pinv(self, x):
        r"""Preprocesses and reconstructs raw measurement vectors.

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: :math:`(BC,2M)`

            :attr:`output`: :math:`(BC,1,H,W)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H**2)
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct_pinv(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        x = self.prep(x)
        x = self.pinv(x, self.acqu.meas_op)
        return x

    def reconstruct_expe(self, x):
        r"""Reconstruction step of a reconstruction network

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
        x, N0_est = self.prep.forward_expe(x, self.acqu.meas_op)  # shape x = [b*c, M]
        # print(N0_est)

        # measurements to image domain processing
        x = self.pinv(x, self.acqu.meas_op)  # shape x = [b*c,N]

        # Image domain denoising
        x = x.reshape(
            bc, 1, self.acqu.meas_op.h, self.acqu.meas_op.w
        )  # shape x = [b*c,1,h,w]
        x = self.denoi(x)  # shape x = [b*c,1,h,w]
        # print(x.max())

        # Denormalization
        x = self.prep.denormalize_expe(
            x, N0_est, self.acqu.meas_op.h, self.acqu.meas_op.w
        )
        # return x
        return x, N0_est


# =============================================================================
class Pinv1Net(nn.Module):
    # =============================================================================
    r"""1D pseudo inverse reconstruction network.

    Considering linear measurements :math:`Y = HX`, where
    :math:`H\in\mathbb{R}^{k\times h}` is the
    measurement matrix and :math:`X \in\mathbb{R}^{h\times w}` is an image, it estimates
    :math:`X` from :math:`Y` by computing

    .. math:: \hat{X} = \mathcal{G}_\theta(H^\dagger Y),

    where :math:`H` is the Moore-Penrose pseudo inverse of :math:`H`, and
    :math:`\mathcal{G}_\theta` is a neural network.

    The pseudo-inverse is computed along the last dimension, while (learnable)
    denoising applies to the last two dimensions.


    Args:
        :attr:`noise`: Acquisition operator that compute (noisy) measurements :math:`Y = HX` (see :class:`~spyrit.core.noise`)

        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)

        :attr:`denoi` (optional): Image denoising operator
        :math:`\mathcal{G}_\theta` (see :class:`~spyrit.core.nnet`). Defaults
        to :class:`~spyrit.core.nnet.Identity`.

    Input / Output:
        :attr:`input`: Ground-truth images :math:`X` with shape :math:`(b,c,h,w)`.

        :attr:`output`: Reconstructed images :math:`\hat{X}` with shape :math:`(b,c,h,w)`.

    Attributes:
        :attr:`acqu`: Acquisition operator initialized as :attr:`noise`.

        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`.

        :attr:`pinv`: Analytical reconstruction operator initialized as
        :class:`~spyrit.core.recon.PseudoInverse()`.

        :attr:`denoi`: Image denoising operator initialized as :attr:`denoi`.

    Example:
        >>> b,c,h,w = 10,1,48,64
        >>> H = torch.rand(15,w)
        >>> meas = Linear(H, meas_shape=(1,w))
        >>> noise = NoNoise(meas)
        >>> prep = DirectPoisson(1.0, meas)
        >>> recnet = Pinv1Net(noise, prep)
        >>> x = torch.FloatTensor(b,c,h,n).uniform_(-1, 1)
        >>> z = recnet(x)
        >>> print(z.shape)
        >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
        torch.Size([10, 1, 64, 64])
        tensor(5.8912e-06)


    .. note::
        The measurement operator applies to the last dimension of the input
        tensor, contrary :class:`~spyrit.core.recon.PinvNet` where it applies
        to the last two dimensions. In both cases, the denoising operator
        applies to the last two dimensions.

    """

    def __init__(self, noise, prep, denoi=nn.Identity()):
        super().__init__()
        self.acqu = noise
        self.prep = prep
        self.pinv = PseudoInverse()
        self.denoi = denoi

    def forward(self, x):
        r"""Full pipeline (image-to-image mapping)

        Args:
            :attr:`x` (torch.tensor): Ground-truth images with shape :math:`(*,h,w)`, where
            :math:`*` is any batch size.

        Output:
            torch.tensor: Reconstructed images with shape :math:`(*,h,w)`, where
            :math:`*` is any batch size.

        Example:
            >>> b,c,h,w = 10,1,48,64
            >>> H = torch.rand(15,w)
            >>> meas = Linear(H, meas_shape=(1,w))
            >>> noise = NoNoise(meas)
            >>> prep = DirectPoisson(1.0, meas)
            >>> recnet = Pinv1Net(noise, prep)
            >>> x = torch.FloatTensor(b,c,h,n).uniform_(-1, 1)
            >>> z = recnet(x)
            >>> print(z.shape)
            >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
            torch.Size([10, 1, 64, 64])
            tensor(5.8912e-06)
        """

        # Acquisition
        x = self.acqu(x)

        # Reconstruction
        x = self.reconstruct(x)

        return x

    def reconstruct(self, x):
        r"""Reconstruction (measurement-to-image mapping)

        Args:
            :attr:`x` (torch.tensor): Raw measurement vectors with shape :math:`(*,h,k)`.

        Output:
            torch.tensor: Reconstructed images with shape :math:`(*,h,w)`
        """
        # Preprocessing in the measurement domain
        x = self.prep(x)

        # measurements to image-domain processing
        x = self.pinv(x, self.acqu.meas_op)
        x = x.squeeze(-2)  # shape x = [*,1,N] -> x = [*,N]

        # Image-domain denoising
        x = self.denoi(x)

        return x

    def reconstruct_expe(self, x):
        r"""Reconstruction (measurement-to-image mapping) for experimental data.

        Args:
            :attr:`x`: Raw measurement vectors with shape :math:`(*,h,k)`.

        Output:
            Reconstructed images with shape :math:`(*,h,w)`
        """

        # Preprocessing
        # x, norm = self.prep.forward_expe(x, self.acqu.meas_op, (-2,-1)) # shape: [*, M]

        # Alternative where the mean is computed on each row
        x, norm = self.prep.forward_expe(x, self.acqu.meas_op)  # shape: [*, M]

        # measurements to image domain processing
        x = self.pinv(x, self.acqu.meas_op)  # shape: [*,N]
        x = x.squeeze(-2)  # shape: [*,1,N] -> [*,N]

        # Image-domain denoising
        x = self.denoi(x)  # shape: [*,h,w]

        # Denormalization
        x = self.prep.denormalize_expe(x, norm, x.shape[-2], x.shape[-1])
        return x, norm


# =============================================================================
class DCNet(nn.Module):
    r"""Denoised completion reconstruction network.

    This is a four step reconstruction method:

    #. Denoising in the measurement domain.
    #. Estimation of the missing measurements from the denoised ones.
    #. Image-domain mapping.
    #. (Learned) Denoising in the image domain.

    The first three steps corresponds to Tikhonov regularisation. Typically, only the last step involves learnable parameters.

    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`)

        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)

        :attr:`sigma`: Covariance prior (for details, see the
        :class:`~spyrit.core.recon.TikhonovMeasurementPriorDiag()` class)

        :attr:`denoi` (optional): Image denoising operator
        (see :class:`~spyrit.core.nnet`).
        Default :class:`~spyrit.core.nnet.Identity`

    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(*,H,W)`, with
        :math:`*` being any batch size.

        :attr:`output`: Reconstructed images with shape :math:`(*,H,W)`, with
        :math:`*` being any batch size.

    Attributes:
        :attr:`Acq`: Acquisition operator initialized as :attr:`noise`

        :attr:`PreP`: Preprocessing operator initialized as :attr:`prep`

        :attr:`DC_Layer`: Data consistency layer initialized as :attr:`tikho`

        :attr:`Denoi`: Image denoising operator initialized as :attr:`denoi`


    Example:
        >>> B, C, H, M = 10, 1, 64, 64**2
        >>> Ord = torch.ones((H,H))
        >>> meas = HadamSplit(M, H, Ord)
        >>> noise = NoNoise(meas)
        >>> prep = SplitPoisson(1.0, M, H*H)
        >>> sigma = torch.rand([H**2, H**2])
        >>> recnet = DCNet(noise,prep,sigma)
        >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
        >>> z = recnet(x)
        >>> print(z.shape)
        torch.Size([10, 1, 64, 64])
    """

    def __init__(
        self,
        noise: NoNoise,
        prep: Union[DirectPoisson, SplitPoisson],
        sigma: torch.tensor,
        denoi=nn.Identity(),
    ):
        super().__init__()
        self.Acq = noise
        self.prep = prep
        self.denoi = denoi
        sigma = sigma.to(torch.float32)

        sigma = noise.reindex(sigma, "rows", False)
        sigma = noise.reindex(sigma, "cols", True)
        sigma_perm = sigma

        # save in tikho
        self.tikho = TikhonovMeasurementPriorDiag(sigma_perm, noise.meas_op.M)

    @property
    def device(self):
        return self.Acq.device

    def forward(self, x):
        r"""Full pipeline of the reconstruction network

        Args:
            :attr:`x`: ground-truth images

        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`

            :attr:`output`: reconstructed images with shape :math:`(B,C,H,W)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> sigma = torch.rand([H**2, H**2])
            >>> recnet = DCNet(noise,prep,sigma)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        x = self.acquire(x)
        x = self.reconstruct(x)
        return x

    def acquire(self, x):
        r"""Simulate data acquisition

        Args:
            :attr:`x`: ground-truth images

        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`

            :attr:`output`: measurement vectors with shape :math:`(BC,2M)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> sigma = torch.rand([H**2, H**2])
            >>> recnet = DCNet(noise,prep,sigma)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet.acquire(x)
            >>> print(z.shape)
            torch.Size([10, 8192])
        """
        return self.Acq(x)

    def reconstruct(self, x):
        r"""Reconstruction step of a reconstruction network

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: raw measurement vectors with shape :math:`(BC,2M)`

            :attr:`output`: reconstructed images with shape :math:`(BC,1,H,W)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> sigma = torch.rand([H**2, H**2])
            >>> recnet = DCNet(noise,prep,sigma)
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        var_noi = self.prep.sigma(x)
        x = self.prep(x)

        # x.shape = (*, M), make x_0 (*, h, w)
        x_0 = torch.zeros(
            (*x.shape[:-1], *self.Acq.meas_op.meas_shape), device=x.device
        )
        x = self.tikho(x, x_0, var_noi, self.Acq.meas_op)

        # Image domain denoising
        return self.denoi(x)

    def reconstruct_expe(self, x):
        r"""Reconstruction step of a reconstruction network

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
        x, N0_est = self.prep.forward_expe(x, self.Acq.meas_op)  # x <- x/N0_est
        x = x / self.prep.gain
        norm = self.prep.gain * N0_est

        # variance of preprocessed measurements
        var_noi = torch.div(
            var_noi, (norm.reshape(-1, 1).expand(bc, self.Acq.meas_op.M)) ** 2
        )

        # measurements to image domain processing
        x_0 = torch.zeros((bc, self.Acq.meas_op.N), device=x.device)
        x = self.tikho(x, x_0, var_noi, self.Acq.meas_op)
        x = x.reshape(
            bc, 1, self.Acq.meas_op.h, self.Acq.meas_op.w
        )  # shape x = [b*c,1,h,w]

        # Image domain denoising
        x = self.denoi(x)  # shape x = [b*c,1,h,w]

        # Denormalization
        x = self.prep.denormalize_expe(x, norm, self.Acq.meas_op.h, self.Acq.meas_op.w)

        return x


# =============================================================================
class TikhoNet(nn.Module):
    r"""Tikhonov reconstruction network.

    This is a two-step reconstruction method. Typically, only the last step involves learnable parameters.

    #. Tikhonov regularisation.
    #. (Learned) Denoising in the image domain.


    Args:
        :attr:`noise` (spyrit.core.noise): Acquisition operator (see :mod:`~spyrit.core.noise`)

        :attr:`prep` (spyrit.core.prep): Preprocessing operator (see :mod:`~spyrit.core.prep`)

        :attr:`sigma` (torch.tensor): Image-domain covariance prior (for details, see the
        :class:`~spyrit.core.recon.Tikhonov()` class)

        :attr:`denoi` (torch.nn.Module, optional): Image denoising operator
        (see :class:`~spyrit.core.nnet`).
        Default :class:`~spyrit.core.nnet.Identity`

    Input / Output:
        :attr:`input` (torch.tensor): Ground-truth images with shape :math:`(*,H,W)`, with
        :math:`*` being any batch size.

        :attr:`output` (torch.tensor): Reconstructed images with shape :math:`(*,H,W)`, with
        :math:`*` being any batch size.

    Attributes:
        :attr:`acqu`: Acquisition operator initialized as :attr:`noise`

        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`

        :attr:`tikho`: Data consistency layer initialized as :attr:`Tikhonov(noise.meas_op, sigma)`

        :attr:`denoi`: Image denoising operator initialized as :attr:`denoi`


    Example:
        >>> B, H, M, N = 85, 17, 32, 64
        >>> sigma = torch.rand(N, N)
        >>> gamma = torch.rand(M, M)
        >>> A = torch.rand([M,N])
        >>> meas = Linear(A, meas_shape=(1,N))
        >>> noise = NoNoise(meas)
        >>> prep = DirectPoisson(1, meas)
        >>> recon = TikhoNet(noise, prep, sigma)
        >>> y = torch.rand(B,H,M)
        >>> x = recon(y, gamma)
        >>> print(y.shape)
        >>> print(x.shape)
        torch.Size([85, 17, 32])
        torch.Size([85, 17, 1, 64])
    """

    def __init__(self, noise, prep, sigma: torch.tensor, denoi=nn.Identity()):

        super().__init__()
        self.acqu = noise
        self.prep = prep
        # device = noise.meas_op.H.device        # spyrit 2.3.
        # sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device)
        self.tikho = Tikhonov(noise.meas_op, sigma)
        self.denoi = denoi

    def forward(self, x):
        """Full pipeline (image-to-image mapping)

        Args:
            x (torch.tensor): Ground-truth images with shape :math:`(*,H,W)`, with
            :math:`*` being any batch size.

        Returns:
            torch.tensor: Reconstruction images with shape :math:`(*,H,W)`, with
            :math:`*` being any batch size.
        """
        # Acquisition
        x = self.acqu(x)  # shape x = [b*c, 2*M]
        # Reconstruction
        x = self.reconstruct(x)  # shape x = [bc, 1, h,w]

        return x

    def reconstruct(self, x):
        r"""Reconstruction (measurement-to-image mapping)

        Args:
            :attr:`x` (torch.tensor): Raw measurement vectors with shape :math:`(*,M)`.

        Output:
            torch.tensor: Reconstructed images with shape :math:`(*,H,W)`
        """
        # Preprocessing
        cov_meas = self.prep.sigma(x)
        x = self.prep(x)

        # covariance of measurements
        cov_meas = torch.diag_embed(cov_meas)  #

        # print(cov_meas)

        # measurements to image domain processing
        x = self.tikho(x, cov_meas)
        x = x.reshape(*x.shape[:-1], self.acqu.meas_op.h, self.acqu.meas_op.w)

        # Image domain denoising
        x = self.denoi(x)

        return x

    def reconstruct_expe(self, x):
        r"""Reconstruction (measurement-to-image mapping) for experimental data.

        Args:
            :attr:`x` (torch.tensor): Raw measurement vectors with shape :math:`(*,M)`.

        Output:
            torch.tensor: Reconstructed images with shape :math:`(*,H,W)`
        """
        # Preprocessing
        cov_meas = self.prep.sigma_expe(x)
        # print(cov_meas)
        # print(self.prep.nbin, self.prep.mudark)
        # x, norm = self.prep.forward_expe(x, self.acqu.meas_op, (-2,-1)) # shape: [*, M]

        # Alternative where the mean is computed on each row
        x, norm = self.prep.forward_expe(x, self.acqu.meas_op)  # shape: [*, M]

        # covariance of measurements
        cov_meas = cov_meas / norm**2
        cov_meas = torch.diag_embed(cov_meas)

        # measurements to image domain processing
        x = self.tikho(x, cov_meas)
        x = x.reshape(*x.shape[:-1], self.acqu.meas_op.h, self.acqu.meas_op.w)

        # Image domain denoising
        x = self.denoi(x)

        # Denormalization
        x = self.prep.denormalize_expe(x, norm, x.shape[-2], x.shape[-1])

        return x, norm


# =============================================================================
class LearnedPGD(nn.Module):
    r"""Learned Proximal Gradient Descent reconstruction network.
    Iterative algorithm that alternates between a gradient step and a proximal step,
    where the proximal operator is learned denoiser. The update rule is given by:

    :math:`x_{k+1} = prox(\hat{x_k} - step * H^T (Hx_k - y))=
    denoi(\hat{x_k} - step * H^T (Hx_k - y))`

    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`)

        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)

        :attr:`denoi` (optional): Image denoising operator
        (see :class:`~spyrit.core.nnet`).
        Default :class:`~spyrit.core.nnet.Identity`

        :attr:`iter_stop` (int): Number of iterations of the LPGD algorithm
        (commonly 3 to 10, trade-off between accuracy and speed).
        Default 3 (for speed and with higher accuracy than post-processing denoising)

        :attr:`step` (float): Step size of the LPGD algorithm. Default is None,
        and it is estimated as the inverse of the Lipschitz constant of the gradient of the
        data fidelity term.
            - If :math:`meas_op.N` is available, the step size is estimated as
            :math:`step=1/L=1/\text{meas_op.N}`, true for Hadamard operators.
            - If not, the step size is estimated from by computing
            the Lipschitz constant as the largest singular value of the
            Hessians, :math:`L=\lambda_{\max}(H^TH)`. If this fails,
            the step size is set to 1e-4.

        :attr:`step_estimation` (bool): Default False. See :attr:`step` for details.

        :attr:`step_grad` (bool): Default False. If True, the step size is learned
        as a parameter of the network. Not tested yet.

        :attr:`wls` (bool): Default False. If True, the data fidelity term is
        modified to be the weighted least squares (WLS) term, which approximates
        the Poisson likelihood. In this case, the data fidelity term is
        :math:`\|Hx-y\|^2_{C^{-1}}`, where :math:`C` is the covariance matrix.
        We assume that :math:`C` is diagonal, and the diagonal elements are
        the measurement noise variances, estimated from :class:`~spyrit.core.prep.sigma`.

        :attr:`gt` (torch.tensor): Ground-truth images. If available, the mean
        squared error (MSE) is computed and logged. Default None.

        :attr:`log_fidelity` (bool): Default False. If True, the data fidelity term
        is logged for each iteration of the LPGD algorithm.

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
        >>> Ord = torch.ones((H,H))
        >>> meas = HadamSplit(M, H, Ord)
        >>> noise = NoNoise(meas)
        >>> prep = SplitPoisson(1.0, M, H*H)
        >>> recnet = LearnedPGD(noise, prep)
        >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
        >>> z = recnet(x)
        >>> print(z.shape)
        torch.Size([10, 1, 64, 64])
        >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
        tensor(5.8912e-06)
    """

    def __init__(
        self,
        noise,
        prep,
        denoi=nn.Identity(),
        iter_stop=3,
        x0=0,
        step=None,
        step_estimation=False,
        step_grad=False,
        step_decay=1,
        wls=False,
        gt=None,
        log_fidelity=False,
        res_learn=False,
    ):
        super().__init__()
        # nn.module
        self.acqu = noise
        self.prep = prep
        self.denoi = denoi

        self.pinv = PseudoInverse()

        # LPGD algo
        self.x0 = x0
        self.iter_stop = iter_stop
        self.step = step
        self.step_estimation = step_estimation
        self.step_grad = step_grad
        self.step_decay = step_decay
        self.res_learn = res_learn

        # Init step size (estimate)
        self.set_stepsize(step)

        # WLS
        self.wls = wls

        # Log fidelity
        self.log_fidelity = log_fidelity

        # Log MSE (Ground truth available)
        if gt is not None:
            self.x_gt = nn.Parameter(
                torch.tensor(gt.reshape(gt.shape[0], -1)), requires_grad=False
            )
        else:
            self.x_gt = None

    def step_schedule(self, step):
        if self.step_decay != 1:
            step = [step * self.step_decay**i for i in range(self.iter_stop)]
        elif self.iter_stop > 1:
            step = [step for i in range(self.iter_stop)]
        else:
            step = [step]
        return step

    def set_stepsize(self, step):
        if step is None:
            # Stimate stepsize from Lipschitz constant
            if hasattr(self.acqu.meas_op, "N"):
                step = 1 / self.acqu.meas_op.N
            else:
                # Estimate step size as 1/sv_max(H^TH); if failed, set to 1e-4
                self.step_estimation = True
                step = 1e-4

        step = self.step_schedule(step)
        # step = nn.Parameter(torch.tensor(step), requires_grad=self.step_grad)
        step = PositiveParameters(step, requires_grad=self.step_grad)
        self.step = step

    def forward(self, x):
        r"""Full pipeline of reconstruction network

        Args:
            :attr:`x`: ground-truth images

        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`

            :attr:`output`: reconstructed images with shape :math:`(B,C,H,W)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> recnet = LearnedPGD(noise, prep)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
            >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
            tensor(5.8912e-06)
        """
        x = self.acquire(x)
        x = self.reconstruct(x)
        return x

    def acquire(self, x):
        r"""Simulate data acquisition

        Args:
            :attr:`x`: ground-truth images

        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`

            :attr:`output`: measurement vectors with shape :math:`(BC,2M)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet.acquire(x)
            >>> print(z.shape)
            torch.Size([10, 8192])
        """
        return self.acqu(x)

    def hessian_sv(self):
        H = self.acqu.meas_op.H
        if self.wls:
            std_mat = 1 / torch.sqrt(self.meas_variance)
            std_mat = torch.diag(std_mat.reshape(-1))
            H = torch.matmul(std_mat, H)
        try:
            s = torch.linalg.svdvals(torch.matmul(H.t(), H))
        except:
            print("svdvals(H^T*H) failed, trying svdvals(H) instead")
            s = torch.linalg.svdvals(H) ** 2
        return s

    def stepsize_gd(self):
        s = self.hessian_sv()
        self.step = 2 / (s.min() + s.max())  # Kressner, EPFL, GD #1/(2*s.max()**2)

    def cost_fun(self, x, y):
        proj = self.acqu.meas_op.forward_H(x)
        res = proj - y
        if self.wls:
            res = res / torch.sqrt(self.meas_variance)
        return torch.linalg.norm(res) ** 2

    def mse_fun(self, x, x_gt):
        return torch.linalg.norm(x - x_gt)

    def reconstruct(self, x):
        r"""Reconstruction step of a reconstruction network

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: :math:`(BC,2M)`

            :attr:`output`: :math:`(BC,1,H,W)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H**2)
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """

        # Compute the stepsize from the Lipschitz constant
        if self.step_estimation:
            self.stepsize_gd()

        step = self.step
        if not isinstance(step, torch.Tensor):
            step = step.params

        # Preprocessing in the measurement domain
        m = self.prep(x)

        if self.wls:
            # Get variance of the measurements
            if hasattr(self.prep, "sigma"):
                meas_variance = self.prep.sigma(x)
                self.meas_variance = meas_variance
            else:
                print(
                    "WLS requires the variance of the measurements to be known!. Estimating var==m"
                )
                meas_variance = m

            # Normalize the stepsize to account for the variance
            meas_variance_img_min, _ = torch.min(meas_variance, 1)  # 128
            step = step.reshape(self.iter_stop, 1).to(x.device)
            # Multiply meas_variance_img_min and step
            step = meas_variance_img_min * step

        # If pinv method is defined
        if self.x0 != 0:
            if hasattr(self.acqu.meas_op, "pinv"):
                x = self.acqu.meas_op.pinv(m)

                # proximal step (prior)
                if isinstance(self.denoi, nn.ModuleList):
                    x = self.denoi[0](x)
                else:
                    x = self.denoi(x)

            if self.res_learn:
                z0 = x.detach().clone()
        else:
            # zero init
            x = torch.zeros(
                (*x.shape[:-1], *self.acqu.meas_op.meas_shape), device=x.device
            )

        print("x shape:", x.shape)

        if self.log_fidelity:
            self.cost = []
            with torch.no_grad():
                # data_fidelity.append(self.data_fidelity(torch.zeros_like(x), m).cpu().numpy().tolist())
                self.cost.append(self.cost_fun(x, m).cpu().numpy().tolist())
        if self.x_gt is not None:
            self.mse = []
            with torch.no_grad():
                self.mse.append(self.mse_fun(x, self.x_gt).cpu().numpy().tolist())

        u = None

        for i in range(self.iter_stop):
            # gradient step (data fidelity)
            res = self.acqu.meas_op.forward_H(x) - m
            if self.wls:
                res = res / meas_variance
                upd = step[i].reshape(bc, 1) * self.acqu.meas_op.adjoint(res)
            else:
                upd = step[i] * self.acqu.meas_op.adjoint(res)
            x = x - upd

            if i == 0 and self.res_learn and self.x0 == 0:
                # if x0 does not exist
                z0 = x.detach().clone()

            # proximal step (prior)
            if isinstance(self.denoi, nn.ModuleList):
                x = self.denoi[i](x)
            else:
                x = self.denoi(x)

            if self.log_fidelity:
                with torch.no_grad():
                    self.cost.append(self.cost_fun(x, m).cpu().numpy().tolist())
            # Compute mse if ground truth is field
            if self.x_gt is not None:
                with torch.no_grad():
                    self.mse.append(self.mse_fun(x, self.x_gt).cpu().numpy().tolist())

        if self.log_fidelity:
            print(f"Data fidelity: {(self.cost)}. Stepsize: {self.step}")
        if self.x_gt is not None:
            print(f"|x - x_gt| = {self.mse}")

        if self.res_learn:
            # z=x-step*grad(L), x = P(z), x_end = z0 + P(z)
            x = x + z0
        return x

    def reconstruct_expe(self, x):
        r"""Reconstruction step of a reconstruction network

        Same as :meth:`reconstruct` reconstruct except that:

        1. The preprocessing step estimates the image intensity for normalization

        2. The output images are "denormalized", i.e., have units of photon counts

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: :math:`(BC,2M)`

            :attr:`output`: :math:`(BC,1,H,W)`
        """
        # Preprocessing
        x, N0_est = self.prep.forward_expe(x, self.acqu.meas_op)  # shape x = [b*c, M]
        # print(N0_est)

        # measurements to image domain processing
        x = self.pinv(x, self.acqu.meas_op)

        # Denoising
        x = self.denoi(x)
        # print(x.max())

        # Denormalization
        x = self.prep.denormalize_expe(
            x, N0_est, self.acqu.meas_op.h, self.acqu.meas_op.w
        )
        return x
