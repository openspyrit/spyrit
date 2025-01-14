"""
Inverse methods for inverse problems.
"""

from typing import Union

import torch
import torch.nn as nn

import spyrit.core.meas as meas
import spyrit.core.torch as spytorch


# =============================================================================
class PseudoInverse(nn.Module):
    r"""Moore-Penrose pseudoinverse.

    Considering linear measurements :math:`y = Hx`, where :math:`H` is the
    measurement matrix and :math:`x` is a vectorized image, it estimates
    :math:`x` from :math:`y` by computing :math:`\hat{x} = H^\dagger y`, where
    :math:`H` is the Moore-Penrose pseudo inverse of :math:`H`.

    The pseudo inverse can either be computed using the function
    :func:`torch.linalg.lstsq` at each forward pass or computed once and stored
    at initialization using the function :func:`torch.linalg.pinv`. The
    parameter :attr:`store_pinv` controls this behavior.

    .. note::
        When :attr:`store_pinv` is True, additional parameters (such as
        regularization parameters) can be passed as keyword arguments to the
        class constructor.

    .. note::
        When :attr:`store_pinv` is False, additional parameters (such as
        regularization parameters) can be passed as keyword arguments to the
        forward method of this class.

    Args:
        :attr:`meas_op`: Measurement operator. Any class that implements a
        :meth:`matrix_to_inverse` method can be used, e.g.,
        :class:`~spyrit.core.meas.Linear`.

        :attr:`store_pinv` (bool): If False, the pseudo-inverse is not computed
        explicitly but instead the least squares solution is computed at each
        forward pass using the function :func:`torch.linalg.lstsq`. This is
        a more numerically stable but slower approach when using large batches.
        If True, computes and stores at initialization the pseudo-inverse
        (:func:`torch.linalg.pinv`) of the measurement matrix. This is useful
        when the same measurement matrix is used multiple times. Default: False

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

    def __init__(
        self,
        meas_op: Union[meas.Linear, meas.DynamicLinear],
        regularization: str = "rcond",
        store_pinv: bool = False,
        reshape_output: bool = False,
        **kwargs,
    ) -> None:

        super().__init__()
        self.meas_op = meas_op
        self.store_pinv = store_pinv
        self.reshape_output = reshape_output

        if self.store_pinv:
            self.pinv = spytorch.regularized_pinv(
                self.meas_op.matrix_to_inverse, regularization, **kwargs
            )

    def forward(self, y: torch.tensor, **kwargs) -> torch.tensor:
        r"""Computes pseudo-inverse of measurements.

        If :attr:`self.store_pinv` is True, computes the product of the
        stored pseudo-inverse and the measurements. If False, computes the
        least squares solution of the measurements. In this case, additional
        keyword arguments can be passed to the forward method to specify
        regularization parameters to the :func:`torch.linalg.lstsq` function.

        Args:
            :attr:`y`: Batch of measurement vectors.

            :attr:`args`: Additional arguments that are passed to
            :func:`torch.linalg.lstsq` when :attr:`store_pinv` is False.

            :attr:`kwargs`: Additional keyword arguments that are passed to
            :func:`torch.linalg.lstsq` when :attr:`store_pinv` is False.

        Shape:

            :attr:`y`: :math:`(*, M)`, where :math:`*` is any number of
            dimensions and :math:`M` is the number of measurements of the
            measurement operator (:attr:`meas_op.n_meas`).

            :attr:`output`: :math:`(*, N)`, where :math:`N` is the same number of
            dimensions, and :math:`N` is the number of items measured (pixels)
            of the measurement operator (:attr:`meas_op.n_pixels`).

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
        # Expand y to the number of columns of the pseudo-inverse
        y = y.unsqueeze(-1)

        if self.store_pinv:
            # Expand the pseudo-inverse to the batch size of y
            pinv = self.pinv.expand(*y.shape[:-1], *self.pinv.shape)
            y = torch.matmul(pinv, y)

        else:
            matrix_to_inverse = self.meas_op.matrix_to_inverse.expand(
                y.shape[:-1], *self.meas_op.matrix_to_inverse.shape
            )
            y = torch.linalg.lstsq(matrix_to_inverse, y).solution

        y = y.squeeze(-1)

        if self.reshape_output:
            y = self.meas_op.unvectorize(y)

        return y


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
        - :attr:`meas_op` : Measurement operator (see :class:`~spyrit.core.meas`).
        Its measurement operator has shape :math:`(M, N)`, with :math:`M` the
        number of measurements and :math:`N` the number of pixels in the image.

        - :attr:`sigma` : Signal covariance prior, of shape :math:`(N, N)`.

        - :attr:`diagonal_approximation` : A boolean indicating whether to set
        the non-diagonal elements of :math:`A \Sigma A^T` to zero. Default is
        False. If True, this speeds up the computation of the inverse
        :math:`(A \Sigma A^T + \Sigma_\alpha)^{-1}`.

    Attributes:
        - :attr:`meas_op` : Measurement operator initialized as :attr:`meas_op`.

        - :attr:`diagonal_approximation` : Indicates if the diagonal approximation
        is used.

        - :attr:`img_shape` : Shape of the image, initialized as :attr:`meas_op.img_shape`.

        - :attr:`sigma_meas` : Measurement covariance prior initialized as
        :math:`A \Sigma A^T`. If :attr:`diagonal_approximation` is True, the
        non-diagonal elements are set to zero.

        - :attr:`sigma_A_T` : Covariance of the missing measurements initialized
        as :math:`\Sigma A^T`.

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

    def __init__(
        self, meas_op, sigma: torch.tensor, approx=False, reshape_output: bool = False
    ):
        super().__init__()

        self.meas_op = meas_op
        self.sigma = sigma
        self.approx = approx
        self.reshape_output = reshape_output
        self.img_shape = meas_op.img_shape

        A = meas_op.matrix_to_inverse  # get H or A

        # *measurement* covariance
        if approx:
            # store onle the diagonal
            sigma_meas = torch.einsum("ij,jk,ik->i", A, sigma, A)
        else:
            sigma_meas = A @ sigma @ A.T
        self.register_buffer("sigma_meas", sigma_meas)

        # estimation of the missing measurements
        sigma_A_T = torch.mm(sigma, A.mT)
        self.register_buffer("sigma_A_T", sigma_A_T)

    def divide(self, y: torch.tensor, gamma: torch.tensor) -> torch.tensor:
        r"""Computes the division :math:`y \cdot (\Sigma \alpha + (A \Sigma A^T))^{-1}`.

        Measurements `y` are divided by the sum of the measurement covariance.

        If :attr:`self.approx` is True, the inverse is approximated as
        a diagonal matrix, speeding up the computation. Otherwise, the
        inverse is computed with the whole matrix.

        Args:
            y (torch.tensor): Input measurement tensor. Shape :math:`(*, M)`.

            gamma (torch.tensor): Noise covariance tensor. Shape :math:`(*, M, M)`.

        Returns:
            torch.tensor: The divided tensor. Shape :math:`(*, M)`.
        """
        if self.approx:
            return y / (self.sigma_meas + torch.diagonal(gamma, dim1=-2, dim2=-1))
        else:
            # we need to expand the matrices for the solve
            batch_shape = y.shape[:-1]
            expand_shape = batch_shape + (self.sigma_meas.shape)
            y = y.unsqueeze(-1)  # add a dimension to y for batch matrix multiplications
            y = torch.linalg.solve((self.sigma_meas + gamma).expand(expand_shape), y)
            return y.squeeze(-1)

    def forward(
        self, y: torch.tensor, gamma: torch.tensor  # x_0: torch.tensor,
    ) -> torch.tensor:
        r"""Reconstructs the signal from measurements and noise covariance.

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

        if self.reshape_output:
            y = self.meas_op.unvectorize(y)

        return y


# =============================================================================
# class TikhonovMeasurementPriorDiag
