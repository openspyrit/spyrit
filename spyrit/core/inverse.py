"""
Inverse methods for inverse problems.
"""

from typing import Union
from typing import Union

import torch
import torch.nn as nn

import spyrit.core.meas as meas
import spyrit.core.torch as spytorch


# =============================================================================
class PseudoInverse(nn.Module):
    r"""Moore-Penrose pseudoinverse.

    This allows to solve the linear problem :math:`Ax = B`, by either
    computing the least-squares solution of the equation, or by
    computing the pseudo-inverse matrix of :math:`A`. This behavior
    is defined by the keyword parameter :attr:`store_H_pinv`.

    This class can also handle regularization in the computation of
    the least-squares solution or the matrix pseudo-inverse. The
    available regularization methods are `rcond` (which truncates the
    matrix's SVD below a certain threshold), `L2` and `H1`.

    .. note::
        When :attr:`store_H_pinv` is `True`, additional parameters (such as
        regularization parameters) can be passed as keyword arguments to the
        class constructor.

    .. note::
        When :attr:`store_pinv` is `False`, additional parameters (such as
        regularization parameters) can be passed as keyword arguments to the
        forward method of this class.

    Args:
        :attr:`meas_op`: Measurement operator. See :mod:`spyrit.core.meas`.

        :attr:`regularization` (str): Regularization method. Can be 'rcond',
        'L2', or 'H1'. Default: 'rcond'.

    Keyword Args:
        :attr:`store_H_pinv` (bool): If False, the least squares solution
        is computed at each forward pass using the function :func:`torch.linalg.lstsq`.
        If True, computes and stores at initialization the pseudo-inverse
        of the measurement matrix using the function :func:`torch.linalg.pinv`.
        Default: False

        :attr:`use_fast_pinv` (bool): If True, uses a fast computation of either
        the measurement matrix pseudo-inverse or the least squares solution.
        This only works if the measurement operator has a fast pseudo-inverse
        method. Default: True.

        :attr:`reshape_output` (bool): If True, reshapes the output to the shape
        of the image using :meth:`meas_op.unvectorize`. Default: True.

        :attr:`reg_kwargs`: Additional keyword arguments that are passed to
        :func:`spyrit.core.torch.regularized_pinv` when :attr:`store_pinv` is True
        or to :func:`spyrit.core.torch.resularized_lstsq` when :attr:`store_pinv`
        is False.

    Attributes:
        :attr:`meas_op`: Measurement operator initialized as :attr:`meas_op`.

        :attr:`regularization`: Regularization method initialized as :attr:`regularization`.

        :attr:`store_H_pinv`: Indicates if the pseudo-inverse is stored.

        :attr:`use_fast_pinv`: Indicates if the fast pseudo-inverse is used.

        :attr:`reshape_output`: Indicates if the output is reshaped.

        :attr:`reg_kwargs`: Additional keyword arguments passed to the
        :func:`spyrit.core.torch.regularized_pinv` or :func:`torch.linalg.lstsq`
        functions.

        :attr:`pinv`: The pseudo-inverse of the measurement matrix. It is computed
        only if :attr:`store_H_pinv` is True.

    Example 1:
        >>> from spyrit.core.meas import Linear
        >>> from spyrit.core.inverse import PseudoInverse
        >>> H = torch.randn(10, 15)
        >>> meas_op = Linear(H)
        >>> pinv_op = PseudoInverse(meas_op)
        >>> x = torch.randn(3, 4, 15)
        >>> y = meas_op(x)
        >>> x = pinv_op(y)
        >>> print(x.shape)
        torch.Size([3, 4, 15])

    Example 2: LinearSplit, pseudo-inverse of H (default)
        >>> from spyrit.core.meas import LinearSplit
        >>> from spyrit.core.inverse import PseudoInverse
        >>> H = torch.randn(10, 15)
        >>> meas_op = LinearSplit(H)
        >>> pinv_op = PseudoInverse(meas_op)
        >>> x = torch.randn(3, 4, 15)
        >>> y = meas_op.measure_H(x)
        >>> x = pinv_op(y)
        >>> print(x.shape)
        torch.Size([3, 4, 15])

    Example 3: LinearSplit, pseudo-inverse of A
        >>> from spyrit.core.meas import LinearSplit
        >>> from spyrit.core.inverse import PseudoInverse
        >>> H = torch.randn(10, 15)
        >>> meas_op = LinearSplit(H)
        >>> meas_op.set_matrix_to_inverse('A')
        >>> pinv_op = PseudoInverse(meas_op)
        >>> x = torch.randn(3, 4, 15)
        >>> y = meas_op(x)
        >>> x = pinv_op(y)
        >>> print(x.shape)
        torch.Size([3, 4, 15])
    """

    def __init__(
        self,
        meas_op: Union[meas.Linear, meas.DynamicLinear],
        regularization: str = "rcond",
        *,
        store_H_pinv: bool = False,
        use_fast_pinv: bool = True,
        reshape_output: bool = True,
        **reg_kwargs,
    ) -> None:

        super().__init__()
        self.meas_op = meas_op
        self.regularization = regularization
        self.store_H_pinv = store_H_pinv
        self.use_fast_pinv = use_fast_pinv
        self.reshape_output = reshape_output
        self.reg_kwargs = reg_kwargs

        if self.store_H_pinv:
            # do we have a fast pseudo-inverse computation available?
            if self.use_fast_pinv and hasattr(self.meas_op, "fast_H_pinv"):
                self.pinv = meas_op.fast_H_pinv()
            else:
                self.pinv = spytorch.regularized_pinv(
                    self.meas_op.get_matrix_to_inverse, regularization, **reg_kwargs
                )

        if type(meas_op) is meas.HadamSplit2d:
            self.reshape_output = False

    def forward(self, y: torch.tensor) -> torch.tensor:
        r"""Computes pseudo-inverse of measurements.

            If :attr:`self.store_H_pinv` is True, computes the product of the
            stored pseudo-inverse and the measurements.

            If :attr:`self.store_H_pinv` is False, computes the least squares solution
            of the measurements. In this case, any additional keyword arguments
            passed to the :class:`PseudoInverse` constructor (and store in
            :attr:`self.reg_kwargs` are used here. These can include:
                - :attr:`rcond` (float): Cutoff for small singular values. It is
                used only when :attr:`regularization` is 'rcond'. This parameter
                is fed directly to :func:`torch.linalg.pinv`.
                - Any other keyword arguments that are passed to :func:`torch.linalg.lstsq`.
                Used only when :attr:`regularization` is 'rcond'.
                - :attr:`eta` (float): Regularization parameter. It is used only
                when :attr:`regularization` is 'L2' or 'H1'. This parameter determines
                the amount of regularization applied to the pseudo-inverse.

            Args:
                :attr:`y` (torch.tensor): Batch of measurement vectors of shape :math:`(*, M)`,
                where :math:`*` is any number of dimensions and :math:`M` is the
                number of measurements of the measurement operator (:attr:`meas_op.M`).

            Returns:
                :attr:`output` (torch.tensor): Batch of reconstructed images of shape
                :math:`(*, N)` or the image shape as defined in the measurement operator
                (in `meas_op.meas_shape`) depending on the value of
                :attr:`self.reshape_output`.

        Example 1:
            >>> from spyrit.core.meas import Linear
            >>> from spyrit.core.inverse import PseudoInverse
            >>> H = torch.randn(10, 15)
            >>> meas_op = Linear(H)
            >>> pinv_op = PseudoInverse(meas_op)
            >>> x = torch.randn(3, 4, 15)
            >>> y = meas_op(x)
            >>> x = pinv_op(y)
            >>> print(x.shape)
            torch.Size([3, 4, 15])

        Example 2: LinearSplit, pseudo-inverse of H (default)
            >>> from spyrit.core.meas import LinearSplit
            >>> from spyrit.core.inverse import PseudoInverse
            >>> H = torch.randn(10, 15)
            >>> meas_op = LinearSplit(H)
            >>> pinv_op = PseudoInverse(meas_op)
            >>> x = torch.randn(3, 4, 15)
            >>> y = meas_op.measure_H(x)
            >>> x = pinv_op(y)
            >>> print(x.shape)
            torch.Size([3, 4, 15])

        Example 3: LinearSplit, pseudo-inverse of A
            >>> from spyrit.core.meas import LinearSplit
            >>> from spyrit.core.inverse import PseudoInverse
            >>> H = torch.randn(10, 15)
            >>> meas_op = LinearSplit(H)
            >>> meas_op.set_matrix_to_inverse('A')
            >>> pinv_op = PseudoInverse(meas_op)
            >>> x = torch.randn(3, 4, 15)
            >>> y = meas_op(x)
            >>> x = pinv_op(y)
            >>> print(x.shape)
            torch.Size([3, 4, 15])
        """
        if self.store_H_pinv:
            # Expand the pseudo-inverse to the batch size of y
            pinv = self.pinv.expand(*y.shape[:-1], *self.pinv.shape)
            y = y.unsqueeze(-1)
            y = torch.matmul(pinv, y)
            y = y.squeeze(-1)

        else:
            if self.use_fast_pinv and hasattr(self.meas_op, "fast_pinv"):
                y = self.meas_op.fast_pinv(y)
            else:
                y = spytorch.regularized_lstsq(
                    self.meas_op.get_matrix_to_inverse,
                    y,
                    self.regularization,
                    **self.reg_kwargs,
                )

        if self.reshape_output:
            y = self.meas_op.unvectorize(y)

        return y


# =============================================================================
class Tikhonov(nn.Module):
    r"""Tikhonov regularization.

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

        - :attr:`sigma` : Signal (image) covariance prior, of shape :math:`(N, N)`.

        - :attr:`approx` : A boolean indicating whether to set
        the non-diagonal elements of :math:`A \Sigma A^T` to zero. Default is
        False. If True, this speeds up the computation of the inverse
        :math:`(A \Sigma A^T + \Sigma_\alpha)^{-1}`.

        - :attr:`reshape_output` : A boolean indicating whether to reshape the
        output to the shape of the image. Default is True.

    Attributes:
        - :attr:`meas_op` : Measurement operator initialized as :attr:`meas_op`.

        - :attr:`sigma` : Signal covariance prior initialized as :attr:`sigma`.

        - :attr:`approx` : Indicates if the diagonal approximation
        is used.

        - :attr:`reshape_output` : Indicates if the output is reshaped.

        - :attr:`img_shape` : Shape of the image, initialized as :attr:`meas_op.img_shape`.

        - :attr:`sigma_meas` : Measurement covariance prior initialized as
        :math:`A \Sigma A^T`. If :attr:`approx` is True, the non-diagonal elements
        are set to zero. It is pre-computed at initialization to speed up future
        computations.

        - :attr:`sigma_A_T` : Covariance of the missing measurements initialized
        as :math:`\Sigma A^T`. It is computed at initialization to speed up future
        computations.

    Example:
        >>> from spyrit.core.meas import Linear
        >>> B, H, M, N = 85, 17, 32, 64
        >>> sigma = torch.rand(N, N)
        >>> gamma = torch.rand(M, M)
        >>> A = torch.rand([M,N])
        >>> meas  = Linear(A, meas_shape=(1,N))
        >>> recon = Tikhonov(meas, sigma)
        >>> y = torch.rand(B,H,M)
        >>> x = recon(y, gamma)
        >>> print(y.shape)
        torch.Size([85, 17, 32])
        >>> print(x.shape)
        torch.Size([85, 17, 1, 64])
    """

    def __init__(
        self,
        meas_op: meas.Linear,
        sigma: torch.tensor,
        approx=False,
        reshape_output: bool = True,
    ):
        super().__init__()

        self.meas_op = meas_op
        self.sigma = sigma
        self.approx = approx
        self.reshape_output = reshape_output
        self.img_shape = meas_op.meas_shape

        A = meas_op.get_matrix_to_inverse  # get H or A

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

        Example:
            >>> from spyrit.core.meas import Linear
            >>> M, N = 32, 64
            >>> meas_op = Linear(torch.rand([M,N]), meas_shape=(1,N))
            >>> sigma = torch.rand(N,N)
            >>> tikho = Tikhonov(meas_op, sigma, approx=False, reshape_output=True)
            >>> y = torch.rand(85, 3, M)
            >>> gamma = torch.eye(M).expand(85, 3, M, M)
            >>> print(tikho.divide(y, gamma).shape)
            torch.Size([85, 3, 32])
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

    def forward(self, y: torch.tensor, gamma: torch.tensor) -> torch.tensor:
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
            of shape :math:`(*, M)`.

            :attr:`gamma` (torch.tensor): A batch of noise covariance :math:`\Gamma`
            of shape :math:`(*, M, M)`.

        Returns:
            (torch.tensor): A batch of reconstructed images of shape :math:`(*, N)`
            or the :meth:`meas_op.unvectorize`d version of the image shape.

        Example 1: With reshape_output = True
            >>> from spyrit.core.meas import Linear
            >>> M, N = 32, 64
            >>> b, c, h, w = 85, 3, 8, 8
            >>> meas_op = Linear(torch.rand([M,N]), meas_shape=(1,N))
            >>> x = torch.randn(b,c,h,w)
            >>> y = meas_op(x)
            >>> sigma = torch.rand(N,N)
            >>> tikho = Tikhonov(meas_op, sigma, approx=False, reshape_output=True)
            >>> gamma = torch.eye(M).expand(b, c, M, M)
            >>> print(tikho(y, gamma).shape)
            torch.Size([85, 3, 1, 64])

        Example 2: With reshape_output = False
            >>> from spyrit.core.meas import Linear
            >>> M, N = 32, 64
            >>> b, c, h, w = 85, 3, 8, 8
            >>> meas_op = Linear(torch.rand([M,N]), meas_shape=(1,N))
            >>> x = torch.randn(b,c,h,w)
            >>> y = meas_op(x)
            >>> sigma = torch.rand(N,N)
            >>> tikho = Tikhonov(meas_op, sigma, approx=False, reshape_output=False)
            >>> gamma = torch.eye(M).expand(b, c, M, M)
            >>> print(tikho(y, gamma).shape)
            torch.Size([85, 3, 64])
        """
        y = self.divide(y, gamma)
        y = torch.matmul(self.sigma_A_T, y.unsqueeze(-1)).squeeze(-1)

        if self.reshape_output:
            y = self.meas_op.unvectorize(y)

        return y


# =============================================================================
class TikhonovMeasurementPriorDiag(nn.Module):
    r"""Tikhonov regularisation with prior in the measurement domain.

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
        :attr:`meas_op`: A Hadamard measurement operator (see :mod:`spyrit.core.meas.HadamSplit2d`).

        :attr:`sigma`: Measurement covariance prior with shape :math:`N` x :math:`N`.

    Attributes:
        :attr:`meas_op`: Measurement operator initialized as :attr:`meas_op`.

        :attr:`denoise_weights`: The learnable denoising layer initialized from
        :math:`\Sigma_1`. This layer is a :class:`nn.Parameter`.

        :attr:`comp`: The learnable matrix initialized from :math:`\Sigma_{21}`.
        This matrix is a :class:`nn.Parameter`.

    Example:
        >>> from spyrit.core.meas import HadamSplit2d
        >>> from spyrit.core.inverse import TikhonovMeasurementPriorDiag
        >>> import torch

        >>> acqu = HadamSplit2d(32, 400)
        >>> sigma = torch.rand([32*32, 32*32])
        >>> recon_op = TikhonovMeasurementPriorDiag(acqu, sigma)
        >>> y = torch.rand([10, 3, 400])
        >>> x0 = torch.rand([10, 3, 32, 32])
        >>> var = torch.rand([10, 3, 400])
        >>> x = recon_op(y, x0, var)
        >>> print(x.shape)
        torch.Size([10, 3, 32, 32])
    """

    def __init__(
        self,
        meas_op: meas.HadamSplit2d,
        sigma: torch.tensor,
        reshape_output: bool = False,
    ):
        super().__init__()

        self.meas_op = meas_op
        self.reshape_output = reshape_output

        M = self.meas_op.M
        var_prior = sigma.diag()[:M]
        self.denoise_weights = nn.Parameter(torch.sqrt(var_prior), requires_grad=False)

        Sigma1 = sigma[:M, :M]
        Sigma21 = sigma[M:, :M]
        W = torch.linalg.solve(Sigma1.T, Sigma21.T).T
        self.comp = nn.Parameter(W, requires_grad=False)

    def wiener_denoise(self, x: torch.tensor, var: torch.tensor) -> torch.tensor:
        r"""Returns a denoised version of the input tensor using the variance prior.

        This uses the attribute self.denoise_weights, which is a learnable
        parameter.

        Inputs:
            x (:class:`torch.tensor`): The input tensor to be denoised.

            var (:class:`torch.tensor`): The variance prior.

        Returns:
            :class:`torch.tensor`: The denoised tensor.

        Example:
            >>> from spyrit.core.meas import HadamSplit2d
            >>> from spyrit.core.inverse import TikhonovMeasurementPriorDiag
            >>> import torch

            >>> acqu = HadamSplit2d(32, 400)
            >>> sigma = torch.rand([32*32, 32*32])
            >>> recon_op = TikhonovMeasurementPriorDiag(acqu, sigma)
            >>> y = torch.rand([10, 3, 400])
            >>> var = torch.rand([10, 3, 400])
            >>> print(recon_op.wiener_denoise(y, var).shape)
            torch.Size([10, 3, 400])
        """
        weights_squared = self.denoise_weights**2
        return torch.mul((weights_squared / (weights_squared + var)), x)

    def forward_no_prior(self, x, var):
        r"""Computes the Tikhonov regularization with prior in the measurement domain.

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
                      [\Sigma_1 + \Sigma_\alpha]^{-1} (m - GF x_0).

        See Lemma B.0.5 of the PhD dissertation of A. Lorente Mur (2021):
        https://theses.hal.science/tel-03670825v1/file/these.pdf

        Args:
            :attr:`x` (:class:`torch.tensor`): A batch of measurement vectors
            :math:`m` of shape :math:`(*, M)`.

            :attr:`var` (:class:`torch.tensor`): A batch of measurement noise variances
            :math:`\Sigma_\alpha` of shape :math:`(*, M)`.

        Returns:
            :class:`torch.tensor`: Batch of reconstructed image of shape :math:`(*, \sqrt{N}, \sqrt{N})`.

        Example:
            >>> from spyrit.core.meas import HadamSplit2d
            >>> from spyrit.core.inverse import TikhonovMeasurementPriorDiag
            >>> import torch

            >>> acqu = HadamSplit2d(32, 400)
            >>> sigma = torch.rand([32*32, 32*32])
            >>> recon_op = TikhonovMeasurementPriorDiag(acqu, sigma)
            >>> y = torch.rand([10, 3, 400])
            >>> var = torch.rand([10, 3, 400])
            >>> x = recon_op.forward_no_prior(y, var)
            >>> print(x.shape)
            torch.Size([10, 3, 32, 32])
        """
        y1 = self.wiener_denoise(x, var)
        y2 = y1 @ self.comp.T

        y = torch.cat((y1, y2), -1)
        y = self.meas_op.fast_pinv(y)
        # if self.reshape_output:
        #    y = self.meas_op.unvectorize(y)
        return y

    def forward(
        self,
        x: torch.tensor,
        x_0: torch.tensor,
        var: torch.tensor,
    ) -> torch.tensor:
        r"""Computes the Tikhonov regularization with prior in the measurement domain.

        This method, unlike the :meth:`forward_no_prior` method, allows for a
        non-zero mean image prior :math:`x_0`. We approximate the solution as:

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
            :attr:`x` (:class:`torch.tensor`): A batch of measurement vectors
            :math:`m` with shape :math:`(*, M)`.

            :attr:`x_0` (:class:`torch.tensor`): A batch of prior images
            :math:`x_0` with shape :math:`(*, \sqrt{N}, \sqrt{N})`.

            :attr:`var` (:class:`torch.tensor`): A batch of measurement noise
            variances :math:`\Sigma_\alpha` with shape :math:`(*, M)`.

            :attr:`meas_op` (:class:`torch.tensor`): A measurement operator
            that provides :math:`GF` and :math:`F^{-1}`.

        Output:
            (:class:`torch.tensor`): A batch of images with shape :math:`(*, \sqrt{N}, \sqrt{N})`.

        Example:
            >>> from spyrit.core.meas import HadamSplit2d
            >>> from spyrit.core.inverse import TikhonovMeasurementPriorDiag
            >>> import torch

            >>> acqu = HadamSplit2d(32, 400)
            >>> sigma = torch.rand([32*32, 32*32])
            >>> recon_op = TikhonovMeasurementPriorDiag(acqu, sigma)
            >>> y = torch.rand([10, 3, 400])
            >>> x0 = torch.rand([10, 3, 32, 32])
            >>> var = torch.rand([10, 3, 400])
            >>> x = recon_op(y, x0, var)
            >>> print(x.shape)
            torch.Size([10, 3, 32, 32])
        """
        x = x - self.meas_op.forward_H(x_0)
        x = self.forward_no_prior(x, var)
        x += x_0
        return x
