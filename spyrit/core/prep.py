"""
Preprocessing operators applying affine transformations to the measurements.

There are two classes in this module: :class:`DirectPoisson` and
:class:`SplitPoisson`. The first one is used for direct measurements (i.e.
without splitting the measurement matrix in its positive and negative parts),
while the second one is used for split measurements.
"""

from typing import Union

import torch
import torch.nn as nn

import spyrit.core.inverse as inverse


# =============================================================================
class Unsplit(nn.Module):
    r"""Preprocess the data acquired with a split measurement operator.

    This class handles split measurements by either adding or subtracting the
    alternating odd- and even-indexed measurements. It is equivalent to either
    of these lines:

    .. code-block:: python

        measmts_preprocessed = measmts[..., 0::2] + measmts[..., 1::2]
        measmts_preprocessed = measmts[..., 0::2] - measmts[..., 1::2]

    Args:
        None

    Attributes:
        None

    Example:
        >>> import spyrit.core.meas as meas
        >>> import spyrit.core.prep as prep
        >>> H = torch.rand([400,32*32])  # 400 measurements, 32x32 image
        >>> img = torch.rand([10,32*32])  # 10 images of size 32x32
        >>> meas_op = meas.LinearSplit(H)
        >>> split_op = prep.Unsplit()
        >>> measmts = meas_op(img)  # shape (10, 800) because of splitting
        >>> print(split_op(measmts).shape)
        torch.Size([10, 400])
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(y: torch.tensor, mode: str = "sub") -> torch.tensor:
        r"""Preprocess to compensate for splitting of the measurement operator.

        It computes `y[..., 0::2] - y[..., 1::2]` if `mode` is 'sub' or
        `y[..., 0::2] + y[..., 1::2]` if `mode` is 'add'.

        Args:
            y (torch.tensor): batch of measurement vectors of shape :math:`(*, 2m)`

            mode (str, optional): 'sub' or 'add'. If 'sub', the difference between
            the even and odd indices is computed. If 'add', the sum is computed.
            Defaults to 'sub'.

        Returns:
            torch.tensor: The tensor of shape :math:`(*, m)` with the even- and
            odd- indexed values of the input tensor subtracted or added.
        """
        if mode == "sub":
            return y[..., 0::2] - y[..., 1::2]
        elif mode == "add":
            return y[..., 0::2] + y[..., 1::2]
        else:
            raise ValueError(f"mode should be either 'sub' or 'add' (found {mode})")


# =============================================================================
class Rescale(nn.Module):
    r"""Rescale a tensor from :math:`[0,\alpha]` to :math:`[0,1]`.

    This effectively divides the input tensor by :math:`\alpha`. It is useful
    if the measurements are acquired with some gain factor that needs to be
    compensated for.

    Args:
        alpha (float): The value to divide the input tensor by.

    Attributes:
        alpha (float): The value to divide the input tensor by.
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, y: torch.tensor) -> torch.tensor:
        r"""Rescale the input tensor by dividing it by :attr:`alpha`.

        Args:
            y (torch.tensor): The input tensor to rescale.

        Returns:
            torch.tensor: The rescaled tensor with same shape.
        """
        return y / self.alpha

    def sigma(self, y: torch.tensor) -> torch.tensor:
        r"""Estimates the variance of raw measurements.

        The variance is estimated as :math:`x / \alpha^2`, where :math:`x` are
        *raw* measurements, i.e. before any rescaling.

        Args:
            x (torch.tensor): batch of measurement vectors, of any shape.

        Returns:
            torch.tensor: The estimated variance of the measurements, with the
            same shape.
        """
        return y / (self.alpha**2)


# =============================================================================
class UnsplitRescale(Rescale):
    r"""Unsplit a tensor and rescale it from :math:`[0,\alpha]` to :math:`[0,1]`.

    This is equivalent to appplying successively a :class:`Unsplit` and a
    :class:`Rescale` operator.

    Args:
        alpha (float): The value to divide the input tensor by.

    Attributes:
        alpha (float): The value to divide the input tensor by.
    """

    def __init__(self, alpha):
        super().__init__(alpha)

    def forward(self, y: torch.tensor) -> torch.tensor:
        r"""Rescale the input tensor and unsplit it.

        Depending on the value of `mode`, it returns `(y[..., 0::2] - y[..., 1::2])/alpha`
        if `mode` is "sub" (default) or `(y[..., 0::2] + y[..., 1::2])/alpha`
        if `mode` is "add.

        Args:
            y (torch.tensor): Input tensor of shape :math:`(*, 2m)`

            mode (str): Whether the odd indices should be added to or subtracted
            from the even indices. If "sub", the difference is computed. If "add",
            the sum is computed. Defaults to "sub".

        Returns:
            torch.tensor: The rescaled and unsplit tensor of shape :math:`(*, m)`.
        """
        y = Unsplit.forward(y, mode="sub")  # Unsplit
        y = super().forward(y)  # Rescale
        return y

    def sigma(self, y: torch.tensor) -> torch.tensor:
        r"""Estimates the variance of raw *split* measurements.

        The variance is estimated as :math:`(x[0::2]+x[1::2]) / \alpha^2`.

        .. important::
            This assumes the measurements have been acquired with a split measurement
            operator (see :class:`spyrit.core.meas.LinearSplit`).

        Args:
            y (torch.tensor): batch of measurements, of shape :math:`(*, 2m)`.

        Returns:
            torch.tensor: The estimated variance of the measurements, with the
            shape :math:`(*, m)`.
        """
        y = Unsplit.forward(y, mode="add")
        y = super().sigma(y)
        return y


# =============================================================================
class RescaleEstim(nn.Module):
    r"""Rescale measurements by an estimated gain value :math:`alpha`.

    The gain value :math:`alpha` to divide the measurements by is estimated in
    two different ways: either by taking the mean of the measurements or by
    taking the maximum value of the pseudo-inverse of the measurements.

    The first method, referred to as `mean`, defines :math:`alpha` as the
    average measurement value of a single pixel. The second method, referred to
    as `pinv`, computes the pseudo inverse of the measurements and defines
    :math:`alpha` as the maximum over the last dimension of the resulting tensor.

    The `mean` method is faster but does not guarantee that the gain value is
    accurate. The `pinv` method is slower, but yields a more accurate estimate
    of the gain value.

    .. important:
        This class is intended to be used only with measurements acquired with
        no splitting. For split measurements, use :class:`UnsplitRescaleEstim`.

    Args:
        meas_op (spyrit.core.meas.Linear): The measurement operator used to get
        the measurements. It should not be a split measurement operator.

        estim_mode (str, optional): The method to estimate the gain value. Can
        be either "mean" or "pinv". Defaults to "mean".

        **pinv_kwargs: Additional keyword arguments to pass to the pseudo-inverse
        computation. Only used if `estim_mode` is "pinv".

    Attributes:
        self.meas_op (spyrit.core.meas.Linear): The measurement operator used to
        get the measurements.

        self.estim_mode (str): The method to estimate the gain value.

        self.pinv_kwargs (dict): Additional keyword arguments to pass to the
        pseudo-inverse initialization. Only used if `estim_mode` is "pinv".

        self.pinv (spyrit.core.inverse.PseudoInverse): The pseudo-inverse
        operator. Exists only if `estim_mode` is "pinv".
    """

    def __init__(self, meas_op, estim_mode: str = "mean", **pinv_kwargs):
        super().__init__()
        self.meas_op = meas_op
        self.estim_mode = estim_mode
        self.pinv_kwargs = pinv_kwargs
        if estim_mode == "pinv":
            self.pinv = inverse.PseudoInverse(self.meas_op, **pinv_kwargs)

    def mean_estim(self, y: torch.tensor) -> torch.tensor:
        r"""Estimate the gain value by taking the mean of the measurements.

        Args:
            y (torch.tensor): The measurements to estimate the gain value from.
            Has shape :math:`(*, m)`, where :math:`m` is the number of
            measurements as defined in the measurement operator (and accessible
            in :attr:`meas_op.M`) and `*` can be any number of dimensions.

        Returns:
            torch.tensor: The estimated gain value of shape :math:`(*, 1)`.
        """
        y = torch.sum(y, -1, keepdim=True)
        # take the matrix *H* because the measurements are NOT split
        divisor = self.meas_op.H.sum(dim=-1, keepdim=True).expand(y.shape)
        alpha = torch.div(y, divisor)
        return alpha

    def pinv_estim(self, y: torch.tensor) -> torch.tensor:
        r"""Estimate the gain value by taking the maximum of the pseudo-inverse.

        Args:
            y (torch.tensor): The measurements to estimate the gain value from.
            Has any shape :math:`(*, m)`, where :math:`m` is the number of
            measurements as defined in the measurement operator (and accessible
            in :attr:`meas_op.M`) and `*` can be any number of dimensions.

        Returns:
            torch.tensor: The estimated gain value of shape :math:`(*, 1)`.
        """
        y_pinv = self.pinv(y)
        alpha = torch.max(y_pinv, -1, keepdim=True)
        return alpha

    def estim_alpha(self, y: torch.tensor) -> torch.tensor:
        r"""Estimate the gain value :math:`alpha` from the measurements.

        This method calls either :meth:`pinv_estim` or :meth:`mean_estim`
        depending on the value of :attr:`estim_mode`.

        Args:
            y (torch.tensor): The measurements to estimate the gain value from.
            Has any shape :math:`(*, m)`, where :math:`m` is the number of
            measurements as defined in the measurement operator (and accessible
            in :attr:`meas_op.M`) and `*` can be any number of dimensions.

        Returns:
            torch.tensor: The estimated gain value of shape :math:`(*, 1)`.
        """
        if self.estim_mode == "pinv":
            return self.pinv_estim(y)
        elif self.estim_mode == "mean":
            return self.mean_estim(y)
        else:
            raise ValueError(
                f"self.estim_mode should be either 'pinv' or 'mean' (found {self.estim_mode})"
            )

    def forward(self, y: torch.tensor) -> torch.tensor:
        r"""Divide the measurements by the estimated gain value.

        The gain value is estimated by calling :meth:`estim_alpha`, which in turn
        calls either :meth:`pinv_estim` or :meth:`mean_estim` depending on the
        value of :attr:`estim_mode`. The measurements are then divided by the
        estimated gain value.

        Args:
            y (torch.tensor): The measurements to divide by the estimated gain
            value. Has any shape :math:`(*, m)`, where :math:`m` is the number of
            measurements as defined in the measurement operator (and accessible
            in :attr:`meas_op.M`) and `*` can be any number of dimensions.

        Returns:
            torch.tensor: The measurements divided by the estimated gain value.
            Have the same shape as the input tensor :math:`(*, m)`.
        """
        alpha = self.estim_alpha(y)
        y = y / alpha
        return y


# =============================================================================
class UnsplitRescaleEstim(RescaleEstim):
    r"""Unsplit a tensor then rescale it using an estimated gain value.

    This is equivalent to applying successively a :class:`Unsplit` and a
    :class:`RescaleEstim` operator.

    The gain value :math:`alpha` to divide the measurements by is estimated in
    two different ways: either by taking the mean of the measurements or by
    taking the maximum value of the pseudo-inverse of the measurements.

    The first method, referred to as `mean`, defines :math:`alpha` as the
    average measurement value of a single pixel. The second method, referred to
    as `pinv`, computes the pseudo inverse of the measurements and defines
    :math:`alpha` as the maximum over the last dimension of the resulting tensor.

    The `mean` method is faster but does not guarantee that the gain value is
    accurate. The `pinv` method is slower, but yields a more accurate estimate
    of the gain value.

    .. important:
        This class is intended to be used only with measurements acquired with
        splitting. For measurements acquired without splitting, use
        :class:`RescaleEstim`.

    Args:
        meas_op (spyrit.core.meas.Linear): The measurement operator used to get
        the measurements. It should not be a split measurement operator.

        estim_mode (str, optional): The method to estimate the gain value. Can
        be either "mean" or "pinv". Defaults to "mean".

        **pinv_kwargs: Additional keyword arguments to pass to the pseudo-inverse
        computation. Only used if `estim_mode` is "pinv".

    Attributes:
        self.meas_op (spyrit.core.meas.Linear): The measurement operator used to
        get the measurements.

        self.estim_mode (str): The method to estimate the gain value.

        self.pinv_kwargs (dict): Additional keyword arguments to pass to the
        pseudo-inverse initialization. Only used if `estim_mode` is "pinv".

        self.pinv (spyrit.core.inverse.PseudoInverse): The pseudo-inverse
        operator. Exists only if `estim_mode` is "pinv".
    """

    def __init__(self, meas_op, estim_mode: str = "mean", **pinv_kwargs):
        super().__init__(meas_op, estim_mode, **pinv_kwargs)

    def mean_estim(self, y):
        r"""Estimate the gain value by taking the mean of the measurements.

        .. important::
            This method is only to be called on measurements acquired with
            split measurement operators.

        Args:
            y (torch.tensor): The measurements to estimate the gain value from.
            Has shape :math:`(*, 2m)`, where :math:`m` is the number of
            measurements as defined in the measurement operator (and accessible
            in :attr:`meas_op.M`) and `*` can be any number of dimensions.

        Returns:
            torch.tensor: The estimated gain value of shape :math:`(*, 1)`.
        """
        y = torch.sum(y, -1, keepdim=True)
        # take the matrix *A* because the measurements ARE split
        divisor = self.meas_op.A.sum(dim=-1, keepdim=True).expand(y.shape)
        alpha = torch.div(y, divisor)
        return alpha

    def pinv_estim(self, y: torch.tensor) -> torch.tensor:
        r"""Estimate the gain value by taking the maximum of the pseudo-inverse.

        Args:
            y (torch.tensor): The measurements to estimate the gain value from.
            Has any shape :math:`(*, 2m)`, where :math:`m` is the number of
            measurements as defined in the measurement operator (and accessible
            in :attr:`meas_op.M`) and `*` can be any number of dimensions.

        Returns:
            torch.tensor: The estimated gain value of shape :math:`(*, 1)`.
        """
        return super().pinv_estim(y)

    def estim_alpha(self, y: torch.tensor) -> torch.tensor:
        r"""Estimate the gain value :math:`alpha` from the measurements.

        This method calls either :meth:`pinv_estim` or :meth:`mean_estim`
        depending on the value of :attr:`estim_mode`.

        Args:
            y (torch.tensor): The measurements to estimate the gain value from.
            Has any shape :math:`(*, 2m)`, where :math:`m` is the number of
            measurements as defined in the measurement operator (and accessible
            in :attr:`meas_op.M`) and `*` can be any number of dimensions.

        Returns:
            torch.tensor: The estimated gain value of shape :math:`(*, 1)`.
        """
        return super().estim_alpha(y)

    def forward(self, y: torch.tensor, mode: str = "sub") -> torch.tensor:
        r"""Divide the measurements by the estimated gain value.

        The gain value is estimated by calling :meth:`estim_alpha`, which in turn
        calls either :meth:`pinv_estim` or :meth:`mean_estim` depending on the
        value of :attr:`estim_mode`. The measurements are then divided by the
        estimated gain value.

        Args:
            y (torch.tensor): The measurements to divide by the estimated gain
            value. Has shape :math:`(*, 2m)`, where :math:`m` is the number of
            measurements as defined in the measurement operator (and accessible
            in :attr:`meas_op.M`) and `*` can be any number of dimensions.

        Returns:
            torch.tensor: The measurements unsplitted and divided by the
            estimated gain value. Has shape :math:`(*, m)`.
        """
        y = Unsplit.forward(y, mode=mode)
        y = super().forward(y)  # estimate alpha and divide
        return y


# =============================================================================
class Rerange(nn.Module):
    r"""Applies the affine transform that maps :math:`[x,y]` to :math:`[a,b]`.

    Args:
        input_range (tuple): The input range :math:`[x,y]`.

        output_range (tuple): The output range :math:`[a,b]`.

    Attributes:
        input_range (tuple): The input range :math:`[x,y]`.

        output_range (tuple): The output range :math:`[a,b]`.
    """

    def __init__(
        self, input_range: Union[tuple, list], output_range: Union[tuple, list]
    ):
        super().__init__()
        self.input_range = input_range
        self.output_range = output_range

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Applies the affine transform to the input tensor.

        It corresponds to the affine transformation that maps the input range
        :math:`[x,y]` to the output range :math:`[a,b]`.

        Args:
            x (torch.tensor): The input tensor to apply the affine transform to.
            Has arbitrary shape.

        Returns:
            torch.tensor: The input tensor with the affine transform applied.
            It has the same shape as the input tensor.
        """
        a, b = self.input_range
        c, d = self.output_range
        return (x - a) * (d - c) / (b - a) + c

    def backward(self, x: torch.tensor) -> torch.tensor:
        r"""Applies the inverse affine transform to the input tensor.

        Args:
            x (torch.tensor): The input tensor to apply the inverse affine
            transform to. Has arbitrary shape.

        Returns:
            torch.tensor: The input tensor with the inverse affine transform
            applied. It has the same shape as the input tensor.
        """
        a, b = self.input_range
        c, d = self.output_range
        return (x - c) * (b - a) / (d - c) + a

    def inverse(self):
        r"""Returns a different instance of the same class with the input and
        output ranges swapped.

        Returns:
            Rerange: The inverse affine transform that maps :math:`[a,b]` to
            :math:`[x,y]`.
        """
        return Rerange(self.output_range, self.input_range)

    def __repr__(self):
        return f"Rerange({self.input_range} -> {self.output_range})"


# =============================================================================
# class DirectPoisson(nn.Module):
#     r"""
#     Preprocess the raw data acquired with a direct measurement operator assuming
#     Poisson noise. It also compensates for the affine transformation applied
#     to the images to get positive intensities.

#     It computes :math:`m = \frac{2}{\alpha}y - H1` and the variance
#     :math:`\sigma^2 = 4\frac{y}{\alpha^{2}}`, where :math:`y = Hx` are obtained
#     using a direct linear measurement operator (see :mod:`spyrit.core.Linear`),
#     :math:`\alpha` is the image intensity, and 1 is the all-ones vector.

#     Args:
#         :attr:`alpha`: maximun image intensity :math:`\alpha` (in counts)

#         :attr:`meas_op`: measurement operator (see :mod:`~spyrit.core.meas`)


#     Example:
#         >>> H = torch.rand([400,32*32])
#         >>> meas_op =  Linear(H)
#         >>> prep_op = DirectPoisson(1.0, meas_op)

#     """

#     def __init__(self, alpha: float, meas_op):
#         super().__init__()
#         self.alpha = alpha
#         self.meas_op = meas_op

#         self.M = meas_op.M
#         self.N = meas_op.N
#         self.h = meas_op.h
#         self.w = meas_op.w

#         self.max = nn.MaxPool2d((self.h, self.w))
#         # self.register_buffer("H_ones", meas_op(torch.ones((1, self.N))))

#     # generate H_ones on the fly as it is memmory intensive and easy to compute
#     # ?? Why does it returns float64 ??
#     @property
#     def H_ones(self):
#         return self.meas_op.H.sum(dim=-1).to(self.device)

#     @property
#     def device(self):
#         return self.meas_op.device

#     def forward(self, x: torch.tensor) -> torch.tensor:
#         r"""
#         Preprocess measurements to compensate for the affine image normalization

#         It computes :math:`\frac{2}{\alpha}x - H1`, where H1 represents the
#         all-ones vector.

#         Args:
#             :attr:`x`: batch of measurement vectors

#         Shape:
#             x: :math:`(B, M)` where :math:`B` is the batch dimension

#             meas_op: the number of measurements :attr:`meas_op.M` should match
#             :math:`M`.

#             Output: :math:`(B, M)`

#         Example:
#             >>> x = torch.rand([10,400], dtype=torch.float)
#             >>> H = torch.rand([400,32*32])
#             >>> meas_op =  Linear(H)
#             >>> prep_op = DirectPoisson(1.0, meas_op)
#             >>> m = prep_op(x)
#             >>> print(m.shape)
#             torch.Size([10, 400])
#         """
#         # normalize
#         # H_ones = self.H_ones.expand(x.shape[0], self.M)
#         x = 2 * x / self.alpha - self.H_ones.to(x.dtype).expand(x.shape)
#         return x

#     def sigma(self, x: torch.tensor) -> torch.tensor:
#         r"""Estimates the variance of raw measurements

#         The variance is estimated as :math:`\frac{4}{\alpha^2} x`

#         Args:
#             :attr:`x`: batch of measurement vectors

#         Shape:
#             :attr:`x`: :math:`(B,M)` where :math:`B` is the batch dimension

#             Output: :math:`(B, M)`

#         Example:
#             >>> x = torch.rand([10,400], dtype=torch.float)
#             >>> v = prep_op.sigma(x)
#             >>> print(v.shape)
#             torch.Size([10, 400])

#         """
#         # *4 to account for the image normalized [-1,1] -> [0,1]
#         return 4 * x / (self.alpha**2)

#     def denormalize_expe(
#         self, x: torch.tensor, beta: torch.tensor, h: int = None, w: int = None
#     ) -> torch.tensor:
#         r"""Denormalize images from the range [-1;1] to the range [0; :math:`\beta`]

#         It computes :math:`m = \frac{\beta}{2}(x+1)`, where
#         :math:`\beta` is the normalization factor, that can be different for each
#         image in the batch.

#         Args:
#             - :attr:`x` (torch.tensor): Batch of images
#             - :attr:`beta` (torch.tensor): Normalization factor. It should have
#             the same shape as the batch.
#             - :attr:`h` (int, optional): Image height. If None, it is
#             deduced from the shape of :attr:`x`. Defaults to None.
#             - :attr:`w` (int): Image width. If None, it is deduced from the
#             shape of :attr:`x`. Defaults to None.

#         Shape:
#             - :attr:`x`: :math:`(*, h, w)` where :math:`*` indicates any batch
#             dimensions
#             - :attr:`beta`: :math:`(*)` or :math:`(1)` if the same for all
#             images
#             - :attr:`h`: int
#             - :attr:`w`: int
#             - Output: :math:`(*, h, w)`

#         Example:
#             >>> x = torch.rand([10, 1, 32,32], dtype=torch.float)
#             >>> beta = 9*torch.rand([10])
#             >>> y = split_op.denormalize_expe(x, beta, 32, 32)
#             >>> print(y.shape)
#             torch.Size([10, 1, 32, 32])
#         """
#         if h is None:
#             h = x.shape[-2]
#         if w is None:
#             w = x.shape[-1]

#         if beta.numel() == 1:
#             beta = beta.expand(x.shape)
#         else:
#             # Denormalization
#             beta = beta.reshape(*beta.shape, 1, 1)
#             beta = beta.expand((*beta.shape[:-2], h, w))

#         return (x + 1) / 2 * beta

#     def unsplit(self, x: torch.tensor, mode: str = "diff") -> torch.tensor:
#         """Unsplits measurements by combining odd and even indices.

#         The parameter `mode` can be either 'diff' or 'sum'. The first one
#         computes the difference between the even and odd indices, while the
#         second one computes the sum.

#         Args:
#             x (torch.tensor): Measurements, can have any shape.

#             mode (str): 'diff' or 'sum'. If 'diff', the difference between the
#             even and odd indices is computed. If 'sum', the sum is computed.
#             Defaults to 'diff'.

#         Returns:
#             torch.tensor: The input tensor with the even and odd indices
#             of the last dimension combined (either by difference or sum).
#         """
#         if mode == "diff":
#             return x[..., 0::2] - x[..., 1::2]
#         elif mode == "sum":
#             return x[..., 0::2] + x[..., 1::2]
#         else:
#             raise ValueError("mode should be either 'diff' or 'sum'")


# # =============================================================================
# class SplitPoisson(DirectPoisson):
#     r"""
#     Preprocess the raw data acquired with a split measurement operator assuming
#     Poisson noise.  It also compensates for the affine transformation applied
#     to the images to get positive intensities.

#     It computes

#     .. math::

#         m = \frac{y_{+}-y_{-}}{\alpha} - H1

#     and the variance

#     .. math::
#         \sigma^2 = \frac{2(y_{+} + y_{-})}{\alpha^{2}},

#     where :math:`y_{+} = H_{+}x` and :math:`y_{-} = H_{-}x` are obtained using a
#     split measurement operator (see :mod:`spyrit.core.LinearSplit`),
#     :math:`\alpha` is the image intensity, and 1 is the all-ones vector.

#     Args:
#         alpha (float): maximun image intensity :math:`\alpha` (in counts)

#         :attr:`meas_op`: measurement operator (see :mod:`~spyrit.core.meas`)


#     Example:
#         >>> H = torch.rand([400,32*32])
#         >>> meas_op =  LinearSplit(H)
#         >>> split_op = SplitPoisson(10, meas_op)

#     Example 2:
#         >>> Perm = torch.rand([32,32])
#         >>> meas_op = HadamSplit(400, 32,  Perm)
#         >>> split_op = SplitPoisson(10, meas_op)

#     """

#     def __init__(self, alpha: float, meas_op):
#         super().__init__(alpha, meas_op)

#     @property
#     def even_index(self):
#         return range(0, 2 * self.M, 2)

#     @property
#     def odd_index(self):
#         return range(1, 2 * self.M, 2)

#     # @property
#     # def H_ones(self):
#     #     return self.unsplit(super().H_ones, mode="diff")

#     def forward(self, x: torch.tensor) -> torch.tensor:
#         r"""
#         Preprocess to compensates for image normalization and splitting of the
#         measurement operator.

#         It computes :math:`\frac{x[0::2]-x[1::2]}{\alpha} - H1`

#         Args:
#             :attr:`x`: batch of measurement vectors

#         Shape:
#             x: :math:`(*, 2M)` where :math:`*` indicates one or more dimensions

#             meas_op: the number of measurements :attr:`meas_op.M` should match
#             :math:`M`.

#             Output: :math:`(*, M)`

#         Example:
#             >>> x = torch.rand([10,2*400], dtype=torch.float)
#             >>> H = torch.rand([400,32*32])
#             >>> meas_op =  LinearSplit(H)
#             >>> split_op = SplitPoisson(10, meas_op)
#             >>> m = split_op(x)
#             >>> print(m.shape)
#             torch.Size([10, 400])

#         Example 2:
#             >>> x = torch.rand([10,2*400], dtype=torch.float)
#             >>> Perm = torch.rand([32,32])
#             >>> meas_op = HadamSplit(400, 32,  Perm)
#             >>> split_op = SplitPoisson(10, meas_op)
#             >>> m = split_op(x)
#             >>> print(m.shape)
#             torch.Size([10, 400])
#         """
#         # s = x.shape[:-1] + torch.Size([self.M])  # torch.Size([*,M])
#         # H_ones = self.H_ones.expand(s)
#         return super().forward(self.unsplit(x, mode="diff"))

#     def forward_expe(
#         self, x: torch.tensor, meas_op: Union[meas.LinearSplit, meas.HadamSplit]
#     ) -> Tuple[torch.tensor, torch.tensor]:
#         r"""Preprocess to compensate for image normalization and splitting of the
#         measurement operator.

#         It computes

#         .. math:: m = \frac{x[0::2]-x[1::2]}{\alpha},

#         where :math:`\alpha = \max H^\dagger (x[0::2]-x[1::2])`.

#         .. note::
#             Contrary to :meth:`~forward`, the image intensity :math:`\alpha`
#             is estimated from the pseudoinverse of the unsplit measurements. This
#             method is typically called for the reconstruction of experimental
#             measurements, while :meth:`~forward` is called in simulations.

#         The method returns a tuple containing both :math:`m` and :math:`\alpha`

#         Args:
#             :attr:`x`: batch of measurement vectors

#             :attr:`meas_op`: measurement operator (required to estimate
#             :math:`\alpha`)

#             Output (:math:`m`, :math:`\alpha`): preprocess measurement and estimated
#             intensities.

#         Shape:
#             x: :math:`(B, 2M)` where :math:`B` is the batch dimension

#             meas_op: the number of measurements :attr:`meas_op.M` should match
#             :math:`M`.

#             :math:`m`: :math:`(B, M)`

#             :math:`\alpha`: :math:`(B)`

#         Example:
#             >>> x = torch.rand([10,2*400], dtype=torch.float)
#             >>> Perm = torch.rand([32,32])
#             >>> meas_op = HadamSplit(400, 32,  Perm)
#             >>> split_op = SplitPoisson(10, meas_op)
#             >>> m, alpha = split_op.forward_expe(x, meas_op)
#             >>> print(m.shape)
#             >>> print(alpha.shape)
#             torch.Size([10, 400])
#             torch.Size([10])
#         """
#         x = self.unsplit(x, mode="diff")

#         # estimate alpha
#         x_pinv = meas_op.pinv(x)
#         alpha = self.max(x_pinv).squeeze(-1)  # shape is now (b, c, 1)

#         # normalize
#         alpha = alpha.expand(x.shape)
#         x = torch.div(x, alpha)
#         x = 2 * x - self.H_ones.expand(x.shape)

#         alpha = alpha[..., 0]  # shape is (b, c)

#         return x, alpha

#     def sigma(self, x: torch.tensor) -> torch.tensor:
#         r"""Estimates the variance of raw measurements

#         The variance is estimated as :math:`\frac{4}{\alpha^2} (x[0::2]+x[1::2])`

#         Args:
#             :attr:`x`: batch of images in the Hadamard domain

#         Shape:
#             - Input: :math:`(*,2*M)` :math:`*` indicates one or more dimensions
#             - Output: :math:`(*, M)`

#         Example:
#             >>> x = torch.rand([10,2*400], dtype=torch.float)
#             >>> v = split_op.sigma(x)
#             >>> print(v.shape)
#             torch.Size([10, 400])

#         """
#         return super().sigma(self.unsplit(x, mode="sum"))

#     def set_expe(self, gain=1.0, mudark=0.0, sigdark=0.0, nbin=1.0):
#         r"""
#         Sets experimental parameters of the sensor

#         Args:
#             - :attr:`gain` (float): gain (in count/electron)
#             - :attr:`mudark` (float): average dark current (in counts)
#             - :attr:`sigdark` (float): standard deviation or dark current (in counts)
#             - :attr:`nbin` (float): number of raw bin in each spectral channel (if input x results from the sommation/binning of the raw data)

#         Example:
#             >>> split_op.set_expe(gain=1.6)
#             >>> print(split_op.gain)
#             1.6
#         """
#         self.gain = gain
#         self.mudark = mudark
#         self.sigdark = sigdark
#         self.nbin = nbin

#     def sigma_expe(self, x: torch.tensor) -> torch.tensor:
#         r"""
#         Estimates the variance of the measurements that are compensated for
#         splitting but **NOT** for image normalization


#         Args:
#             :attr:`x`: Batch of images in the Hadamard domain.

#         Shape:
#             Input: :math:`(B,2*M)` where :math:`B` is the batch dimension

#             Output: :math:`(B, M)`

#         Example:
#             >>> x = torch.rand([10,2*32*32], dtype=torch.float)
#             >>> split_op.set_expe(gain=1.6)
#             >>> v = split_op.sigma_expe(x)
#             >>> print(v.shape)
#             torch.Size([10, 400])
#         """
#         x = self.unsplit(x, mode="sum")
#         x = (
#             self.gain * (x - 2 * self.nbin * self.mudark)
#             + 2 * self.nbin * self.sigdark**2
#         )
#         x = 4 * x  # to get the cov of an image in [-1,1], not in [0,1]

#         return x

#     def sigma_from_image(
#         self, x: torch.tensor, meas_op: Union[meas.LinearSplit, meas.HadamSplit]
#     ) -> torch.tensor:
#         r"""
#         Estimates the variance of the preprocessed measurements corresponding
#         to images through a measurement operator

#         The variance is estimated as
#         :math:`\frac{4}{\alpha} \{(Px)[0::2] + (Px)[1::2]\}`

#         Args:
#             :attr:`x`: Batch of images

#             :attr:`meas_op`: Measurement operator

#         Shape:
#             :attr:`x`: :math:`(*,N)`

#             :attr:`meas_op`: An operator such that :attr:`meas_op.N` :math:`=N`
#             and :attr:`meas_op.M` :math:`=M`

#             Output: :math:`(*, M)`

#         Example:
#             >>> x = torch.rand([10,2*400], dtype=torch.float)
#             >>> Perm = torch.rand([32,32])
#             >>> meas_op = HadamSplit(400, 32,  Perm)
#             >>> split_op = SplitPoisson(10, meas_op)
#             >>> v = split_op.sigma_from_image(x, meas_op)
#             >>> print(v.shape)
#             torch.Size([10, 400])

#         """
#         x = meas_op(x)
#         x = self.unsplit(x, mode="sum")
#         x = 4 * x / self.alpha  # here alpha should not be squared
#         return x


# # =============================================================================
# class SplitPoissonRaw(SplitPoisson):
#     # ==============================================================================
#     r"""
#     Preprocess the raw data acquired with a split measurement operator assuming
#     Poisson noise.  It also compensates for the affine transformation applied
#     to the images to get positive intensities.

#     It computes the differential measurements

#     .. math::

#         m = \frac{y_{+}-y_{-}}{\alpha} - H1

#     and the corresponding variance

#     .. math::
#         \sigma^2 = \frac{2(y_{+} + y_{-})}{\alpha^{2}},

#     where :math:`y_{+} = H_{+}x` and :math:`y_{-} = H_{-}x` are obtained using a
#     split measurement operator (see :mod:`spyrit.core.LinearSplit`),
#     :math:`\alpha` is a normalisation factor, and 1 is the all-ones vector. This
#     class also estimates the normalisation factor :math:`\alpha`.


#     .. note::

#         Contrary to :class:`SplitPoisson`, the estimation of the normalisation
#         factor is based on the mean of the raw measurement, **not** on the
#         pseudo inverse of the differential mesurements.


#     Args:
#         alpha (float): maximun image intensity :math:`\alpha` (in counts)

#         :attr:`meas_op`: measurement operator (see :mod:`~spyrit.core.meas`)


#     Example:
#         >>> H = torch.rand([400,32*32])
#         >>> meas_op =  LinearSplit(H)
#         >>> split_op = SplitPoissonRaw(10, meas_op)

#     """

#     def __init__(self, alpha: float, meas_op):
#         super().__init__(alpha, meas_op)

#     def forward_expe(
#         self, x: torch.tensor, meas_op: Union[meas.LinearSplit, meas.HadamSplit], dim=-1
#     ) -> Tuple[torch.tensor, torch.tensor]:
#         r"""Preprocess to compensate for image normalization and splitting of the
#         measurement operator.

#         .. note::
#             Contrary to :meth:`~forward`, the image intensity :math:`\alpha`
#             is estimated from the raw measurements. This
#             method is typically called for the reconstruction of experimental
#             measurements, while :meth:`~forward` is called in simulations.

#         Args:
#             :attr:`x`: batch of measurement vectors with shape :math:`(*, 2M)`

#             :attr:`meas_op`: measurement operator (see :mod:`~spyrit.core.meas`).
#             The number of measurements :attr:`meas_op.M` should be equal to
#             :math:`M`.

#             :attr:`dim`: dimensions where the max of the pseudo inverse is
#             computed. Defaults to -1 (i.e., last dimension).

#         Output:
#             Preprocessed measurements :math:`m` with shape :math:`(*, M)`.

#             Estimated intensities :math:`\alpha` with shape :math:`(*)`.

#         Example:
#             >>> H = torch.rand([400,32*32])
#             >>> meas =  LinearSplit(H)
#             >>> split = SplitPoissonRaw(10, meas_op)
#             >>> x = torch.rand([10,2*400], dtype=torch.float)
#             >>> split.set_expe()
#             >>> m, alpha = split.forward_expe(x, meas)
#             >>> print(m.shape)
#             >>> print(alpha.shape)
#             torch.Size([10, 400])
#             torch.Size([1])
#         """

#         # estimate intensity (in counts)
#         z = x[..., self.even_index] + x[..., self.odd_index]
#         mu = torch.mean(z, dim, keepdim=True)
#         alpha = (2 / self.N) * (mu - 2 * self.mudark) / self.gain

#         # alternative based on the variance
#         # var = torch.var(z, dim, keepdim=True)
#         # alpha_2 = (2/self.N)*(var - 2*self.sigdark**2)/self.gain**2

#         # gain = (var - 2*self.sigdark**2)/(mu - 2*self.mudark)

#         # Alternative where all rows of an image have the same normalization
#         alpha = torch.amax(alpha, -2, keepdim=True)

#         # intensity x gain (in counts)
#         norm = alpha * self.gain

#         # unsplit
#         x = x[..., self.even_index] - x[..., self.odd_index]

#         # normalize
#         x = x / norm
#         x = 2 * x - self.H_ones.to(x.dtype)

#         return x, norm  # or alpha? Double check

#     def sigma_expe(self, x: torch.tensor) -> torch.tensor:
#         r"""
#         Estimates the variance of the measurements that are compensated for
#         splitting but **NOT** for image normalization


#         Args:
#             :attr:`x`: Raw measurements with shape :math:`(*, 2M)`.


#         Output:
#             Variance with shape :math:`(*, M)`.

#         Example:
#             >>> x = torch.rand([10,2*32*32], dtype=torch.float)
#             >>> split_op.set_expe(gain=1.6)
#             >>> v = split_op.sigma_expe(x)
#             >>> print(v.shape)
#             torch.Size([10, 400])
#         """
#         # Input shape (b*c, 2*M)
#         # output shape (b*c, M)
#         x = x[..., self.even_index] + x[..., self.odd_index]
#         x = (
#             self.gain * (x - 2 * self.nbin * self.mudark)
#             + 2 * self.nbin * self.sigdark**2
#         )
#         x = 4 * x  # to get the cov of an image in [-1,1], not in [0,1]

#         return x

#     def denormalize_expe(
#         self, x: torch.tensor, beta: torch.tensor, h: int = None, w: int = None
#     ) -> torch.tensor:

#         return (x + 1) / 2 * beta
