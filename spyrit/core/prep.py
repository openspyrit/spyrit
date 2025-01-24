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
