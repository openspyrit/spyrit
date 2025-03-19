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
import spyrit.core.meas as meas


# =============================================================================
class Unsplit(nn.Module):
    r"""Preprocess measurements simulated using split measurement operator.

    Given measurements :math:`y\in\mathbb{R}^{2M}`, it computes either :math:`y_+ - y_-` or :math:`y_+ + y_-`. The positive and negative measurements :math:`y_+\in\mathbb{R}^{M}` and :math:`y_-\in\mathbb{R}^{M}` are given by

    .. math::
        y_+ =
        \begin{bmatrix}
            y[0]\\
            y[2]\\
            y[2M-2]\\
        \end{bmatrix}
        \quad\text{and}\quad
        y_- =
        \begin{bmatrix}
            y[1]\\
            y[3]\\
            y[2M-1]\\
        \end{bmatrix}.

    Args:
        None

    Attributes:
        None

    Example:
        >>> import torch
        >>> import spyrit.core.meas as meas
        >>> import spyrit.core.prep as prep
        >>> H = torch.rand([400,32])
        >>> img = torch.rand([10,32])
        >>> meas_op = meas.LinearSplit(H)
        >>> split_op = prep.Unsplit()
        >>> y = meas_op(img)
        >>> m = split_op(y)
        >>> print(y.shape)
        torch.Size([10, 800])
        >>> print(m.shape)
        torch.Size([10, 400])
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(y: torch.tensor, mode: str = "sub") -> torch.tensor:
        r"""Preprocess measurements simulated using split measurement operator.

        Given measurements :math:`y\in\mathbb{R}^{2M}`, it computes either :math:`y_+ - y_-` or :math:`y_+ + y_-`. The positive and negative measurements :math:`y_+\in\mathbb{R}^{M}` and :math:`y_-\in\mathbb{R}^{M}` are given by

        .. math::
            y_+ =
            \begin{bmatrix}
                y[0]\\
                y[2]\\
                y[2M-2]\\
            \end{bmatrix}
            \quad\text{and}\quad
            y_- =
            \begin{bmatrix}
                y[1]\\
                y[3]\\
                y[2M-1]\\
            \end{bmatrix}.

        Args:
            :attr:`y` (:class:`torch.tensor`): Measurement of shape :math:`(*, 2M)`

            :attr:`mode` (str, optional): 'sub' or 'add'. If 'sub', :math:`y_+ - y_-` is returned. If 'add', :math:`y_+ + y_-` is returned.
            Defaults to 'sub'.

        Returns:
            :class:`torch.tensor`: Preprocessed measurements of shape :math:`(*, M)`.

        Example:
            >>> import torch
            >>> import spyrit.core.meas as meas
            >>> import spyrit.core.prep as prep
            >>> H = torch.rand([400,32])
            >>> img = torch.rand([10,32])
            >>> meas_op = meas.LinearSplit(H)
            >>> split_op = prep.Unsplit()
            >>> y = meas_op(img)
            >>> m = split_op(y)
            >>> print(y.shape)
            torch.Size([10, 800])
            >>> print(m.shape)
            torch.Size([10, 400])
        """
        if mode == "sub":
            return y[..., 0::2] - y[..., 1::2]
        elif mode == "add":
            return y[..., 0::2] + y[..., 1::2]
        else:
            raise ValueError(f"mode should be either 'sub' or 'add' (found {mode})")


# =============================================================================
class Rescale(nn.Module):
    r"""Rescale measurements as

    .. math::

        m = \frac{y}{\alpha}

    where :math:`y` is the input tensor and :math:`\alpha` represents some gain.

    .. note::

        This rescale the input tensor from :math:`[0,\alpha]` to :math:`[0,1]`. When measurements are simulated using some gain factor (e.g., Poisson corrupted measurements), the gain is compensated for.

    Args:
        :attr:`alpha` (float): Gain :math:`\alpha`.

    Attributes:
        :attr:`alpha`  (float): Gain :math:`\alpha`.
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, y: torch.tensor) -> torch.tensor:
        r"""Rescale the tensor by dividing it by :math:`\alpha`.

        Args:
            :attr:`y` (:class:`torch.tensor`): Input tensor with arbitrary shape.

        Returns:
            :class:`torch.tensor`: Rescaled tensor with the same shape as the input.
        """
        return y / self.alpha

    def sigma(self, v: torch.tensor) -> torch.tensor:
        r"""Rescale the variance of the measurements

        .. math::

            \text{var}(m) = \frac{\text{var}(y)}{\alpha^2}

        Args:
            :attr:`v` (:class:`torch.tensor`): Variance of :math:`y` with arbitrary shape.

        Returns:
            :class:`torch.tensor`: Rescaled variance with the same shape as the input.
        """
        return v / (self.alpha**2)


# =============================================================================
class UnsplitRescale(Rescale):
    r"""Unsplit and rescale measurements as

    .. math::

        m = \frac{y_+ - y_-}{\alpha},

    where :math:`y_-` and :math:`y_+` are the raw measurements and :math:`\alpha` represents their intensity.

    Args:
        :attr:`\alpha` (float): Measurement intensity :math:`\alpha`.

    Attributes:
        :attr:`\alpha` (float): Measurement intensity :math:`\alpha`.
    """

    def __init__(self, alpha=1.0):
        super().__init__(alpha)

    def forward(self, y: torch.tensor) -> torch.tensor:
        r"""Unsplit and rescale measurements

        .. math::
            m = \frac{y_+ - y_-}{\alpha},

        where :math:`y_-` :math:`y_+` are the raw measurements and :math:`\alpha` represents their intensity.

        Args:
            :attr:`y` (:class:`torch.tensor`): Raw measurements of shape :math:`(*, 2M)`

        Returns:
            :class:`torch.tensor`: Rescaled and unsplit measurements of shape :math:`(*, M)`.
        """
        y = Unsplit.forward(y, mode="sub")  # Unsplit
        y = super().forward(y)  # Rescale
        return y

    def sigma(self, y: torch.tensor) -> torch.tensor:
        r"""Rescale the variance

        .. math::

            \text{var}(m) = \frac{\text{var}(y_+) + \text{var}(y_-)}{\alpha^2}

        Args:
            v (:class:`torch.tensor`): Variance of :math:`y`  with shape :math:`(*, 2M)`.

        Returns:
            :class:`torch.tensor`: Variance of :math:`m` with shape :math:`(*, M)`.
        """
        y = Unsplit.forward(y, mode="add")
        y = super().sigma(y)
        return y


# =============================================================================
class RescaleEstim(nn.Module):
    r"""Rescale measurements as

    .. math::

        m = \frac{y}{\alpha},

    where :math:`y` is the measurement and :math:`\alpha` represents some
    gain/intensity that needs to be estimated from :math:`y`.

    .. important:
        This class is designed for measurements acquired without splitting. For
        split measurements, use :class:`UnsplitRescaleEstim`.

    Args:
        :attr:`meas_op` (spyrit.core.meas.Linear): Measurement operator used to get
        the measurements :math:`y`. It should not be a split measurement operator.

        **pinv_kwargs: Additional keyword arguments to pass to the pseudo-inverse
        computation.

    Attributes:
        :attr:`self.alpha` (:class:`torch.tensor`): Estimated gain/intensity.

        :attr:`self.meas_op` (spyrit.core.meas.Linear): Measurement operator used to
        simulate the measurements.

        :attr:`self.estim_mode` (str): Set to "pinv".

        :attr:`self.pinv_kwargs` (dict): Additional keyword arguments to pass to the
        pseudo-inverse initialization.

        :attr:`self.pinv` (spyrit.core.inverse.PseudoInverse): Pseudo-inverse
        operator.
    """

    # :attr:`estim_mode` (str, optional): The method to estimate the gain value. Can
    # be either "mean" or "pinv". Defaults to "mean".

    # The gain value :math:`\alpha` to divide the measurements by is estimated in
    # two different ways: either by taking the mean of the measurements or by
    # taking the maximum value of the pseudo-inverse of the measurements.

    # The first method, referred to as `mean`, defines :math:`alpha` as the
    # average measurement value of a single pixel. The second method, referred to
    # as `pinv`, computes the pseudo inverse of the measurements and defines
    # :math:`alpha` as the maximum over the last dimension of the resulting tensor.

    # The `mean` method is faster but does not guarantee that the gain value is
    # accurate. The `pinv` method is slower, but yields a more accurate estimate
    # of the gain value.

    def __init__(self, meas_op: meas.Linear, **pinv_kwargs):
        super().__init__()
        self.meas_op = meas_op
        self.alpha = None
        self.estim_mode = "pinv"
        self.pinv_kwargs = pinv_kwargs
        self.pinv = inverse.PseudoInverse(self.meas_op, **pinv_kwargs)

    def pinv_estim(self, y: torch.tensor) -> torch.tensor:
        r"""Estimate gain from pseudo-inverse.

        Args:
            :attr:`y` (:class:`torch.tensor`): The measurements with shape :math:`(*, M)`, where :math:`*` can be any number of dimensions and :math:`M` matches the number of measurements defined by :attr:`meas_op.M`.

        Returns:
            :class:`torch.tensor`: Estimated intensity :math:`\alpha` of shape :math:`(*, 1)`.
        """
        y_pinv = self.meas_op.vectorize(self.pinv(y))
        alpha = torch.max(y_pinv, -1, keepdim=True).values
        return alpha

    def estim_alpha(self, y: torch.tensor) -> torch.tensor:
        r"""Estimate gain from pseudo-inverse.

        Args:
            :attr:`y` (:class:`torch.tensor`): The measurements with shape :math:`(*, M)`, where :math:`*` can be any number of dimensions and :math:`M` matches the number of measurements defined by :attr:`meas_op.M`.

        Returns:
            :class:`torch.tensor`: Estimated intensity :math:`\alpha` of shape :math:`(*, 1)`.
        """
        return self.pinv_estim(y)

    def forward(self, y: torch.tensor) -> torch.tensor:
        r"""Rescale measurements as

        .. math::

            m = \frac{y}{\alpha},

        where :math:`y` is the measurement and :math:`\alpha` represents some
        gain/intensity that needs to be estimated from :math:`y`.

        Args:
            :attr:`y` (:class:`torch.tensor`):  The measurements with shape :math:`(*, M)`, where :math:`*` can be any number of dimensions and :math:`M` matches the number of measurements defined by :attr:`meas_op.M`.

        Returns:
            :class:`torch.tensor`: Rescaled measurements of shape :math:`(*, M)`.
        """
        self.alpha = self.estim_alpha(y)
        return y / self.alpha


# =============================================================================
class UnsplitRescaleEstim(RescaleEstim):
    r"""Unsplit and rescale measurements as

    .. math::

        m = \frac{y_+ - y_-}{\alpha},

    where :math:`y_-` and :math:`y_+` are the raw measurements and :math:`\alpha` represents a gain/intensity that needs to be estimated from :math:`y_-` and :math:`y_+`.

    .. important:
        This class is designed for measurements acquired with splitting. For
        unsplit measurements, use :class:`RescaleEstim`.

    Args:
        :attr:`meas_op` (spyrit.core.meas.LinearSplit): Measurement operator used to get
        the measurements :math:`y`. It should be a split measurement operator.

        :attr:`estim_mode` (str, optional): Method to estimate the gain. Can
        be either "mean" or "pinv". Defaults to "pinv".

        **pinv_kwargs: Additional keyword arguments to pass to the pseudo-inverse
        computation. Only used if `estim_mode` is "pinv".

    Attributes:
        :attr:`self.alpha` (:class:`torch.tensor`): Estimated gain/intensity.

        :attr:`self.meas_op` (spyrit.core.meas.LinearSplit): Measurement
        operator used to simulate the measurements.

        :attr:`self.estim_mode` (str): Method to estimate the gain value.

        :attr:`self.pinv_kwargs` (dict): Additional keyword arguments to pass to the
        pseudo-inverse initialization. Only used if `estim_mode` is "pinv".

        :attr:`self.pinv` (spyrit.core.inverse.PseudoInverse): Pseudo-inverse
        operator. Exists only if `estim_mode` is "pinv".
    """

    # The gain value :math:`alpha` to divide the measurements by is estimated in
    # two different ways: either by taking the mean of the measurements or by
    # taking the maximum value of the pseudo-inverse of the measurements.

    # The first method, referred to as `mean`, defines :math:`alpha` as the
    # average measurement value of a single pixel. The second method, referred to
    # as `pinv`, computes the pseudo inverse of the measurements and defines
    # :math:`alpha` as the maximum over the last dimension of the resulting tensor.

    # The `mean` method is faster but does not guarantee that the gain value is
    # accurate. The `pinv` method is slower, but yields a more accurate estimate
    # of the gain value.

    def __init__(self, meas_op, **pinv_kwargs):

        if not isinstance(meas_op, meas.LinearSplit):
            raise ValueError("meas_op should be a LinearSplit")
        super().__init__(meas_op, **pinv_kwargs)

    def mean_estim(self, y):
        r"""(Not tested yet) Estimate the gain from the mean of the raw measurements.

        .. important::
            This method is only to be called on measurements acquired with
            split measurement operators.

        Args:
            :attr:`y` (:class:`torch.tensor`):  The measurements with shape :math:`(*, 2M)`, where :math:`*` can be any number of dimensions and :math:`M` matches the number of measurements defined by :attr:`meas_op.M`.

        Returns:
            :class:`torch.tensor`: The estimated gain value of shape :math:`(*, 1)`.
        """
        if not isinstance(self.meas_op, meas.HadamSplit2d):
            Warning("Mean estimation is only exact for HadamSplit2d operators")

        y = torch.sum(y, -1, keepdim=True)
        # take the matrix *A* because the measurements ARE split
        divisor = self.meas_op.A.sum(dim=-1, keepdim=True).expand(y.shape)
        alpha = torch.div(y, divisor)
        return alpha

    def estim_alpha(self, y: torch.tensor) -> torch.tensor:
        r"""Estimate the gain from the raw measurements.

        This calls either :meth:`pinv_estim` or :meth:`mean_estim` depending on :attr:`self.estim_mode`.

        Args:
            :attr:`y` (:class:`torch.tensor`):  The measurements with shape :math:`(*, 2M)`, where :math:`*` can be any number of dimensions and :math:`M` matches the number of measurements defined by :attr:`meas_op.M`.

        Returns:
            :class:`torch.tensor`: The estimated gain value of shape :math:`(*, 1)`.
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
        r"""Unsplit and rescale raw measurements

        .. math::

            m = \frac{y_+ - y_-}{\alpha}

        where :math:`y_-` :math:`y_+` are the raw measurements and :math:`\alpha` is the intensity estimated by calling :meth:`estim_alpha`.

        Args:
            :attr:`y` (:class:`torch.tensor`):  The measurements with shape :math:`(*, 2M)`, where :math:`*` can be any number of dimensions and :math:`M` matches the number of measurements defined by :attr:`meas_op.M`.

        Returns:
            :class:`torch.tensor`: Rescaled unsplit measurements with shape :math:`(*, M)`.
        """
        y = Unsplit.forward(y, mode="sub")
        return super().forward(y)

    def sigma(self, y: torch.tensor) -> torch.tensor:
        r"""Estimate the variance of raw split measurements as

        .. math::

            \sigma^2 = \frac{y_+ + y_-}{\alpha^2}

        where :math:`y_-` :math:`y_+` are the raw measurements.

        .. important::
            This function takes the raw measurments as input and must be called before :meth:`forward()`.

        .. note::

            alpha could be saved to avoid to recomputing it.

        Args:
            y (:class:`torch.tensor`): batch of measurements of shape :math:`(*, 2M)`.

        Returns:
            torch.tensor: Estimated variance with shape :math:`(*, M)`.
        """
        alpha = self.estim_alpha(Unsplit.forward(y, mode="sub"))
        v = Unsplit.forward(y, mode="add")
        return v / alpha**2


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
            x (:class:`torch.tensor`): The input tensor to apply the affine transform to.
            Has arbitrary shape.

        Returns:
            :class:`torch.tensor`: The input tensor with the affine transform applied.
            It has the same shape as the input tensor.
        """
        a, b = self.input_range
        c, d = self.output_range
        return (x - a) * (d - c) / (b - a) + c

    def backward(self, x: torch.tensor) -> torch.tensor:
        r"""Applies the inverse affine transform to the input tensor.

        Args:
            x (:class:`torch.tensor`): The input tensor to apply the inverse affine
            transform to. Has arbitrary shape.

        Returns:
            :class:`torch.tensor`: The input tensor with the inverse affine transform
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
