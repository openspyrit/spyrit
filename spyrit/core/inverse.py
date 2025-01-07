"""
Inverse methods for inverse problems.
"""

from typing import Union, OrderedDict

import torch
import torch.nn as nn

import spyrit.core.meas as meas
import spyrit.core.torch as spytorch


# =============================================================================


def regularized_pinv(
    tensor: torch.tensor, regularization: str, *args, **kwargs
) -> torch.tensor:
    """Returns a regularized pseudo-inverse of a tensor.

    The regularizations supported are:

        - "rcond": Uses the function :func:`torch.linalg.pinv`. Additional
            arguments can be passed to this function through the `args` and
            `kwargs` parameters, such as the `rcond` parameter.

        - "L2": Uses the L2 regularization method. The regularization parameter
            `eta` must be passed as a keyword argument. It controls the amount
            of regularization applied to the pseudo-inverse.

        - "H1": Uses the H1 regularization method. The regularization parameters
            `eta` and `img_shape` must be passed as keyword arguments. The
            `eta` parameter controls the amount of regularization applied to the
            pseudo-inverse, and the `img_shape` parameter is the shape of the
            image to which the pseudo-inverse will be applied. This is used to
            compute the finite difference operator.

    .. note::
        The H1 regularization method is only implemented for application to 2D
        images (i.e., `image_shape` must be 2D).

    Args:
        tensor (torch.tensor): input tensor to compute the pseudo-inverse. Must
        be 2D.

        regularization (str): Regularization method to use. Supported methods
        are "rcond", "L2", and "H1".

        *args: Additional arguments to pass to the regularization method.

        **kwargs: Additional keyword arguments to pass to the regularization
        method. Must include the regularization parameter `eta` when using the
        "L2" and "H1" regularization methods, and the image shape `img_shape`
        when using the "H1" regularization method.

    Raises:
        NotImplementedError: If the regularization method is not supported.

    Returns:
        torch.tensor: The regularized pseudo-inverse of the input tensor.
    """

    if regularization == "rcond":
        pinv = torch.linalg.pinv(tensor, *args, **kwargs)

    elif regularization == "L2":
        eta = kwargs.get("eta")
        if tensor.shape[0] >= tensor.shape[1]:
            pinv = (
                torch.linalg.inv(
                    tensor.T @ tensor
                    + eta * torch.eye(tensor.shape[1], device=tensor.device)
                )
                @ tensor.T
            )
        else:
            pinv = tensor.T @ torch.linalg.inv(
                tensor @ tensor.T
                + eta * torch.eye(tensor.shape[0], device=tensor.device)
            )

    elif regularization == "H1":
        eta = kwargs.get("eta")
        img_shape = kwargs.get("img_shape")
        Dx, Dy = spytorch.neumann_boundary(img_shape)
        D2 = (Dx.T @ Dx + Dy.T @ Dy).to(tensor.device)
        pinv = torch.linalg.inv(tensor.T @ tensor + eta * D2) @ tensor.T

    else:
        raise NotImplementedError(
            f"Regularization method {regularization} not implemented. Currently supported methods are 'rcond', 'L2', and 'H1'."
        )

    return pinv


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
        *args,
        **kwargs,
    ) -> None:

        super().__init__()
        self.meas_op = meas_op
        self.store_pinv = store_pinv

        if self.store_pinv:
            self.pinv = regularized_pinv(
                self.meas_op.matrix_to_inverse, regularization, *args, **kwargs
            )

    def forward(self, y: torch.tensor, *args, **kwargs) -> torch.tensor:
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

        return y.squeeze(-1)


# =============================================================================
class RegularizedPinv(PseudoInverse):
    """ """

    def __init__(self, meas_op, store_pinv=False, *args, **kwargs):
        super().__init__(meas_op, store_pinv, *args, **kwargs)
