"""
Measurement operators, static and dynamic.

There are six classes contained in this module, each representing a different
type of measurement operator. Three of them are static, i.e. they are used to
simulate measurements of still images, and three are dynamic, i.e. they are used
to simulate measurements of moving objects, represented as a sequence of images.
 The inheritance tree is as follows::

      Linear          DynamicLinear
        |                   |
        V                   V
    LinearSplit     DynamicLinearSplit
        |                   |
        V                   V
    HadamSplit      DynamicHadamSplit

"""

import warnings

# import memory_profiler as mprof

import math
import torch
import torch.nn as nn

from spyrit.core.warp import DeformationField
import spyrit.core.torch as spytorch


# =============================================================================
# BASE CLASS - FOR INHERITANCE ONLY (INTERAL USE)
# =============================================================================
class _Base(nn.Module):

    def __init__(
        self,
        H_static: torch.tensor,
        Ord: torch.tensor = None,
        meas_shape: tuple = None,
    ) -> None:
        super().__init__()

        # store meas_shape and check it is correct
        if meas_shape is None:
            self._meas_shape = (
                int(math.sqrt(H_static.shape[1])),
                int(math.sqrt(H_static.shape[1])),
            )
        else:
            self._meas_shape = meas_shape
        if self._meas_shape[0] * self._meas_shape[1] != H_static.shape[1]:
            raise ValueError(
                f"The number of pixels in the measurement matrix H "
                + f"({H_static.shape[1]}) does not match the measurement shape "
                + f"{self._meas_shape}."
            )
        self._img_shape = self._meas_shape

        if Ord is not None:
            H_static, ind = spytorch.sort_by_significance(
                H_static, Ord, "rows", False, get_indices=True
            )
        else:
            ind = torch.arange(H_static.shape[0])
            Ord = torch.arange(H_static.shape[0], 0, -1)

        # convert H to float32 if it is not float64
        if H_static.dtype != torch.float64:
            H_static = H_static.to(torch.float32)

        # attributes for internal use
        self._param_H_static = nn.Parameter(H_static, requires_grad=False)
        # need to store M because H_static may be cropped (see HadamSplit)
        self._M = H_static.shape[0]

        self._param_Ord = nn.Parameter(Ord.to(torch.float32), requires_grad=False)
        self._indices = ind.to(torch.int32)
        self._device_tracker = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    ### PROPERTIES ------
    @property
    def M(self) -> int:
        """Number of measurements (first dimension of H)"""
        return self._M

    @property
    def N(self) -> int:
        """Number of pixels in the image"""
        return self.img_h * self.img_w

    @property
    def h(self) -> int:
        """Measurement pattern height"""
        return self.meas_shape[0]

    @property
    def w(self) -> int:
        """Measurement pattern width"""
        return self.meas_shape[1]

    @property
    def meas_shape(self) -> tuple:
        """Shape of the measurement patterns (height, width). Note that
        `height * width = N`."""
        return self._meas_shape

    @property
    def img_shape(self) -> tuple:
        """Shape of the image (height, width)."""
        return self._img_shape

    @property
    def img_h(self) -> int:
        """Height of the image"""
        return self._img_shape[0]

    @property
    def img_w(self) -> int:
        """Width of the image"""
        return self._img_shape[1]

    @property
    def indices(self) -> torch.tensor:
        """Indices used to sort the rows of H"""
        return self._indices

    @property
    def Ord(self) -> torch.tensor:
        """Order matrix used to sort the rows of H"""
        return self._param_Ord.data

    @Ord.setter
    def Ord(self, value: torch.tensor) -> None:
        self._set_Ord(value)

    @property
    def H_static(self) -> torch.tensor:
        """Static measurement matrix H."""
        return self._param_H_static.data[: self.M, :]

    @property
    def P(self) -> torch.tensor:
        """Measurement matrix P with positive and negative components. Used in
        classes *Split and *HadamSplit."""
        return self._param_P.data[: 2 * self.M, :]

    @property
    def device(self) -> torch.device:
        return self._device_tracker.device

    ### -------------------

    def pinv(
        self, x: torch.tensor, reg: str = "rcond", eta: float = 1e-3, diff=False
    ) -> torch.tensor:
        r"""Computes the pseudo inverse solution :math:`y = H^\dagger x`.

        This method will compute the pseudo inverse solution using the
        measurement matrix pseudo-inverse :math:`H^\dagger` if it has been
        calculated and stored in the attribute :attr:`H_pinv`. If not, the
        pseudo inverse will be not be explicitly computed and the torch
        function :func:`torch.linalg.lstsq` will be used to solve the linear
        system.

        Args:
            :attr:`x` (torch.tensor): batch of measurement vectors. If x has
            more than 1 dimension, the pseudo inverse is applied to each
            image in the batch.

            :attr:`reg` (str, optional): Regularization method to use.
            Available options are 'rcond', 'L2' and 'H1'. 'rcond' uses the
            :attr:`rcond` parameter found in :func:`torch.linalg.lstsq`.
            This parameter must be specified if the pseudo inverse has not been
            computed. Defaults to None.

            :attr:`eta` (float, optional): Regularization parameter. Only
            relevant when :attr:`reg` is specified. Defaults to None.

            :attr:`diff` (bool, optional): Use only if a split operator is used
            and if the pseudo inverse has not been computed. Whether to use the
            difference of positive and negative patterns.
            The difference is applied to the measurements and to the dynamic
            measurement matrix. Defaults to False.

        Shape:
            :math:`x`: :math:`(*, M)` where * denotes the batch size and `M`
            the number of measurements.

            Output: :math:`(*, N)` where * denotes the batch size and `N`
            the number of pixels in the image.

        Example:
            >>> H = torch.randn([400, 1600])
            >>> meas_op = Linear(H, True)
            >>> x = torch.randn([10, 400])
            >>> y = meas_op.pinv(x)
            >>> print(y.shape)
            torch.Size([10, 1600])
        """
        # have we calculated the pseudo inverse ?
        if hasattr(self, "H_pinv"):
            ans = self._pinv_mult(x)

        else:

            if isinstance(self, Linear):
                H_to_inv = self.H_static
            elif type(self) == DynamicLinear:
                H_to_inv = self.H_dyn
            elif isinstance(self, DynamicLinearSplit):
                if diff:
                    x = x[..., ::2] - x[..., 1::2]
                    H_to_inv = self.H_dyn[::2, :] - self.H_dyn[1::2, :]
                else:
                    H_to_inv = self.H_dyn
            else:
                raise NotImplementedError(
                    "It seems you have instanciated a _Base element. This class "
                    + "Should not be called on its own."
                )

            # cast to dtype of x
            H_to_inv = H_to_inv.to(x.dtype)
            # devices are supposed to be the same, don't bother checking

            driver = "gelsd"
            if H_to_inv.device != torch.device("cpu"):
                H_to_inv = H_to_inv.cpu()
                x = x.cpu()
                # driver = 'gels'

            if reg == "rcond":
                A = H_to_inv.expand(*x.shape[:-1], *H_to_inv.shape)  # shape (*, M, N)
                B = x.unsqueeze(-1).to(A.dtype)  # shape (*, M, 1)
                ans = torch.linalg.lstsq(A, B, rcond=eta, driver=driver)
                ans = ans.solution.to(x.dtype).squeeze(-1)  # shape (*, N)

            elif reg == "L2":
                A = torch.matmul(H_to_inv.mT, H_to_inv) + eta * torch.eye(
                    H_to_inv.shape[1]
                )
                A = A.expand(*x.shape[:-1], *A.shape)
                B = torch.matmul(x.to(H_to_inv.dtype), H_to_inv)
                ans = torch.linalg.solve(A, B).to(x.dtype)

            elif reg == "H1":
                Dx, Dy = spytorch.neumann_boundary(self.img_shape)
                D2 = Dx.T @ Dx + Dy.T @ Dy

                A = torch.matmul(H_to_inv.mT, H_to_inv) + eta * D2
                A = A.expand(*x.shape[:-1], *A.shape)
                B = torch.matmul(x.to(H_to_inv.dtype), H_to_inv)
                ans = torch.linalg.solve(A, B).to(x.dtype)

            elif reg is None:
                raise ValueError(
                    "Regularization method not specified. Please compute "
                    + "the dynamic pseudo-inverse or specify a regularization "
                    + "method."
                )
            else:
                raise NotImplementedError(
                    f"Regularization method ({reg}) not implemented. Please "
                    + "use 'rcond', 'L2' or 'H1'."
                )

        # if we used bicubic b spline, convolve with the kernel
        if hasattr(self, "recon_mode") and self.recon_mode == "bicubic":
            kernel = torch.tensor([[1, 4, 1], [4, 16, 4], [1, 4, 1]]) / 36
            conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            conv.weight.data = kernel.reshape(1, 1, 3, 3).to(ans.dtype)

            ans = (
                conv(ans.reshape(-1, 1, self.img_h, self.img_w))
                .reshape(-1, self.img_h * self.img_w)
                .data
            )

        return ans.reshape(*ans.shape[:-1], *self.img_shape)

    def reindex(
        self, x: torch.tensor, axis: str = "rows", inverse_permutation: bool = False
    ) -> torch.tensor:
        """Sorts a tensor along a specified axis using the indices tensor. The
        indices tensor is contained in the attribute :attr:`self.indices`.

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
        return spytorch.reindex(x, self.indices.to(x.device), axis, inverse_permutation)

    def unvectorize(self, input: torch.tensor) -> torch.tensor:
        """Reshape a vectorized tensor to the measurement shape (heigth, width).

        Input:
            input (torch.tensor): A tensor of shape (*, N) where * denotes the
            batch size and N the total number of pixels in the image.

        Output:
            torch.tensor: A tensor of shape (*, h, w) where * denotes the batch
            size and h, w the height and width of the measurement patterns.
        """
        return input.reshape(*input.shape[:-1], *self.meas_shape)

    def vectorize(self, input: torch.tensor) -> torch.tensor:
        """Vectorize an image-shaped tensor.

        Input:
            input (torch.tensor): A tensor of shape (*, h, w) where * denotes the
            batch size and h, w the height and width of the measurement patterns.

        Output:
            torch.tensor: A tensor of shape (*, N) where * denotes the batch size
            and N the total number of pixels in the image.
        """
        return input.reshape(*input.shape[:-2], self.N)

    def _static_forward_with_op(
        self, x: torch.tensor, op: torch.tensor
    ) -> torch.tensor:
        return torch.einsum("mhw,...hw->...m", self.unvectorize(op).to(x.dtype), x)

    # @mprof.profile
    def _dynamic_forward_with_op(
        self, x: torch.tensor, op: torch.tensor
    ) -> torch.tensor:
        x = spytorch.center_crop(x, self.meas_shape)
        return torch.einsum("thw,...tchw->...ct", self.unvectorize(op).to(x.dtype), x)

    def _pinv_mult(self, y: torch.tensor) -> torch.tensor:
        """Uses the pre-calculated pseudo inverse to compute the solution.
        We assume that the pseudo inverse has been calculated and stored in the
        attribute :attr:`H_pinv`.
        """
        A = self.H_pinv.expand(*y.shape[:-1], *self.H_pinv.shape)
        B = y.unsqueeze(-1).to(A.dtype)
        ans = torch.matmul(A, B).to(y.dtype).squeeze(-1)
        return ans

    def _set_Ord(self, Ord: torch.tensor) -> None:
        """Set the order matrix used to sort the rows of H. This is used in
        the Ord.setter property. This method is defined for simplified
        inheritance. For internal use only."""
        # unsort the rows of H
        H_natural = self.reindex(self.H_static, "rows", inverse_permutation=True)
        # resort the rows of H ; store indices in self._indices
        H_resorted, self._indices = spytorch.sort_by_significance(
            H_natural, Ord, "rows", False, get_indices=True
        )
        # update values of H, Ord
        self._param_H_static.data = H_resorted.to(self.device)
        self._param_Ord.data = Ord.to(self.device)

    def _set_P(self, H_static: torch.tensor) -> None:
        """Set the positive and negative components of the measurement matrix
        P from the static measurement matrix H_static. For internal use only.
        Used in classes *Split and *HadamSplit."""
        H_pos = nn.functional.relu(H_static)
        H_neg = nn.functional.relu(-H_static)
        self._param_P = nn.Parameter(
            torch.cat([H_pos, H_neg], 1).reshape(
                2 * H_static.shape[0], H_static.shape[1]
            ),
            requires_grad=False,
        )

    def _build_pinv(self, tensor: torch.tensor, reg: str, eta: float) -> torch.tensor:

        if reg == "rcond":
            pinv = torch.linalg.pinv(tensor, atol=eta)

        elif reg == "L2":
            if tensor.shape[0] >= tensor.shape[1]:
                pinv = (
                    torch.linalg.inv(
                        tensor.T @ tensor + eta * torch.eye(tensor.shape[1])
                    )
                    @ tensor.T
                )
            else:
                pinv = tensor.T @ torch.linalg.inv(
                    tensor @ tensor.T + eta * torch.eye(tensor.shape[0])
                )

        elif reg == "H1":
            # Boundary condition matrices
            Dx, Dy = spytorch.neumann_boundary(self.img_shape)
            D2 = (Dx.T @ Dx + Dy.T @ Dy).to(tensor.device)
            pinv = torch.linalg.inv(tensor.T @ tensor + eta * D2) @ tensor.T

        else:
            raise NotImplementedError(
                f"Regularization method '{reg}' is not implemented. Please "
                + "choose either 'rcond', 'L2' or 'H1'."
            )
        return pinv.to(self.device)

    def _attributeslist(self) -> list:
        _list = [
            ("M", "self.M", _Base),
            ("N", "self.N", _Base),
            ("H.shape", "self.H_static.shape", _Base),
            ("meas_shape", "self._meas_shape", _Base),
            ("H_dyn", "hasattr(self, 'H_dyn')", DynamicLinear),
            ("img_shape", "self.img_shape", DynamicLinear),
            ("H_pinv", "hasattr(self, 'H_pinv')", _Base),
            ("P.shape", "self.P.shape", (LinearSplit, DynamicLinearSplit)),
        ]
        return _list

    def __repr__(self) -> str:
        s_begin = f"{self.__class__.__name__}(\n  "
        s_fill = "\n  ".join(
            [
                f"({k}): {eval(v)}"
                for k, v, t in self._attributeslist()
                if isinstance(self, t)
            ]
        )
        s_end = "\n)"
        return s_begin + s_fill + s_end


# =============================================================================
class Linear(_Base):
    # =========================================================================
    r"""
    Simulates linear measurements :math:`y = Hx`.

    Computes linear measurements from incoming images: :math:`y = Hx`,
    where :math:`H` is a given linear operator (matrix) and :math:`x` is a
    vectorized image or batch of images.

    The class is constructed from a matrix :math:`H` of shape :math:`(M,N)`,
    where :math:`N` represents the number of pixels in the image and
    :math:`M` the number of measurements.

    Args:
        :attr:`H` (:class:`torch.tensor`): measurement matrix (linear operator)
        with shape :math:`(M, N)`. Only real values are supported.

        :attr:`pinv` (bool): Whether to store the pseudo inverse of the
        measurement matrix :math:`H`. If `True`, the pseudo inverse is
        initialized as :math:`H^\dagger` and stored in the attribute
        :attr:`H_pinv`. It is alwats possible to compute and store the pseudo
        inverse later using the method :meth:`build_H_pinv`. Defaults to `False`.

        :attr:`rtol` (float, optional): Cutoff for small singular values (see
        :mod:`torch.linalg.pinv`). Only relevant when :attr:`pinv` is `True`.

        :attr:`Ord` (torch.tensor, optional): Order matrix used to reorder the
        rows of the measurement matrix :math:`H`. The first new row of :math:`H`
        will correspond to the highest value in :math:`Ord`. Must contain
        :math:`M` values. If some values repeat, the order is kept. Defaults to
        None.

        :attr:`meas_shape` (tuple, optional): Shape of the measurement patterns.
        Must be a tuple of two integers representing the height and width of the
        patterns. If not specified, the shape is suppposed to be a square image.
        If not, an error is raised. Defaults to None.

    Attributes:
        :attr:`H` (torch.tensor): The learnable measurement matrix of shape
        :math:`(M, N)` initialized as :math:`H`.

        :attr:`H_static` (torch.tensor): alias for :attr:`H`.

        :attr:`H_pinv` (torch.tensor, optional): The learnable pseudo inverse
        measurement matrix :math:`H^\dagger` of shape :math:`(N, M)`.

        :attr:`M` (int): Number of measurements performed by the linear operator.

        :attr:`N` (int): Number of pixels in the image.

        :attr:`h` (int): Measurement pattern height.

        :attr:`w` (int): Measurement pattern width.

        :attr:`meas_shape` (tuple): Shape of the measurement patterns
        (height, width). Is equal to `(self.h, self.w)`.

        :attr:`indices` (torch.tensor): Indices used to sort the rows of H.	It
        is used by the method :meth:`reindex()`.

        :attr:`Ord` (torch.tensor): Order matrix used to sort the rows of H. It
        is used by :func:`~spyrit.core.torch.sort_by_significance()`.

    .. note::
        If you know the pseudo inverse of :math:`H` and want to store it, it is
        best to initialize the class with :attr:`pinv` set to `False` and then
        call :meth:`build_H_pinv` to store the pseudo inverse.

    Example 1:
        >>> H = torch.rand([400, 1600])
        >>> meas_op = Linear(H, pinv=False)
        >>> print(meas_op)
        Linear(
          (M): 400
          (N): 1600
          (H.shape): torch.Size([400, 1600])
          (meas_shape): (40, 40)
          (H_pinv): False
        )

    Example 2:
        >>> H = torch.rand([400, 1600])
        >>> meas_op = Linear(H, True)
        >>> print(meas_op)
        Linear(
          (M): 400
          (N): 1600
          (H.shape): torch.Size([400, 1600])
          (meas_shape): (40, 40)
          (H_pinv): True
        )
    """

    def __init__(
        self,
        H: torch.tensor,
        pinv: bool = False,
        rtol: float = None,
        Ord: torch.tensor = None,
        meas_shape: tuple = None,  # (height, width)
    ):
        super().__init__(H, Ord, meas_shape)
        if pinv:
            self.build_H_pinv(reg="rcond", eta=rtol)

    @property
    def H(self) -> torch.tensor:
        return self.H_static

    @property
    def H_pinv(self) -> torch.tensor:
        return self._param_H_static_pinv.data

    @H_pinv.setter
    def H_pinv(self, value: torch.tensor) -> None:
        self._param_H_static_pinv = nn.Parameter(
            value.to(torch.float64), requires_grad=False
        )

    @H_pinv.deleter
    def H_pinv(self) -> None:
        del self._param_H_static_pinv

    # Deprecated method - included for backwards compatibility but to remove
    def get_H(self) -> torch.tensor:
        """Deprecated method. Use the attribute self.H instead."""
        warnings.warn(
            "The method get_H() is deprecated and will be removed in a future "
            + "version. Please use the attribute self.H instead."
        )
        return self.H

    def build_H_pinv(self, reg: str = "rcond", eta: float = 1e-3) -> None:
        """Used to set the pseudo inverse of the measurement matrix :math:`H`
        using `torch.linalg.pinv`. The result is stored in the attribute
        :attr:`H_pinv`.

        Args:
            reg (str, optional): Regularization method to use. Available options
            are 'rcond', 'L2' and 'H1'. 'rcond' uses the :attr:`rcond` parameter
            found in :func:`torch.linalg.lstsq`. This parameter must be specified
            if the pseudo inverse has not been computed. Defaults to None.

            eta (float, optional): Regularization parameter (cutoff for small
            singular values, see :mod:`torch.linalg.pinv`). Defaults to None,
            in which case the default value of :mod:`torch.linalg.pinv` is used.

        Returns:
            None. The pseudo inverse is stored in the attribute :attr:`H_pinv`.
        """
        self.H_pinv = self._build_pinv(self.H_static, reg, eta)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Applies linear transform to incoming images: :math:`y = Hx`.

        This is equivalent to computing :math:`x \cdot H^T`. The input images
        must be unvectorized.

        Args:
            :math:`x` (torch.tensor): Batch of images of shape :math:`(*, h, w)`.
            `*` can have any number of dimensions, for instance `(b, c)` where
            `b` is the batch size and `c` the number of channels. `h` and `w`
            are the height and width of the images.

        Shape:
            :math:`x`: :math:`(*, h, w)` where * denotes the batch size and `N`
            the total number of pixels in the image.

            Output: :math:`(*, M)` where * denotes any number of dimensions
            and `M` the number of measurements.

        Example:
            >>> H = torch.randn([400, 1600])
            >>> meas_op = Linear(H)
            >>> x = torch.randn([10, 40, 40])
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([10, 400])
        """
        # left multiplication with transpose is equivalent to right mult
        # return x @ self.H.T.to(x.dtype).to(x.device)
        return self._static_forward_with_op(x, self.H)

    def adjoint(self, y: torch.tensor) -> torch.tensor:
        r"""Applies adjoint transform to incoming measurements :math:`x = H^{T}y`

        This brings back the measurements in the image domain, but is not
        equivalent to the inverse of the forward operator.

        Args:
            :math:`y` (torch.tensor): batch of measurement vectors of shape
            :math:`(*, M)` where * denotes any number of dimensions (e.g.
            `(b,c)` where `b` is the batch size and `c` the number of channels)
            and `M` the number of measurements.

        Output:
            torch.tensor: The adjoint of the input measurements, which are
            in the image domain. It has shape :math:`(*, h, w)` where * denotes
            any number of dimensions and `h`, `w` the height and width of the
            images.

        Shape:
            :math:`y`: :math:`(*, M)`

            Output: :math:`(*, h, w)`

        Example:
            >>> H = torch.randn([400, 1600])
            >>> meas_op = Linear(H)
            >>> y = torch.randn([10, 400]
            >>> x = meas_op.adjoint(y)
            >>> print(x.shape)
            torch.Size([10, 40, 40])
        """
        # return x @ self.H.to(x.dtype).to(x.device)
        return torch.einsum("mhw,...m->...hw", self.unvectorize(self.H).to(y.dtype), y)

    def _set_Ord(self, Ord: torch.tensor) -> None:
        """Set the order matrix used to sort the rows of H."""
        super()._set_Ord(Ord)
        # delete self.H_pinv (self._param_H_static_pinv)
        try:
            del self._param_H_static_pinv
            warnings.warn(
                "The pseudo-inverse H_pinv has been deleted. Please call "
                + "build_H_pinv() to recompute it."
            )
        except AttributeError:
            pass


# =============================================================================
class LinearSplit(Linear):
    # =========================================================================
    r"""
    Simulates splitted measurements :math:`y = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}x`.

    Computes linear measurements from incoming images: :math:`y = Px`,
    where :math:`P` is a linear operator (matrix) and :math:`x` is a
    vectorized image or batch of vectorized images.

    The matrix :math:`P` contains only positive values and is obtained by
    splitting a measurement matrix :math:`H` such that
    :math:`P` has a shape of :math:`(2M, N)` and `P[0::2, :] = H_{+}` and
    `P[1::2, :] = H_{-}`, where :math:`H_{+} = \max(0,H)` and
    :math:`H_{-} = \max(0,-H)`.

    The class is constructed from the :math:`M` by :math:`N` matrix :math:`H`,
    where :math:`N` represents the number of pixels in the image and
    :math:`M` the number of measurements. Therefore, the shape of :math:`P` is
    :math:`(2M, N)`.

    Args:
        :attr:`H` (:class:`torch.tensor`): measurement matrix (linear operator)
        with shape :math:`(M, N)`. Only real values are supported.

        :attr:`pinv` (bool): Whether to store the pseudo inverse of the
        measurement matrix :math:`H`. If `True`, the pseudo inverse is
        initialized as :math:`H^\dagger` and stored in the attribute
        :attr:`H_pinv`. It is alwats possible to compute and store the pseudo
        inverse later using the method :meth:`build_H_pinv`. Defaults to `False`.

        :attr:`rtol` (float, optional): Cutoff for small singular values (see
        :mod:`torch.linalg.pinv`). Only relevant when :attr:`pinv` is `True`.

        :attr:`Ord` (torch.tensor, optional): Order matrix used to reorder the
        rows of the measurement matrix :math:`H`. The first new row of :math:`H`
        will correspond to the highest value in :math:`Ord`. Must contain
        :math:`M` values. If some values repeat, the order is kept. Defaults to
        None.

        :attr:`meas_shape` (tuple, optional): Shape of the measurement patterns.
        Must be a tuple of two integers representing the height and width of the
        patterns. If not specified, the shape is suppposed to be a square image.
        If not, an error is raised. Defaults to None.

    Attributes:
        :attr:`H` (torch.tensor): The learnable measurement matrix of shape
        :math:`(M, N)` initialized as :math:`H`.

        :attr:`H_static` (torch.tensor): alias for :attr:`H`.

        :attr:`P` (torch.tensor): The splitted measurement matrix of shape
        :math:`(2M, N)`.

        :attr:`H_pinv` (torch.tensor, optional): The learnable pseudo inverse
        measurement matrix :math:`H^\dagger` of shape :math:`(N, M)`.

        :attr:`M` (int): Number of measurements performed by the linear operator.

        :attr:`N` (int): Number of pixels in the image.

        :attr:`h` (int): Measurement pattern height.

        :attr:`w` (int): Measurement pattern width.

        :attr:`meas_shape` (tuple): Shape of the measurement patterns
        (height, width). Is equal to `(self.h, self.w)`.

        :attr:`indices` (torch.tensor): Indices used to sort the rows of H.	It
        is used by the method :meth:`reindex()`.

        :attr:`Ord` (torch.tensor): Order matrix used to sort the rows of H. It
        is used by :func:`~spyrit.core.torch.sort_by_significance()`.

    .. note::
        If you know the pseudo inverse of :math:`H` and want to store it, it is
        best to initialize the class with :attr:`pinv` set to `False` and then
        call :meth:`build_H_pinv` to store the pseudo inverse.

    .. note::
        :math:`H = H_{+} - H_{-}`

    Example:
        >>> H = torch.randn(400, 1600)
        >>> meas_op = LinearSplit(H, False)
        >>> print(meas_op)
        LinearSplit(
          (M): 400
          (N): 1600
          (H.shape): torch.Size([400, 1600])
          (meas_shape): (40, 40)
          (H_pinv): False
          (P.shape): torch.Size([800, 1600])
        )
    """

    def __init__(
        self,
        H: torch.tensor,
        pinv: bool = False,
        rtol: float = None,
        Ord: torch.tensor = None,
        meas_shape: tuple = None,  # (height, width)
    ):
        super().__init__(H, pinv, rtol, Ord, meas_shape)
        self._set_P(self.H_static)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Applies linear transform to incoming images: :math:`y = Px`.

        This is equivalent to computing :math:`x \cdot P^T`. The input images
        must be unvectorized. The matrix :math:`P` is obtained by splitting
        the measurement matrix :math:`H` such that :math:`P` has a shape of
        :math:`(2M, N)` and `P[0::2, :] = H_{+}` and `P[1::2, :] = H_{-}`,
        where :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

        .. warning::
            This method uses the splitted measurement matrix :math:`P` to compute
            the linear measurements from incoming images. If you want to apply
            the operator :math:`H` directly, use the method :meth:`forward_H`.

        Args:
            :math:`x` (torch.tensor): Batch of images of shape :math:`(*, h, w)`.
            `*` can have any number of dimensions, for instance `(b, c)` where
            `b` is the batch size and `c` the number of channels. `h` and `w`
            are the height and width of the images.

        Output:
            torch.tensor: The linear measurements of the input images. It has
            shape :math:`(*, 2M)` where * denotes any number of dimensions and
            `M` the number of measurements as defined by the parameter :attr:`M`,
            which is equal to the number of rows in the measurement matrix :math:`H`
            defined at initialization.

        Shape:
            :math:`x`: :math:`(*, N)` where * denotes the batch size and `N`
            the total number of pixels in the image.

            Output: :math:`(*, 2M)` where * denotes the batch size and `M`
            the number of measurements as defined by the parameter :attr:`M`,
            which is equal to the number of rows in the measurement matrix :math:`H`
            defined at initialization.

        Example:
            >>> H = torch.randn(400, 1600)
            >>> meas_op = LinearSplit(H)
            >>> x = torch.randn(10, 40, 40)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([10, 800])
        """
        # return x @ self.P.T.to(x.dtype)
        return self._static_forward_with_op(x, self.P)

    def forward_H(self, x: torch.tensor) -> torch.tensor:
        r"""Applies linear transform to incoming images: :math:`m = Hx`.

        This is equivalent to computing :math:`x \cdot H^T`. The input images
        must be unvectorized.

        .. warning::
            This method uses the measurement matrix :math:`H` to compute the linear
            measurements from incoming images. If you want to apply the splitted
            operator :math:`P`, use the method :meth:`forward`.

        Args:
            :attr:`x` (torch.tensor): Batch of images of shape :math:`(*, h, w)`.
            `*` can have any number of dimensions, for instance `(b, c)` where
            `b` is the batch size and `c` the number of channels. `h` and `w`
            are the height and width of the images.

        Output:
            torch.tensor: The linear measurements of the input images. It has
            shape :math:`(*, M)` where * denotes any number of dimensions and
            `M` the number of measurements.

        Shape:
            :attr:`x`: :math:`(*, h, w)` where * denotes the batch size and `h`
            and `w` are the height and width of the images.

            Output: :math:`(*, M)` where * denotes the batch size and `M`
            the number of measurements.

        Example:
            >>> H = torch.randn(400, 1600)
            >>> meas_op = LinearSplit(H)
            >>> x = torch.randn(10, 40, 40)
            >>> y = meas_op.forward_H(x)
            >>> print(y.shape)
            torch.Size([10, 400])
        """
        # call Linear.forward() method
        return super().forward(x)

    def _set_Ord(self, Ord):
        super()._set_Ord(Ord)
        self._set_P(self.H_static)


# =============================================================================
class HadamSplit(LinearSplit):
    # =========================================================================
    r"""
    Simulates splitted measurements :math:`y = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}x`
    with :math:`H` a Hadamard matrix.

    Computes linear measurements from incoming images: :math:`y = Px`,
    where :math:`P` is a linear operator (matrix) and :math:`x` is a
    vectorized image or batch of vectorized images.

    The matrix :math:`P` contains only positive values and is obtained by
    splitting a Hadamard-based matrix :math:`H` such that
    :math:`P` has a shape of :math:`(2M, N)` and `P[0::2, :] = H_{+}` and
    `P[1::2, :] = H_{-}`, where :math:`H_{+} = \max(0,H)` and
    :math:`H_{-} = \max(0,-H)`.

    :math:`H` is obtained by selecting a re-ordered subsample of :math:`M` rows
    of a "full" Hadamard matrix :math:`F` with shape :math:`(N^2, N^2)`.
    :math:`N` must be a power of 2.

    Args:
        :attr:`M` (int): Number of measurements. It determines the size of the
        Hadamard matrix subsample :math:`H`.

        :attr:`h` (int): Measurement pattern height. The width is taken to be
        equal to the height, so the measurement pattern is square. The Hadamard
        matrix will have shape :math:`(h^2, h^2)`.

        :attr:`Ord` (torch.tensor, optional): Order matrix used to reorder the
        rows of the measurement matrix :math:`H`. The first new row of :math:`H`
        will correspond to the highest value in :math:`Ord`. Must contain
        :math:`M` values. If some values repeat, the order is kept. Defaults to
        None.

    Attributes:
        :attr:`H` (torch.tensor): The learnable measurement matrix of shape
        :math:`(M, N)`.

        :attr:`H_static` (torch.tensor): alias for :attr:`H`.

        :attr:`P` (torch.tensor): The splitted measurement matrix of shape
        :math:`(2M, N)`.

        :attr:`H_pinv` (torch.tensor, optional): The learnable pseudo inverse
        measurement matrix :math:`H^\dagger` of shape :math:`(N, M)`.

        :attr:`M` (int): Number of measurements performed by the linear operator.
        Is equal to the parameter :attr:`M`.

        :attr:`N` (int): Number of pixels in the image, is equal to :math:`h^2`.

        :attr:`h` (int): Measurement pattern height.

        :attr:`w` (int): Measurement pattern width. Is equal to :math:`h`.

        :attr:`meas_shape` (tuple): Shape of the measurement patterns
        (height, width). Is equal to `(self.h, self.h)`.

        :attr:`indices` (torch.tensor): Indices used to sort the rows of H.	It
        is used by the method :meth:`reindex()`.

        :attr:`Ord` (torch.tensor): Order matrix used to sort the rows of H. It
        is used by :func:`~spyrit.core.torch.sort_by_significance()`.

    .. note::
        The computation of a Hadamard transform :math:`Fx` benefits a fast
        algorithm, as well as the computation of inverse Hadamard transforms.

    .. note::
        :math:`H = H_{+} - H_{-}`

    Example:
        >>> h = 32
        >>> Ord = torch.randn(h, h)
        >>> meas_op = HadamSplit(400, h, Ord)
        >>> print(meas_op)
        HadamSplit(
          (M): 400
          (N): 1024
          (H.shape): torch.Size([400, 1024])
          (meas_shape): (32, 32)
          (H_pinv): True
          (P.shape): torch.Size([800, 1024])
        )
    """

    def __init__(
        self,
        M: int,
        h: int,
        Ord: torch.tensor = None,
    ):

        F = spytorch.walsh2_matrix(h)

        # we pass the whole F matrix to the constructor, but override the
        # calls self.H etc to only return the first M rows
        super().__init__(F, pinv=False, Ord=Ord, meas_shape=(h, h))
        self._M = M

    @property
    def H_pinv(self) -> torch.tensor:
        return self._param_H_static_pinv.data / self.N

    @H_pinv.setter
    def H_pinv(self, value: torch.tensor) -> None:
        self._param_H_static_pinv = nn.Parameter(
            value.to(torch.float64), requires_grad=False
        )

    @H_pinv.deleter
    def H_pinv(self) -> None:
        del self._param_H_static_pinv

    def forward_H(self, x: torch.tensor) -> torch.tensor:
        r"""Optimized measurement simulation using the Fast Hadamard Transform.

        The 2D fast Walsh-ordered Walsh-Hadamard transform is applied to the
        incoming images :math:`x`. This is equivalent to computing :math:`x \cdot H^T`.

        Args:
            :math:`x` (torch.tensor): Batch of images of shape :math:`(*,h,w)`.
            `*` denotes any dimension, for instance `(b,c)` where `b` is the
            batch size and `c` the number of channels. `h` and `w` are the height
            and width of the images.

        Output:
            torch.tensor: The linear measurements of the input images. It has
            shape :math:`(*,M)` where * denotes any number of dimensions and
            `M` the number of measurements.

        Shape:
            :math:`x`: :math:`(*,h,w)` where * denotes any dimension, for
            instance `(b,c)` where `b` is the batch size and `c` the number of
            channels. `h` and `w` are the height and width of the images.

            Output: :math:`(*,M)` where * denotes denotes any number of
            dimensions and `M` the number of measurements.
        """
        m = spytorch.fwht_2d(x)
        m_flat = self.vectorize(m)
        return self.reindex(m_flat, "cols", True)[..., : self.M]

    def build_H_pinv(self):
        """Build the pseudo-inverse (inverse) of the Hadamard matrix H.

        This computes the pseudo-inverse of the Hadamard matrix H, and stores it
        in the attribute H_pinv. In the case of an invertible matrix, the
        pseudo-inverse is the inverse.

        Args:
            None.

        Returns:
            None. The pseudo-inverse is stored in the attribute H_pinv.
        """
        # the division by self.N is done in the property so as to avoid
        # memory overconsumption
        self.H_pinv = self.H.T

    def pinv(self, x, reg="rcond", eta=0.001, diff=False):
        return super().pinv(x, reg, eta, diff)

    def inverse(self, y: torch.tensor) -> torch.tensor:
        r"""Inverse transform of Hadamard-domain images.

        It can be described as :math:`x = H_{had}^{-1}G y`, where :math:`y` is
        the input Hadamard-domain measurements, :math:`H_{had}^{-1}` is the inverse
        Hadamard transform, and :math:`G` is the reordering matrix.

        .. note::
            For this inverse to work, the input vector must have the same number
            of measurements as there are pixels in the original image
            (:math:`M = N`), i.e. no subsampling is allowed.

        .. warning::
            This method is deprecated and will be removed in a future version.
            Use self.pinv instead.

        Args:
            :math:`y`: batch of images in the Hadamard domain of shape
            :math:`(*,c,M)`. `*` denotes any size, `c` the number of
            channels, and `M` the number of measurements (with `M = N`).

        Output:
            :math:`x`: batch of images of shape :math:`(*,c,h,w)`. `*` denotes
            any size, `c` the number of channels, and `h`, `w` the height and
            width of the image (with `h \times w = N = M`).

        Shape:
            :math:`y`: :math:`(*, c, M)` with :math:`*` any size,
            :math:`c` the number of channels, and :math:`N` the number of
            measurements (with `M = N`).

            Output: math:`(*, c, h, w)` with :math:`h` and :math:`w` the height
            and width of the image.

        Example:
            >>> h = 32
            >>> Ord = torch.randn(h, h)
            >>> meas_op = HadamSplit(400, h, Ord)
            >>> y = torch.randn(10, h**2)
            >>> x = meas_op.inverse(y)
            >>> print(x.shape)
            torch.Size([10, 32, 32])
        """
        # permutations
        y = self.reindex(y, "cols", False)
        y = self.unvectorize(y)
        # inverse of full transform
        x = 1 / self.N * spytorch.fwht_2d(y, True)
        return x

    def _pinv_mult(self, y):
        """We use fast walsh-hadamard transform to compute the pseudo inverse.

        Args:
            y (torch.tensor): batch of images in the Hadamard domain of shape
            (*,M). * denotes any size, and M the number of measurements.

        Returns:
            torch.tensor: batch of images in the image domain of shape (*,N).
        """
        # zero-pad the measurements until size N
        y_shape = y.shape
        y_new_shape = y_shape[:-1] + (self.N,)
        y_new = torch.zeros(y_new_shape, device=y.device, dtype=y.dtype)
        y_new[..., : y_shape[-1]] = y

        # unsort the measurements
        y_new = self.reindex(y_new, "cols", False)
        y_new = self.unvectorize(y_new)

        # inverse of full transform
        return 1 / self.N * spytorch.fwht_2d(y_new, True)

    def _set_Ord(self, Ord: torch.tensor) -> None:
        """Set the order matrix used to sort the rows of H."""
        # get only the indices, as done in spyrit.core.torch.sort_by_significance
        self._indices = torch.argsort(-Ord.flatten(), stable=True).to(torch.int32)
        # update the Ord attribute
        self._param_Ord.data = Ord.to(self.device)


# =============================================================================
class DynamicLinear(_Base):
    # =========================================================================
    r"""
    Simulates the measurement of a moving object :math:`y = H \cdot x(t)`.

    Computes linear measurements :math:`y` from incoming images: :math:`y = Hx`,
    where :math:`H` is a linear operator (matrix) and :math:`x` is a
    batch of vectorized images representing a motion picture.

    The class is constructed from a matrix :math:`H` of shape :math:`(M, N)`,
    where :math:`N` represents the number of pixels in the image and
    :math:`M` the number of measurements and the number of frames in the
    animated object.

    .. warning::
        For each call, there must be **exactly** as many images in :math:`x` as
        there are measurements in the linear operator used to initialize the class.

    Args:
        :attr:`H` (torch.tensor): measurement matrix (linear operator) with
        shape :math:`(M, N)`.

        :attr:`Ord` (torch.tensor, optional): Order matrix used to reorder the
        rows of the measurement matrix :math:`H`. The first new row of :math:`H`
        will correspond to the highest value in :math:`Ord`. Must contain
        :math:`M` values. If some values repeat, the order is kept. Defaults to
        None.

        :attr:`meas_shape` (tuple, optional): Shape of the measurement patterns.
        Must be a tuple of two integers representing the height and width of the
        patterns. If not specified, the shape is suppposed to be a square image.
        If not, an error is raised. Defaults to None.

        :attr:`img_shape` (tuple, optional): Shape of the image. Must be a tuple
        of two integers representing the height and width of the image. If not
        specified, the shape is taken as equal to `meas_shape`. Setting this
        value is particularly useful when using an :ref:`extended field of view <_MICCAI24>`.


    Attributes:
        :attr:`H_static` (torch.nn.Parameter): The learnable measurement matrix
        of shape :math:`(M,N)` initialized as :math:`H`.  Only real values are supported.

        :attr:`M` (int): Number of measurements performed by the linear operator.

        :attr:`N` (int): Number of pixels in the image.

        :attr:`h` (int): Measurement pattern height.

        :attr:`w` (int): Measurement pattern width.

        :attr:`meas_shape` (tuple): Shape of the measurement patterns
        (height, width). Is equal to `(self.h, self.w)`.

        :attr:`img_h` (int): Image height.

        :attr:`img_w` (int): Image width.

        :attr:`img_shape` (tuple): Shape of the image (height, width). Is equal
        to `(self.img_h, self.img_w)`.

        :attr:`H_dyn` (torch.tensor): Dynamic measurement matrix :math:`H`.
        Must be set using the method :meth:`build_H_dyn` before being accessed.

        :attr:`H` (torch.tensor): Alias for :attr:`H_dyn`.

        :attr:`H_dyn_pinv` (torch.tensor): Dynamic pseudo-inverse measurement
        matrix :math:`H_{dyn}^\dagger`. Must be set using the method
        :meth:`build_H_dyn_pinv` before being accessed.

        :attr:`H_pinv` (torch.tensor): Alias for :attr:`H_dyn_pinv`.

    .. warning::
        The attributes :attr:`H` and :attr:`H_pinv` are used as aliases for
        :attr:`H_dyn` and :attr:`H_dyn_pinv`. If you want to access the static
        versions of the attributes, be sure to include the suffix `_static`.

    Example:
        >>> H_static = torch.rand([400, 1600])
        >>> meas_op = DynamicLinear(H_static)
        >>> print(meas_op)
        DynamicLinear(
          (M): 400
          (N): 1600
          (H.shape): torch.Size([400, 1600])
          (meas_shape): (40, 40)
          (H_dyn): False
          (img_shape): (40, 40)
          (H_pinv): False
        )

    Reference:
    .. _MICCAI24:
        [MaBP24] (MICCAI 2024 paper #883) Thomas Maitre, Elie Bretin, Romain Phan, Nicolas Ducros,
        Michaël Sdika. Dynamic Single-Pixel Imaging on an Extended Field of View
        without Warping the Patterns. 2024. hal-04533981
    """

    # Class variable
    _measurement_mode = "static"

    def __init__(
        self,
        H: torch.tensor,
        Ord: torch.tensor = None,
        meas_shape: tuple = None,  # (height, width)
        img_shape: tuple = None,  # (height, width)
    ):
        super().__init__(H, Ord, meas_shape)

        if img_shape is not None:
            self._img_shape = img_shape
            if img_shape[0] < self.meas_shape[0] or img_shape[1] < self.meas_shape[1]:
                raise ValueError(
                    "The image shape must be at least as large as the measurement "
                    + f"shape. Got image shape {img_shape} and measurement shape "
                    + f"{self.meas_shape}."
                )
        # else, it is done in the _Base class __init__ (set to meas_shape)

    @property
    def H(self) -> torch.tensor:
        """Dynamic measurement matrix H. Equal to self.H_dyn."""
        return self.H_dyn

    @property
    def H_dyn(self) -> torch.tensor:
        """Dynamic measurement matrix H."""
        try:
            return self._param_H_dyn.data
        except AttributeError as e:
            raise AttributeError(
                "The dynamic measurement matrix H has not been set yet. "
                + "Please call build_H_dyn() before accessing the attribute "
                + "H_dyn (or H)."
            ) from e

    @H_dyn.setter
    def H_dyn(self, value: torch.tensor) -> None:
        self._param_H_dyn = nn.Parameter(value.to(torch.float64), requires_grad=False)
        del H_pinv

    @property
    def recon_mode(self) -> str:
        """Interpolation mode used for reconstruction."""
        return self._recon_mode

    @property
    def H_pinv(self) -> torch.tensor:
        """Dynamic pseudo-inverse H_pinv. Equal to self.H_dyn_pinv."""
        return self.H_dyn_pinv

    @H_pinv.deleter
    def H_pinv(self) -> None:
        del self.H_dyn_pinv

    @property
    def H_dyn_pinv(self) -> torch.tensor:
        """Dynamic pseudo-inverse H_pinv."""
        try:
            return self._param_H_dyn_pinv.data
        except AttributeError as e:
            raise AttributeError(
                "The dynamic pseudo-inverse H_pinv has not been set yet. "
                + "Please call build_H_dyn_pinv() before accessing the attribute "
                + "H_dyn_pinv (or H_pinv)."
            ) from e

    @H_dyn_pinv.setter
    def H_dyn_pinv(self, value: torch.tensor) -> None:
        self._param_H_dyn_pinv = nn.Parameter(
            value.to(torch.float64), requires_grad=False
        )

    @H_dyn_pinv.deleter
    def H_dyn_pinv(self) -> None:
        try:
            del self._param_H_dyn_pinv
        except UnboundLocalError:
            pass

    # @mprof.profile
    def build_H_dyn(self, motion: DeformationField, mode: str = "bilinear") -> None:
        """Build the dynamic measurement matrix `H_dyn`.

        Compute and store the dynamic measurement matrix `H_dyn` from the static
        measurement matrix `H_static` and the deformation field `motion`. The
        output is stored in the attribute `self.H_dyn`.

        This is done using the physical version explained in [MaBP24]_.

        Args:

            :attr:`motion` (DeformationField): Deformation field representing the
            motion of the image.

            :attr:`mode` (str): Interpolation mode. Can only be 'bilinear' for
            now. Bicubic interpolation will be available in a future release.
            Defaults to 'bilinear'.

        Returns:

            None. The dynamic measurement matrix is stored in the attribute
            `self.H_dyn`.

        References:
        .. _MaBP24:
            [MaBP24] (MICCAI 2024 paper #883) Thomas Maitre, Elie Bretin, Romain Phan, Nicolas Ducros,
            Michaël Sdika. Dynamic Single-Pixel Imaging on an Extended Field of View
            without Warping the Patterns. 2024. hal-04533981
        """

        if self.device != motion.device:
            raise RuntimeError(
                "The device of the motion and the measurement operator must be the same."
            )

        # store the mode in attribute
        self._recon_mode = mode

        try:
            del self._param_H_dyn
            del self._param_H_dyn_pinv
            warnings.warn(
                "The dynamic measurement matrix pseudo-inverse H_pinv has "
                + "been deleted. Please call self.build_H_dyn_pinv() to "
                + "recompute it.",
                UserWarning,
            )
        except AttributeError:
            pass

        n_frames = motion.n_frames

        # get deformation field from motion
        # scale from [-1;1] x [-1;1] to [0;width-1] x [0;height-1]
        scale_factor = (torch.tensor(self.img_shape) - 1).to(self.device)
        def_field = (motion.field + 1) / 2 * scale_factor

        # drawings of the kernels for bilinear and bicubic 'interpolation'
        #   00    point      01
        #    +------+--------+
        #    |      |        |
        #    |      |        |
        #    +------+--------+ point
        #    |      |        |
        #    +------+--------+
        #   10               11

        #      00          01   point   02          03
        #       +-----------+-----+-----+-----------+
        #       |           |           |           |
        #       |           |     |     |           |
        #       |        11 |           | 12        |
        #    10 +-----------+-----+-----+-----------+ 13
        #       |           |     |     |           |
        #       + - - - - - + - - + - - + - - - - - + point
        #       |           |     |     |           |
        #    20 +-----------+-----+-----+-----------+ 23
        #       |        21 |     |     | 22        |
        #       |           |           |           |
        #       |           |     |     |           |
        #       +-----------+-----+-----+-----------+
        #      30          31           32          33

        kernel_size = self._spline(torch.tensor([0]), mode).shape[1]
        kernel_width = kernel_size - 1
        kernel_n_pts = kernel_size**2

        # PART 1: SEPARATE THE INTEGER AND DECIMAL PARTS OF THE FIELD
        # _________________________________________________________________
        # crop def_field to keep only measured area
        # moveaxis because crop expects (h,w) as last dimensions
        def_field = spytorch.center_crop(
            def_field.moveaxis(-1, 0), self.meas_shape
        ).moveaxis(
            0, -1
        )  # shape (n_frames, meas_h, meas_w, 2)

        # coordinate of top-left closest corner
        def_field_floor = def_field.floor().to(torch.int64)
        # shape (n_frames, meas_h, meas_w, 2)
        # compute decimal part in x y direction
        dx, dy = torch.split((def_field - def_field_floor), [1, 1], dim=-1)
        del def_field
        dx, dy = dx.squeeze(-1), dy.squeeze(-1)
        # dx.shape = dy.shape = (n_frames, meas_h, meas_w)
        # evaluate the spline at the decimal part
        dxy = torch.einsum(
            "iajk,ibjk->iabjk", self._spline(dy, mode), self._spline(dx, mode)
        ).reshape(n_frames, kernel_n_pts, self.h * self.w)
        # shape (n_frames, kernel_n_pts, meas_h*meas_w)
        del dx, dy

        # PART 2: FLATTEN THE INDICES
        # _________________________________________________________________
        # we consider an expanded grid (img_h+k)x(img_w+k), where k is
        # (kernel_width). This allows each part of the (kernel_size^2)-
        # point grid to contribute to the interpolation.
        # get coordinate of point _00
        def_field_00 = def_field_floor - (kernel_size // 2 - 1)
        del def_field_floor
        # shift the grid for phantom rows/columns
        def_field_00 += kernel_width
        # create a mask indicating if either of the 2 indices is out of bounds
        # (w,h) because the def_field is in (x,y) coordinates
        maxs = torch.tensor(
            [self.img_w + kernel_width, self.img_h + kernel_width], device=self.device
        )
        mask = torch.logical_or(
            (def_field_00 < 0).any(dim=-1), (def_field_00 >= maxs).any(dim=-1)
        )  # shape (n_frames, meas_h, meas_w)
        # trash index receives all the out-of-bounds indices
        trash = (maxs[0] * maxs[1]).to(torch.int64).to(self.device)
        # if the indices are out of bounds, we put the trash index
        # otherwise we put the flattened index (y*w + x)
        flattened_indices = torch.where(
            mask,
            trash,
            def_field_00[..., 0] + def_field_00[..., 1] * (self.img_w + kernel_width),
        ).reshape(n_frames, self.h * self.w)
        del def_field_00, mask

        # PART 3: WARP H MATRIX WITH FLATTENED INDICES
        # _________________________________________________________________
        # Build 4 submatrices with 4 weights for bilinear interpolation
        if isinstance(self, DynamicLinearSplit):
            meas_pattern = self.P
        else:
            meas_pattern = self.H_static
        meas_dxy = (
            meas_pattern.reshape(n_frames, 1, self.h * self.w).to(dxy.dtype) * dxy
        )
        del dxy, meas_pattern
        # shape (n_frames, kernel_size^2, meas_h*meas_w)
        # Create a larger H_dyn that will be folded
        meas_dxy_sorted = torch.zeros(
            (
                n_frames,
                kernel_n_pts,
                (self.img_h + kernel_width) * (self.img_w + kernel_width)
                + 1,  # +1 for trash
            ),
            dtype=torch.float64,
            device=self.device,
        )
        # add at flattened_indices the values of meas_dxy (~warping)
        meas_dxy_sorted.scatter_add_(
            2, flattened_indices.unsqueeze(1).expand_as(meas_dxy), meas_dxy
        )
        del flattened_indices, meas_dxy
        # drop last column (trash)
        meas_dxy_sorted = meas_dxy_sorted[:, :, :-1]
        self.meas_dxy_sorted = meas_dxy_sorted
        # PART 4: FOLD THE MATRIX
        # _________________________________________________________________
        # define operator
        fold = nn.Fold(
            output_size=(self.img_h, self.img_w),
            kernel_size=(kernel_size, kernel_size),
            padding=kernel_width,
        )
        H_dyn = fold(meas_dxy_sorted).reshape(n_frames, self.img_h * self.img_w)
        # store in _param_H_dyn
        self._param_H_dyn = nn.Parameter(H_dyn, requires_grad=False).to(self.device)

    def build_H_dyn_pinv(self, reg: str = "rcond", eta: float = 1e-3) -> None:
        """Computes the pseudo-inverse of the dynamic measurement matrix
        `H_dyn` and stores it in the attribute `H_dyn_pinv`.

        This method supposes that the dynamic measurement matrix `H_dyn` has
        already been set using the method `build_H_dyn()`. An error will be
        raised if `H_dyn` has not been set yet.

        Args:
            :attr:`reg` (str): Regularization method. Can be either 'rcond',
            'L2' or 'H1'. Defaults to 'rcond'.

            :attr:`eta` (float): Regularization parameter. Defaults to 1e-6.

        Raises:
            AttributeError: If the dynamic measurement matrix `H_dyn` has not
            been set yet.
        """
        # later do with regularization parameter
        try:
            H_dyn = self.H_dyn.to(torch.float64)
        except AttributeError as e:
            raise AttributeError(
                "The dynamic measurement matrix H has not been set yet. "
                + "Please call build_H_dyn() before computing the pseudo-inverse."
            ) from e
        self.H_dyn_pinv = self._build_pinv(H_dyn, reg, eta)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates the measurement of a motion picture :math:`y = H \cdot x(t)`.

        The output :math:`y` is computed as :math:`y = Hx`, where :math:`H` is
        the measurement matrix and :math:`x` is a batch of images.

        Args:
            :math:`x`: Batch of images of shape :math:`(*, t, c, h, w)`. `*`
            denotes any dimension (e.g. the batch size), `t` the number of frames,
            `c` the number of channels, and `h`, `w` the height and width of the
            images.

        Output:
            :math:`y`: Linear measurements of the input images. It has shape
            :math:`(*, c, M)` where * denotes any number of dimensions, `c` the
            number of channels, and `M` the number of measurements.

        .. warning::
            There must be **exactly** as many images as there are measurements
            in the linear operator used to initialize the class, i.e.
            `t = M`.

        Shape:
            :math:`x`: :math:`(*, t, c, h, w)`, where * denotes the batch size,
            `t` the number of frames, `c` the number of channels, and `h`, `w`
            the height and width of the images.

            :math:`output`: :math:`(*, c, M)`, where * denotes the batch size,
            `c` the number of channels, and `M` the number of measurements.

        Example:
            >>> x = torch.rand([10, 400, 3, 40, 40])
            >>> H = torch.rand([400, 1600])
            >>> meas_op = DynamicLinear(H)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([10, 3, 400])
        """
        return self._dynamic_forward_with_op(x, self.H_static)

    def forward_H_dyn(self, x: torch.tensor) -> torch.tensor:
        """Simulates the acquisition of measurements using the dynamic measurement matrix H_dyn.

        This supposes the dynamic measurement matrix H_dyn has been set using the
        method build_H_dyn(). An error will be raised if H_dyn has not been set yet.

        Args:
            x (torch.tensor): still image of shape (*, h, w). * denotes any dimension.
            h and w are the height and width of the image. If h and w are larger
            than the measurement pattern, the image is center-cropped to the measurement
            pattern size.

        Returns:
            torch.tensor: Measurement of the input image. It has shape (*, M).
        """
        x = spytorch.center_crop(x, self.meas_shape)
        return self._static_forward_with_op(x, self.H_dyn)

    def _set_Ord(self, Ord: torch.tensor) -> None:
        """Set the order matrix used to sort the rows of H."""
        super()._set_Ord(Ord)
        # delete self.H (self._param_H_dyn)
        try:
            del self._param_H_dyn
            warnings.warn(
                "The dynamic measurement matrix H has been deleted. "
                + "Please call build_H_dyn() to recompute it."
            )
        except AttributeError:
            pass
        # delete self.H_pinv (self._param_H_dyn_pinv)
        try:
            del self._param_H_dyn_pinv
            warnings.warn(
                "The dynamic pseudo-inverse H_pinv has been deleted. "
                + "Please call build_H_dyn_pinv() to recompute it."
            )
        except AttributeError:
            pass

    def _spline(self, dx, mode):
        """
        Returns a 2D row-like tensor containing the values of dx evaluated at
        each B-spline (2 values for bilinear, 4 for bicubic).
        dx must be between 0 and 1.

        Shapes
            dx: (n_frames, self.h, self.w)
            out: (n_frames, {2,4}, self.h, self.w)
        """
        if mode == "bilinear":
            ans = torch.stack((1 - dx, dx), dim=1)
        elif mode == "bicubic":
            ans = torch.stack(
                (
                    (1 - dx) ** 3 / 6,
                    2 / 3 - dx**2 * (2 - dx) / 2,
                    2 / 3 - (1 - dx) ** 2 * (1 + dx) / 2,
                    dx**3 / 6,
                ),
                dim=1,
            )
        elif mode == "schaum":
            ans = torch.stack(
                (
                    dx / 6 * (dx - 1) * (2 - dx),
                    (1 - dx / 2) * (1 - dx**2),
                    (1 + (dx - 1) / 2) * (1 - (dx - 1) ** 2),
                    1 / 6 * (dx + 1) * dx * (dx - 1),
                ),
                dim=1,
            )
        else:
            raise NotImplementedError(
                f"The mode {mode} is invalid, please choose bilinear, "
                + "bicubic or schaum."
            )
        return ans.to(self.device)


# =============================================================================
class DynamicLinearSplit(DynamicLinear):
    # =========================================================================
    r"""
    Simulates the measurement of a moving object using a splitted operator
    :math:`y = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix} \cdot x(t)`.

    Computes linear measurements :math:`y` from incoming images: :math:`y = Px`,
    where :math:`P` is a linear operator (matrix) and :math:`x` is a batch of
    vectorized images representing a motion picture.

    The matrix :math:`P` contains only positive values and is obtained by
    splitting a measurement matrix :math:`H` such that
    :math:`P` has a shape of :math:`(2M, N)` and `P[0::2, :] = H_{+}` and
    `P[1::2, :] = H_{-}`, where :math:`H_{+} = \max(0,H)` and
    :math:`H_{-} = \max(0,-H)`.

    The class is constructed from the :math:`M` by :math:`N` matrix :math:`H`,
    where :math:`N` represents the number of pixels in the image and
    :math:`M` the number of measurements. Therefore, the shape of :math:`P` is
    :math:`(2M, N)`.

    Args:
        :attr:`H` (torch.tensor): measurement matrix (linear operator) with
        shape :math:`(M, N)` where :math:`M` is the number of measurements and
        :math:`N` the number of pixels in the image. Only real values are supported.

        :attr:`Ord` (torch.tensor, optional): Order matrix used to reorder the
        rows of the measurement matrix :math:`H`. The first new row of :math:`H`
        will correspond to the highest value in :math:`Ord`. Must contain
        :math:`M` values. If some values repeat, the order is kept. Defaults to
        None.

        :attr:`meas_shape` (tuple, optional): Shape of the measurement patterns.
        Must be a tuple of two integers representing the height and width of the
        patterns. If not specified, the shape is suppposed to be a square image.
        If not, an error is raised. Defaults to None.

        :attr:`img_shape` (tuple, optional): Shape of the image. Must be a tuple
        of two integers representing the height and width of the image. If not
        specified, the shape is taken as equal to `meas_shape`. Setting this
        value is particularly useful when using an :ref:`extended field of view <_MICCAI24>`.

    Attributes:
        :attr:`H_static` (torch.nn.Parameter): The learnable measurement matrix
        of shape :math:`(M,N)` initialized as :math:`H`.

        :attr:`P` (torch.nn.Parameter): The splitted measurement matrix of
        shape :math:`(2M, N)` such that `P[0::2, :] = H_{+}` and `P[1::2, :] = H_{-}`.

        :attr:`M` (int): Number of measurements performed by the linear operator.

        :attr:`N` (int): Number of pixels in the image.

        :attr:`h` (int): Measurement pattern height.

        :attr:`w` (int): Measurement pattern width.

        :attr:`meas_shape` (tuple): Shape of the measurement patterns
        (height, width). Is equal to `(self.h, self.w)`.

        :attr:`img_h` (int): Image height.

        :attr:`img_w` (int): Image width.

        :attr:`img_shape` (tuple): Shape of the image (height, width). Is equal
        to `(self.img_h, self.img_w)`.

        :attr:`H_dyn` (torch.tensor): Dynamic measurement matrix :math:`H`.
        Must be set using the method :meth:`build_H_dyn` before being accessed.

        :attr:`H` (torch.tensor): Alias for :attr:`H_dyn`.

        :attr:`H_dyn_pinv` (torch.tensor): Dynamic pseudo-inverse measurement
        matrix :math:`H_{dyn}^\dagger`. Must be set using the method
        :meth:`build_H_dyn_pinv` before being accessed.

        :attr:`H_pinv` (torch.tensor): Alias for :attr:`H_dyn_pinv`.

    .. warning::
        For each call, there must be **exactly** as many images in :math:`x` as
        there are measurements in the linear operator :math:`P`.

    Example:
        >>> H = torch.rand([400,1600])
        >>> meas_op = DynamicLinearSplit(H)
        >>> print(meas_op)
        DynamicLinearSplit(
          (M): 400
          (N): 1600
          (H.shape): torch.Size([400, 1600])
          (meas_shape): (40, 40)
          (H_dyn): False
          (img_shape): (40, 40)
          (H_pinv): False
          (P.shape): torch.Size([800, 1600])
        )

    Reference:
    .. _MICCAI24:
        [MaBP24] (MICCAI 2024 paper #883) Thomas Maitre, Elie Bretin, Romain Phan, Nicolas Ducros,
        Michaël Sdika. Dynamic Single-Pixel Imaging on an Extended Field of View
        without Warping the Patterns. 2024. hal-04533981
    """

    def __init__(
        self,
        H: torch.tensor,
        Ord: torch.tensor = None,
        meas_shape: tuple = None,  # (height, width)
        img_shape: tuple = None,  # (height, width)
    ):
        # call constructor of DynamicLinear
        super().__init__(H, Ord, meas_shape, img_shape)
        self._set_P(self.H_static)

    @property  # override _Base definition
    def operator(self) -> torch.tensor:
        return self.P

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates the measurement of a motion picture :math:`y = P \cdot x(t)`.

        The output :math:`y` is computed as :math:`y = Px`, where :math:`P` is
        the measurement matrix and :math:`x` is a batch of images.

        The matrix :math:`P` contains only positive values and is obtained by
        splitting a measurement matrix :math:`H` such that
        :math:`P` has a shape of :math:`(2M, N)` and `P[0::2, :] = H_{+}` and
        `P[1::2, :] = H_{-}`, where :math:`H_{+} = \max(0,H)` and
        :math:`H_{-} = \max(0,-H)`.

        If you want to measure with the original matrix :math:`H`, use the
        method :meth:`forward_H`.

        Args:
            :attr:`x`: Batch of images of shape :math:`(*, t, c, h, w)` where *
            denotes any dimension (e.g. the batch size), :math:`t` the number of
            frames, :math:`c` the number of channels, and :math:`h`, :math:`w`
            the height and width of the images.

        Output:
            :math:`y`: Linear measurements of the input images. It has shape
            :math:`(*, c, 2M)` where * denotes any number of dimensions, :math:`c`
            the number of channels, and :math:`M` the number of measurements.

        .. important::
            There must be as many images as there are measurements in the split
            linear operator, i.e. :math:`t = 2M`.

        Shape:
            :math:`x`: :math:`(*, t, c, h, w)`

            :math:`P` has a shape of :math:`(2M, N)` where :math:`M` is the
            number of measurements as defined by the first dimension of :math:`H`
            and :math:`N` is the number of pixels in the image.

            :math:`output`: :math:`(*, c, 2M)` or :math:`(*, c, t)`

        Example:
            >>> x = torch.rand([10, 800, 3, 40, 40])
            >>> H = torch.rand([400, 1600])
            >>> meas_op = DynamicLinearSplit(H)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([10, 3, 800])
        """
        return self._dynamic_forward_with_op(x, self.P)

    def forward_H(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates the measurement of a motion picture :math:`y = H \cdot x(t)`.

        The output :math:`y` is computed as :math:`y = Hx`, where :math:`H` is
        the measurement matrix and :math:`x` is a batch of images.

        The matrix :math:`H` can contain positive and negative values and is
        given by the user at initialization. If you want to measure with the
        splitted matrix :math:`P`, use the method :meth:`forward`.

        Args:
            :attr:`x`: Batch of images of shape :math:`(*, t, c, h, w)` where *
            denotes any dimension (e.g. the batch size), :math:`t` the number of
            frames, :math:`c` the number of channels, and :math:`h`, :math:`w`
            the height and width of the images.

        Output:
            :math:`y`: Linear measurements of the input images. It has shape
            :math:`(*, c, M)` where * denotes any number of dimensions, :math:`c`
            the number of channels, and :math:`M` the number of measurements.

        .. important::
            There must be as many images as there are measurements in the original
            linear operator, i.e. :math:`t = M`.

        Shape:
            :math:`x`: :math:`(*, t, c, h, w)`

            :math:`H` has a shape of :math:`(M, N)` where :math:`M` is the
            number of measurements and :math:`N` is the number of pixels in the
            image.

            :math:`output`: :math:`(*, c, M)`

        Example:
            >>> x = torch.rand([10, 400, 3, 40, 40])
            >>> H = torch.rand([400, 1600])
            >>> meas_op = LinearDynamicSplit(H)
            >>> y = meas_op.forward_H(x)
            >>> print(y.shape)
            torch.Size([10, 3, 400])
        """
        return super().forward(x)

    def _set_Ord(self, Ord: torch.tensor) -> None:
        """Set the order matrix used to sort the rows of H."""
        super()._set_Ord(Ord)
        # update P
        self._set_P(self.H_static)


# =============================================================================
class DynamicHadamSplit(DynamicLinearSplit):
    # =========================================================================
    r"""
    Simulates the measurement of a moving object using a splitted operator
    :math:`y = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix} \cdot x(t)` with
    :math:`H` a Hadamard matrix.

    Computes linear measurements from incoming images: :math:`y = Px`,
    where :math:`P` is a linear operator (matrix) with positive entries and
    :math:`x` is a batch of vectorized images representing a motion picture.

    The matrix :math:`P` contains only positive values and is obtained by
    splitting a Hadamard-based matrix :math:`H` such that
    :math:`P` has a shape of :math:`(2M, N)` and `P[0::2, :] = H_{+}` and
    `P[1::2, :] = H_{-}`, where :math:`H_{+} = \max(0,H)` and
    :math:`H_{-} = \max(0,-H)`.

    :math:`H` is obtained by selecting a re-ordered subsample of :math:`M` rows
    of a "full" Hadamard matrix :math:`F` with shape :math:`(N^2, N^2)`.
    :math:`N` must be a power of 2.

    Args:
        :attr:`M` (int): Number of measurements. If :math:`M < h^2`, the
        measurement matrix :math:`H` is cropped to :math:`M` rows.

        :attr:`h` (int): Measurement pattern height, must be a power of 2. The
        image is assumed to be square, so the number of pixels in the image is
        :math:`N = h^2`.

        :attr:`Ord` (torch.tensor, optional): Order matrix used to reorder the
        rows of the measurement matrix :math:`H`. The first new row of :math:`H`
        will correspond to the highest value in :math:`Ord`. Must contain
        :math:`M` values. If some values repeat, the order is kept. Defaults to
        None.

        :attr:`img_shape` (tuple, optional): Shape of the image. Must be a tuple
        of two integers representing the height and width of the image. If not
        specified, the shape is taken as equal to `meas_shape`. Setting this
        value is particularly useful when using an :ref:`extended field of view <_MICCAI24>`.


    Attributes:
        :attr:`H_static` (torch.nn.Parameter): The learnable measurement matrix
        of shape :math:`(M,N)` initialized as :math:`H`.

        :attr:`P` (torch.nn.Parameter): The splitted measurement matrix of
        shape :math:`(2M, N)` such that `P[0::2, :] = H_{+}` and `P[1::2, :] = H_{-}`.

        :attr:`M` (int): Number of measurements performed by the linear operator.

        :attr:`N` (int): Number of pixels in the image.

        :attr:`h` (int): Measurement pattern height.

        :attr:`w` (int): Measurement pattern width.

        :attr:`meas_shape` (tuple): Shape of the measurement patterns
        (height, width). Is equal to `(self.h, self.w)`.

        :attr:`img_h` (int): Image height.

        :attr:`img_w` (int): Image width.

        :attr:`img_shape` (tuple): Shape of the image (height, width). Is equal
        to `(self.img_h, self.img_w)`.

        :attr:`H_dyn` (torch.tensor): Dynamic measurement matrix :math:`H`.
        Must be set using the method :meth:`build_H_dyn` before being accessed.

        :attr:`H` (torch.tensor): Alias for :attr:`H_dyn`.

        :attr:`H_dyn_pinv` (torch.tensor): Dynamic pseudo-inverse measurement
        matrix :math:`H_{dyn}^\dagger`. Must be set using the method
        :meth:`build_H_dyn_pinv` before being accessed.

        :attr:`H_pinv` (torch.tensor): Alias for :attr:`H_dyn_pinv`.

    .. note::
        The computation of a Hadamard transform :math:`Fx` benefits a fast
        algorithm, as well as the computation of inverse Hadamard transforms.

    .. note::
        :math:`H = H_{+} - H_{-}`

    Example:
        >>> Ord = torch.rand([32,32])
        >>> meas_op = HadamSplitDynamic(400, 32, Ord)
        >>> print(meas_op)
        DynamicHadamSplit(
          (M): 400
          (N): 1024
          (H.shape): torch.Size([400, 1024])
          (meas_shape): (32, 32)
          (H_dyn): False
          (img_shape): (32, 32)
          (H_pinv): False
          (P.shape): torch.Size([800, 1024])
        )

    Reference:
    .. _MICCAI24:
        [MaBP24] (MICCAI 2024 paper #883) Thomas Maitre, Elie Bretin, Romain Phan, Nicolas Ducros,
        Michaël Sdika. Dynamic Single-Pixel Imaging on an Extended Field of View
        without Warping the Patterns. 2024. hal-04533981
    """

    def __init__(
        self,
        M: int,
        h: int,
        Ord: torch.tensor = None,
        img_shape: tuple = None,  # (height, width)
    ):

        F = spytorch.walsh2_matrix(h)
        # empty = torch.empty(h**2, h**2)  # just to get the shape

        # we pass the whole F matrix to the constructor
        super().__init__(F, Ord, (h, h), img_shape)
        self._M = M

    def _set_Ord(self, Ord: torch.tensor) -> None:
        """Set the order matrix used to sort the rows of H."""
        # get only the indices, as done in spyrit.core.torch.sort_by_significance
        self._indices = torch.argsort(-Ord.flatten(), stable=True).to(torch.int32)
        # update the Ord attribute
        self._param_Ord.data = Ord.to(self.device)
