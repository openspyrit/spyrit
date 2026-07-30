"""
Measurement operators, static and dynamic.

There are 10 classes contained in this module, each representing a different
type of measurement operator. Seven of them are static, i.e. they are used to
simulate measurements of still images, and three are dynamic, i.e. they are used
to simulate measurements of moving objects, represented as a sequence of images.
The inheritance tree is as follows::

      Linear --------------------------------------> DynamicLinear
        |----> FreeformLinear                              |
        |      |---> FreeformSmatrix                       |
        |                                                  |
        |----> HadamSmatrix2d                              | 
        V                                                  V
    LinearSplit                                    DynamicLinearSplit
        |----> FreeformLinearSplit                         |
        V                                                  V
    HadamSplit2d                                   DynamicHadamSplit2d

"""

import math
import warnings
from typing import Any, Union
from collections.abc import Iterable

# import memory_profiler as mprof

import numpy as np
import torch
import torch.nn as nn

from spyrit.core.warp import DeformationField
import spyrit.core.torch as spytorch
import spyrit.misc.walsh_hadamard as wh


# =============================================================================
class Linear(nn.Module):
    r"""
    Simulates linear measurements

    .. math::
        m =\mathcal{N}\left(Hx\right),

    where :math:`\mathcal{N} \colon\, \mathbb{R}^M \to \mathbb{R}^M` represents a noise operator (e.g., Gaussian),
    :math:`H\in\mathbb{R}^{M\times N}` is the acquisition matrix, :math:`x \in \mathbb{R}^N` is the signal of interest,
    :math:`M` is the number of measurements, and :math:`N` is the dimension of the signal.

    .. important::
        The vector :math:`x \in \mathbb{R}^N` represents a multi-dimensional array
        (e.g, an image :math:`X \in \mathbb{R}^{N_1 \times N_2}` with :math:`N = N_1 \times N_2`).

    Args:
        :attr:`H` (:class:`torch.tensor`): measurement matrix (linear operator)
        with shape :math:`(M, N)`. Only real values are supported.

        :attr:`meas_shape` (tuple, optional): Shape of the underliying
        multi-dimensional array :math:`X`. Must be a tuple of integers
        :math:`(N_1, ... ,N_k)` such that :math:`\prod_k N_k = N`. If not, an
        error is raised. Defaults to None.

        :attr:`meas_dims` (tuple, optional): Dimensions of :math:`X` the
        acquisition matrix applies to. Must be a tuple with the same length as
        :attr:`meas_shape`. If not, an error is raised. Defaults to the last
        dimensions of the multi-dimensional array :math:`X` (e.g., `(-2,-1)`
        when `len(meas_shape)`).

        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model :math:`\mathcal{N}`.
        Defaults to = `torch.nn.Identity()`.

    Attributes:
        :attr:`H` (:class:`torch.tensor`): (Learnable) measurement matrix of shape
        :math:`(M, N)` initialized as :math:`H`.

        :attr:`meas_shape` (tuple): Shape of the underlying
        multi-dimensional array :math:`X`.

        :attr:`meas_dims` (tuple): Dimensions the acquisition matrix applies to.

        :attr:`meas_ndim` (int): Number of dimensions the
        acquisition matrix applies to. This is `len(meas_dims)`

        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model :math:`\mathcal{N}`.

        :attr:`M` (int): Number of measurements :math:`M`.

    .. note::
        The full matrix :math:`H` might not actually be stored in memory
        as an :class:`torch.nn.Parameter`. Some subclasses (e.g.
        :class:`HadamSplit2d`, :class:`HadamSmatrix2d`) instead expose
        :attr:`H` as a computed property, built on the fly from a much
        smaller matrix by exploiting a separable structure, to avoid
        materializing a potentially very large :math:`M\times N` matrix
        (e.g. :math:`h^2\times h^2` for an :math:`h\times h` image). This
        is controlled by the class attribute :attr:`_store_H_as_parameter`
        (True by default in :class:`Linear`); subclasses that compute
        :attr:`H` on the fly instead set it to False.

    Example 1: :meth:`measure` applies the matrix :math:`H` to a batch of
    flat signals, computing :math:`y = Hx`.
        >>> H = torch.randn(10, 15)
        >>> meas_op = Linear(H)
        >>> x = torch.randn(3, 15)
        >>> y = meas_op.measure(x)
        >>> print(y.shape)
        torch.Size([3, 10])
        >>> print(torch.allclose(y, torch.einsum("mn,bn->bm", H, x)))
        True

    Example 2: :meth:`forward` additionally applies :attr:`noise_model` to
    the measurements; with the default (:class:`torch.nn.Identity`), it is
    equivalent to :meth:`measure`. To simulate realistic acquisitions with
    noise, pass a noise model from :mod:`spyrit.core.noise`.
        >>> H = torch.randn(10, 15)
        >>> meas_op = Linear(H)  # noise_model defaults to nn.Identity()
        >>> x = torch.randn(3, 15)
        >>> print(torch.equal(meas_op(x), meas_op.measure(x)))
        True

    Example 3: With :attr:`meas_shape`, :math:`H` is applied to a
    multi-dimensional (e.g. image) signal instead of a flat vector -- the
    dimensions in :attr:`meas_dims` (the last two by default) are
    flattened internally before multiplying by :math:`H`.
        >>> H = torch.randn(20, 12 * 8)
        >>> meas_op = Linear(H, meas_shape=(12, 8))
        >>> img = torch.rand(5, 12, 8)
        >>> y = meas_op(img)
        >>> print(y.shape)
        torch.Size([5, 20])

    Example 4: :meth:`adjoint` returns a flat vector of length :math:`N`
    by default. Passing :attr:`unvectorize` = True reshapes it back to the
    original signal shape (:attr:`meas_shape`, placed at :attr:`meas_dims`)
    instead, using :meth:`unvectorize`.
        >>> x_hat = meas_op.adjoint(y, unvectorize=True)
        >>> print(x_hat.shape)
        torch.Size([5, 12, 8])
    """

    # Subclasses that expose H as a computed @property instead of storing it
    # as an nn.Parameter (typically to avoid materializing a very large
    # matrix, e.g. HadamSplit2d, HadamSmatrix2d) should override this to
    # False.
    _store_H_as_parameter = True

    def __init__(
        self,
        H: torch.tensor,
        meas_shape: Union[int, torch.Size, Iterable[int]] = None,
        meas_dims: Union[int, torch.Size, Iterable[int]] = None,
        *,
        noise_model: nn.Module = nn.Identity(),
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        if meas_shape is None:
            meas_shape = H.shape[-1]

        if type(meas_shape) is int:
            meas_shape = [meas_shape]
        self.meas_shape = torch.Size(meas_shape)

        if meas_dims is None:
            meas_dims = list(range(-len(self.meas_shape), 0))
        if type(meas_dims) is int:
            meas_dims = [meas_dims]
        self.meas_dims = torch.Size(meas_dims)

        # don't store H as a Parameter for subclasses that expose H as a
        # computed property instead (e.g. HadamSplit2d, HadamSmatrix2d),
        # typically to avoid materializing a very large matrix. Such
        # subclasses set the class attribute _store_H_as_parameter = False.
        if self._store_H_as_parameter:
            self.H = nn.Parameter(H, requires_grad=False).to(dtype=dtype, device=device)
        self.noise_model = noise_model

        # additional attributes
        self.M = H.shape[0]
        self.meas_ndim = len(self.meas_dims)
        self.N = self.meas_shape.numel()
        self.last_dims = tuple(range(-self.meas_ndim, 0))  # for permutations

        if len(self.meas_shape) != len(self.meas_dims):
            raise ValueError("meas_shape and meas_dims must have the same length")
        if H.ndim != 2:
            raise ValueError("matrix must have 2 dimensions")
        if H.shape[1] != self.N:
            raise ValueError(
                f"The number of columns in the matrix ({H.shape[1]}) does "
                + f"not match the number of measured items ({self.N}) "
                + f"in the measurement shape {self.meas_shape}."
            )

        # define the available matrices for reconstruction
        self._available_pinv_matrices = ["H"]
        self._selected_pinv_matrix = "H"  # select default here (no choice)

    @property
    def device(self) -> torch.device:
        # if we have a split object, it has a A matrix
        if self.H.device == getattr(self, "A", self.H).device:
            return self.H.device
        else:
            raise RuntimeError(
                f"device undefined, H and A are on different device (found {self.H.device} and {self.A.device} respectively)"
            )

    @property
    def dtype(self) -> torch.dtype:
        # if we have a split object, it has a A matrix
        if self.H.dtype == getattr(self, "A", self.H).dtype:
            return self.H.dtype
        else:
            raise RuntimeError(
                f"dtype undefined, H and A are of different dtype (found {self.H.dtype} and {self.A.dtype} respectively)"
            )

    @property
    def matrix_to_inverse(self) -> str:
        return self._selected_pinv_matrix

    @property
    def get_matrix_to_inverse(self) -> torch.tensor:
        return getattr(self, self._selected_pinv_matrix)

    def set_matrix_to_inverse(self, matrix_name: str) -> None:
        if matrix_name in self._available_pinv_matrices:
            self._selected_pinv_matrix = matrix_name
        else:
            raise KeyError(
                f"Matrix {matrix_name} not available for pinv. Available matrices: {self._available_pinv_matrices.keys()}"
            )

    def measure(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate noiseless measurements

        .. math::
            m = Hx,

        where :math:`H\in\mathbb{R}^{M\times N}` is the acquisition matrix,
        :math:`x \in \mathbb{R}^N` is the signal of interest,
        :math:`M` is the number of measurements, and
        :math:`N` is the dimension of the signal.

        .. note::
            This method does not degrade measurement with noise. To do so, see :func:`~spyrit.core.meas.forward()`

        Args:
            :attr:`x` (:class:`torch.tensor`): A batch of signals :math:`x`. The
            dimensions indexed by :attr:`self.meas_dims` must match the measurement
            shape :attr:`self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A batch of measurement of shape :math:`(*, M)` where * denotes
            all the dimensions of the input tensor that are not included in :attr:`self.meas_dims`.

        Example:
            (3, 4) signals of length 15 are measured with an acquisition matrix of shape (10, 15). This produces (3, 4) measurements of length 10.

            >>> H = torch.randn(10, 15)
            >>> meas_op = Linear(H)
            >>> x = torch.randn(3, 4, 15)
            >>> y = meas_op.measure(x)
            >>> print(y.shape)
            torch.Size([3, 4, 10])

            3 signals of length (15, 4) are measured with an acquisition matrix of shape (10, 60). This produces 3 measurements of length 10. The acquisition matrix applies to both dimensions -2 and -1.

            >>> H = torch.randn(10, 60)
            >>> meas_op = Linear(H, meas_shape=(15, 4))
            >>> x = torch.randn(3, 15, 4)
            >>> y = meas_op.measure(x)
            >>> print(y.shape)
            torch.Size([3, 10])
            >>> print(meas_op.meas_dims)
            torch.Size([-2, -1])
        """
        x = self.vectorize(x)
        x = torch.einsum("mn,...n->...m", self.H, x)
        return x

    def forward(self, x: torch.tensor):
        r"""Simulate noisy measurements

        .. math::
            m =\mathcal{N}\left(Hx\right),

        where :math:`\mathcal{N} \colon\, \mathbb{R}^M \to \mathbb{R}^M` represents a noise operator (e.g., Gaussian), :math:`H\in\mathbb{R}^{M\times N}` is the acquisition matrix, :math:`x \in \mathbb{R}^N` is the signal of interest, :math:`M` is the number of measurements, and :math:`N` is the dimension of the signal.

        .. note::
            This method degrades measurements with noise. To compute :math:`Hx` only, see :func:`~spyrit.core.meas.measure()`.

        Args:
            :attr:`x` (:class:`torch.tensor`): A batch of signals :math:`x`. The
            dimensions indexed by :attr:`self.meas_dims` must match the measurement
            shape :attr:`self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A batch of measurement of shape :math:`(*, M)` where * denotes
            all the dimensions of the input tensor that are not included in :attr:`self.meas_dims`.

        Example:
            Example 1: (3, 4) signals of length 15 are measured with an acquisition matrix of shape (10, 15). This produces (3, 4) measurements of length 10.

            >>> H = torch.randn(10, 15)
            >>> meas_op = Linear(H)
            >>> x = torch.randn(3, 4, 15)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([3, 4, 10])

            Example 2: 3 signals of length (15, 4) are measured with an acquisition matrix of shape (10, 60). This produces 3 measurements of length 10. The acquisition matrix applies to both dimensions -2 and -1.

            >>> H = torch.randn(10, 60)
            >>> meas_op = Linear(H, meas_shape=(15, 4))
            >>> x = torch.randn(3, 15, 4)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([3, 10])
            >>> print(meas_op.meas_dims)
            torch.Size([-2, -1])
        """
        x = self.measure(x)
        x = self.noise_model(x)
        return x

    def unvectorize(self, input: torch.tensor) -> torch.tensor:
        r"""Unflatten the measured dimensions.

        This method first expands the last dimension into the measurement
        shape (:attr:`self.meas_shape`), and then moves the expanded dimensions to
        their original positions as defined by :attr:`self.meas_dims`.

        Input:
            input (:class:`torch.tensor`): A tensor of shape (:attr:`*, self.N`) where * denotes any batch size.

        Output:
            :class:`torch.tensor`: A tensor whose dimensions given by :attr:`self.meas_dims` have shape :attr:`self.meas_shape`.

        See also:
            For the opposite operation use :meth:`vectorize()`.

        Example:
            >>> import spyrit.core.meas as meas
            >>> matrix = torch.randn(10, 60)
            >>> meas_op = meas.Linear(matrix, meas_shape=(12, 5), meas_dims=(-1,-3))
            >>> x = torch.randn(3, 7, 60)
            >>> print(meas_op.unvectorize(x).shape)
            torch.Size([3, 5, 7, 12])
        """
        # unvectorize the last dimension
        input = input.reshape(*input.shape[:-1], *self.meas_shape)
        # move the measured dimensions to their original positions
        if self.meas_dims != self.last_dims:
            input = torch.movedim(input, self.last_dims, self.meas_dims)
        return input

    def adjoint(self, m: torch.tensor, unvectorize=False):
        r"""Apply adjoint of matrix H.

        It computes

        .. math::
            x = H^Tm,

        where :math:`H^T\in\mathbb{R}^{N\times M}` is the adjoint of the
        acquisition matrix, :math:`m \in \mathbb{R}^M` is a measurement.

        Args:
            :attr:`m` (:class:`torch.tensor`): A batch of measurement
            :math:`m` of shape :math:`(*, M)` where :math:`*`  denotes all the
            dimensions that are not included in :attr:`self.meas_dims`

            :attr:`unvectorize` (:obj:`bool`, optional): Whether to unvectorize
            the measurement dimensions. This calls
            :meth:`~spyrit.core.meas.unvectorize()` after mutiplication by the
            adjoint. Defaults to False.

        Returns:
            :class:`torch.tensor`: A batch of signals :math:`x`.
            If :attr:`unvectorize` is :obj:`False`, :math:`x` has shape
            :math:`(*, N)` where :math:`*` is the same as for :attr:`m`. If
            :attr:`unvectorize` is :obj:`True`, :math:`x` is reshaped such that
            the dimensions :attr:`self.meas_dims` match the measurement shape
            :attr:`self.meas_shape`.


        Example:
            Example 2: (3, 4) measurements of length 10 produces (3, 4) signals
            of length 10.

            >>> H = torch.randn(10, 15)
            >>> meas_op = Linear(H)
            >>> m = torch.randn(3, 4, 10)
            >>> x = meas_op.adjoint(m)
            >>> print(x.shape)
            torch.Size([3, 4, 15])


            Example 2: 3 measurements of length 10 produces 3 signals of length
            60

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 60)
            >>> meas_op = meas.Linear(H, meas_shape=(15, 4))
            >>> m = torch.randn(3, 10)
            >>> x = meas_op.adjoint(m)
            >>> print(x.shape)
            torch.Size([3, 60])

            Using unvectorize=True produces 3 signals of length (15, 4)

            >>> x = meas_op.adjoint(m, unvectorize=True)
            >>> print(x.shape)
            torch.Size([3, 15, 4])
        """
        m = torch.einsum("mn,...m->...n", self.H, m)
        if unvectorize:
            m = self.unvectorize(m)
        return m

    def vectorize(self, input: torch.tensor) -> torch.tensor:
        r"""Flatten the measured dimensions.

        The tensor is flattened at the indicated `self.meas_dims` dimensions. The
        flattened dimensions are then collapsed into one, which is the last
        dimension of the output tensor.

        Input:
            input (:class:`torch.tensor`): A tensor whose dimensions given by :attr:`self.meas_dims` have shape :attr:`self.meas_shape`.

        Output:
            :class:`torch.tensor`: A tensor of shape (:attr:`*, self.meas_shape`) where * denotes all the dimensions of the input tensor not included in :attr:`self.meas_dims`.

        See also:
            For the opposite operation use :meth:`unvectorize()`.

        Example:
            >>> import spyrit.core.meas as meas
            >>> matrix = torch.randn(10, 60)
            >>> meas_op = meas.Linear(matrix, meas_shape=(12, 5), meas_dims=(-1,-3))
            >>> x = torch.randn(3, 5, 7, 12)
            >>> print(meas_op.vectorize(x).shape)
            torch.Size([3, 7, 60])
        """
        # move all measured dimensions to the end
        if self.meas_dims != self.last_dims:
            input = torch.movedim(input, self.meas_dims, self.last_dims)
        # flatten the measured dimensions
        input = input.reshape(*input.shape[: -self.meas_ndim], self.N)
        return input

    def unvectorize(self, input: torch.tensor) -> torch.tensor:
        r"""Unflatten the measured dimensions.

        This method first expands the last dimension into the measurement
        shape (:attr:`self.meas_shape`), and then moves the expanded dimensions to
        their original positions as defined by :attr:`self.meas_dims`.

        Input:
            :class:`input` (:class:`torch.tensor`): A tensor of shape (:attr:`*, self.N`) where * denotes any batch size.

        Output:
            :class:`torch.tensor`: A tensor whose dimensions given by :attr:`self.meas_dims` have shape :attr:`self.meas_shape`.

        See also:
            For the opposite operation use :meth:`vectorize()`.

        Example:
            >>> import spyrit.core.meas as meas
            >>> matrix = torch.randn(10, 60)
            >>> meas_op = meas.Linear(matrix, meas_shape=(12, 5), meas_dims=(-1,-3))
            >>> x = torch.randn(3, 7, 60)
            >>> print(meas_op.unvectorize(x).shape)
            torch.Size([3, 5, 7, 12])
        """
        # unvectorize the last dimension
        input = input.reshape(*input.shape[:-1], *self.meas_shape)
        # move the measured dimensions to their original positions
        if self.meas_dims != self.last_dims:
            input = torch.movedim(input, self.last_dims, self.meas_dims)
        return input


# =============================================================================
class FreeformLinear(Linear):
    r"""Simulate linear measurements in a region of interest

    .. math::
        m =\mathcal{N}\left(Hx\right), \quad \text{where }x = \text{mask}(\tilde{x})

    where :math:`\mathcal{N} \colon\, \mathbb{R}^M \to \mathbb{R}^M` represents a noise operator (e.g., Gaussian), :math:`H\in\mathbb{R}^{M\times N}` is the acquisition matrix, :math:`x \in \mathbb{R}^N` is the signal in the region of interest, :math:`M` is the number of measurements, :math:`N` is the number of pixels in the region of interest, :math:`\text{mask} \colon\, \mathbb{R}^\tilde{N} \to \mathbb{R}^N` represents the masking operation, :math:`\tilde{x} \in \mathbb{R}^\tilde{N}` is the full signal, and :math:`\tilde{N}\ge N` is the dimension of the full signal :math:`\tilde{x}`.

    Args:
        :attr:`H` (:class:`torch.tensor`): measurement matrix (linear operator)
        with shape :math:`(M, N)`. Only real values are supported.

        :attr:`meas_shape` (tuple): Shape of the underliying
        multi-dimensional array :math:`X`. Must be a tuple of integers
        :math:`(N_1, N_2)` such that :math:`N_1 \times N_2 \ge N`. If not, an
        error is raised.

        :attr:`meas_dims` (tuple, optional): Dimensions of :math:`X` the
        acquisition matrix applies to. Must be a tuple with the same length as
        :attr:`meas_shape`. If not, an error is raised. Defaults to the last
        dimensions of the multi-dimensional array :math:`X` (e.g., `(-2,-1)`
        when `len(meas_shape)`).

        :attr:`index_masked` (:class:`torch.tensor`): Indices of :math:`X`
        where measurement applies. This is a tensor with shape shape
        :math:`(2, N)`.

        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model :math:`\mathcal{N}`. Defaults to = :obj:`torch.nn.Identity`.

    .. note::

        Only tested for measurements in 2D using mask indices.

    Example: Select one every second pixel on the diagonal of a batch of images
        >>> images = torch.rand(17, 3, 40, 40)
        >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
        >>> H = torch.randn(13, 20)
        >>> meas_op = FreeformLinear(H, meas_shape=(40,40), index_mask=mask)
        >>> x_masked = meas_op(images)
        >>> print(x_masked.shape)
        torch.Size([17, 3, 13])
    """

    def __init__(
        self,
        H: torch.tensor,
        meas_shape: Union[int, torch.Size, Iterable[int]] = None,
        meas_dims: Union[int, torch.Size, Iterable[int]] = None,
        index_mask: torch.tensor = None,  # must have shape (len(meas_shape), H.shape[-1])
        bool_mask: torch.tensor = None,
        *,
        noise_model: nn.Module = nn.Identity(),
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            H,
            H.shape[-1],  # meas_shape,
            -1,  # meas_dims
            noise_model=noise_model,
            dtype=dtype,
            device=device,
        )

        # select mask type
        if index_mask is not None:
            if bool_mask is not None:
                warnings.warn(
                    "Both index_mask and bool_mask have been specified. Using index_mask."
                )
            self.index_mask = index_mask
            self.mask_type = "index"
            # measurement shape
            if meas_shape is None:
                raise ValueError("meas_shape must be specified when index_mask is used")
            self.meas_shape = meas_shape
            # measurement dimensions
            if meas_dims is None:
                meas_dims = list(range(-len(self.meas_shape), 0))
            if type(meas_dims) is int:
                meas_dims = [meas_dims]
            self.meas_dims = torch.Size(meas_dims)
            self.meas_ndim = len(self.meas_dims)
            self.last_dims = tuple(range(-self.meas_ndim, 0))  # for permutations

        else:
            if bool_mask is not None:
                self.bool_mask = bool_mask
                self.mask_type = "bool"
                self.meas_shape = bool_mask.shape
            else:
                raise ValueError("Either index_mask or bool_mask must be specified.")

        # check mask dimensions in the case of index mask
        if self.mask_type == "index":
            if index_mask.ndim != 2:
                raise ValueError("index_mask must have 2 dimensions.")
            if index_mask.shape[0] != len(self.meas_shape):
                raise ValueError(
                    "The first dimension of index_mask must match the number of dimensions in meas_shape."
                )
            if index_mask.shape[1] != self.N:
                raise ValueError(
                    f"The second dimension of index_mask ({index_mask.shape[1]}) must "
                    + f"match the number of measured items ({self.N})."
                )
        # check in the case of bool mask
        else:
            if bool_mask.shape != meas_shape:
                raise ValueError("bool_mask must have the same shape as meas_shape.")

    def vectorize(self, x: torch.tensor) -> torch.tensor:
        r"""Appplies the saved mask to the input tensor, where the masked
        dimensions are collapsed into one.

        This method first selects the elements from the input tensor at the
        specified dimensions `self.meas_dims` and based on the mask. The selected
        elements are then flattened into a single dimension which is the last
        dimension of the output tensor.

        Args:
            :attr:`x` (:class:`torch.tensor`): The input tensor to select the mask from. The
            dimensions indexed by `self.meas_dims` should match the measurement shape
            `self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A tensor of shape (\*, self.N) where \* denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example: Select one every second pixel on the diagonal of a batch of images
            >>> images = torch.rand(17, 3, 40, 40)
            >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
            >>> H = torch.randn(13, 20)
            >>> meas_op = FreeformLinear(H, meas_shape=(40,40), index_mask=mask)
            >>> x_masked = meas_op.vectorize(images)
            >>> print(x_masked.shape)
            torch.Size([17, 3, 20])
        """
        x = torch.movedim(x, self.meas_dims, self.last_dims)

        if self.mask_type == "index":
            return x[(..., *self.index_mask)]

        # not tested yet
        elif self.mask_type == "bool":
            # flatten along the masked dimensions
            x = x.reshape(*x.shape[: -self.meas_ndim], self.N)
            return x[..., self.bool_mask.reshape(-1)]

        else:
            raise ValueError(
                f"mask_type must be either 'index' or 'bool', found {self.mask_type}."
            )

    def measure(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate noiseless measurements.

        The mask is first applied to the input tensor, then the input tensor
        is multiplied by the measurement patterns.

        .. note::
            This method does not include the noise model.

        Args:
            :attr:`x` (:class:`torch.tensor`): A tensor where the dimensions indexed by
            `self.meas_dims` match the measurement shape `self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A tensor of shape (\*, self.M) where \* denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example: Select one every second pixel on the diagonal of a batch of images

            >>> images = torch.rand(17, 3, 40, 40)
            >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
            >>> H = torch.randn(13, 20)
            >>> meas_op = FreeformLinear(H, meas_shape=(40,40), index_mask=mask)
            >>> x_masked = meas_op.measure(images)
            >>> print(x_masked.shape)
            torch.Size([17, 3, 13])
        """
        x = self.vectorize(x)
        return torch.einsum("mn,...n->...m", self.H, x)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate measurements.

        The mask is first applied to the input tensor, then the input tensor
        is multiplied by the measurement patterns.

        .. note::
            This method does not include the noise model.

        Args:
            :attr:`x` (:class:`torch.tensor`): A tensor where the dimensions indexed by
            `self.meas_dims` match the measurement shape `self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A tensor of shape (\*, self.M) where \* denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example: Select one every second pixel on the diagonal of a batch of images
            >>> from spyrit.core.meas import FreeformLinear
            >>> import torch
            >>> images = torch.rand(17, 3, 40, 40)
            >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
            >>> H = torch.randn(13, 20)
            >>> meas_op = FreeformLinear(H, meas_shape=(40,40), index_mask=mask)
            >>> x_masked = meas_op(images)
            >>> print(x_masked.shape)
            torch.Size([17, 3, 13])
        """
        x = self.measure(x)
        x = self.noise_model(x)
        return x

    def unvectorize(self, x: torch.tensor, fill_value: Any = 0) -> torch.tensor:
        r"""Unflatten the last dimension of a tensor to the measurement shape at
        the measured dimensions based on the mask.

        This method expands the last dimension into the measurement shape
        `self.meas_shape`, filling the elements not in the mask with the
        `fill_value`. The expanded dimensions are then moved to their original
        positions as defined by `self.meas_dims`.

        .. note::
            This function creates a new tensor filled with the `fill_value` and
            then fills the elements in the mask with the corresponding elements.
            The output tensor is not a view of the input tensor.

        Args:
            :attr:`x` (:class:`torch.tensor`): tensor to be expanded. Its last dimension must
            contain `self.N` elements.

            :attr:`fill_value` (Any, optional): Fill value for all the indices not
            covered by the mask. Defaults to 0.

        Returns:
            :class:`torch.tensor`: A tensor where the dimensions indexed by `self.meas_dims`
            match the measurement shape `self.meas_shape`.

        See also:
            For the opposite operation use :meth:`vectorize()`.

        Example:
            >>> images = torch.rand(17, 3, 40, 40)
            >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
            >>> H = torch.randn(13, 20)
            >>> meas_op = FreeformLinear(H, meas_shape=(40,40), index_mask=mask)
            >>> x = torch.randn(17, 3, 20)
            >>> print(meas_op.unvectorize(x).shape)
            torch.Size([17, 3, 40, 40])
        """

        if self.mask_type == "index":
            # create a new tensor with the final shape
            output = torch.full(
                (*x.shape[:-1], *self.meas_shape),
                fill_value,
                dtype=x.dtype,
                device=x.device,
            )
            output[(..., *self.index_mask)] = x

        # not tested yet
        elif self.mask_type == "bool":
            # create a new tensor with an intermediate shape
            output = torch.full(
                (*x.shape[:-1], self.N),
                fill_value,
                dtype=x.dtype,
                device=x.device,
            )
            output[..., self.bool_mask.reshape(-1)] = x
            output = output.reshape(*output.shape[:-1], *self.meas_shape)

        else:
            raise ValueError(
                f"mask_type must be either 'index' or 'bool', found {self.mask_type}."
            )

        return torch.movedim(output, self.last_dims, self.meas_dims)


# =============================================================================
class FreeformSmatrix(FreeformLinear):
    r"""Simulate linear measurements in a freeform region of interest,
    using an S-matrix as the acquisition matrix.

    .. math::
        m =\mathcal{N}\left(Hx\right), \quad \text{where }x = \text{mask}(\tilde{x})

    where :math:`\mathcal{N} \colon\, \mathbb{R}^M \to \mathbb{R}^M` represents 
    a noise operator (e.g., Gaussian), :math:`S\in\mathbb{R}^{M\times N}` is a
    S matrix, :math:`x \in \mathbb{R}^N` is the signal in the region 
    of interest, :math:`M` is the number of measurements, :math:`N` is the 
    number of pixels in the region of interest, 
    :math:`\text{mask} \colon\, \mathbb{R}^\tilde{N} \to \mathbb{R}^N` 
    represents the masking operation, 
    :math:`\tilde{x} \in \mathbb{R}^\tilde{N}` is the full signal, and 
    :math:`\tilde{N}\ge N` is the dimension of the full signal :math:`\tilde{x}`.

    This class plays the same role as :class:`FreeformLinear`, but instead
    of accepting an arbitrary measurement matrix :math:`H`, it
    sets :math:`H` as an S-matrix (see
    :func:`spyrit.misc.walsh_hadamard.walsh_S_matrix`). The S-matrix is built 
    from a Hadamard matrix of order :math:`N+1`, so :math:`N+1` must be
    a power of two.

    Args:
        :attr:`meas_shape` (tuple): Shape of the underlying
        multi-dimensional array :math:`X`. See :class:`FreeformLinear`.

        :attr:`M` (int, optional): Number of measurements. Defaults to
        :math:`N` (no subsampling), where :math:`N` is the number of
        masked pixels (deduced from :attr:`index_mask` or
        :attr:`bool_mask`).

        :attr:`index_mask` (:class:`torch.tensor`, optional): See
        :class:`FreeformLinear`.

        :attr:`bool_mask` (:class:`torch.tensor`, optional): See
        :class:`FreeformLinear`.

        :attr:`order` (:class:`torch.tensor`, optional): Length-:math:`N`
        order vector that defines the measurements to keep (one value per
        masked pixel). The first component of :math:`y` will correspond to
        the index where :attr:`order` is the highest. Defaults to `None`
        (keeps the natural S-matrix row order).

        :attr:`computation` (str, optional): Either `"dense"` or
        `"dyadic"`. See the note above for the tradeoff. Defaults to
        `"dyadic"`.

        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model
        :math:`\mathcal{N}`. Defaults to `torch.nn.Identity()`.

        :attr:`dtype` (:class:`torch.dtype`, optional): Data type of the
        measurement matrix. Defaults to `torch.float32`.

        :attr:`device` (:obj:`torch.device`, optional): Device of the
        measurement matrix. Defaults to `torch.device("cpu")`.

    Attributes:
        :attr:`H` (:class:`torch.tensor`): The (subsampled) :math:`M\times
        N` S measurement matrix. When :attr:`computation` is `"dense"`,
        this is precomputed and stored; when `"dyadic"`, it is built on
        the fly if accessed (e.g. by code expecting the generic
        :class:`Linear` interface), which can be memory-heavy for large
        :math:`N` -- prefer :meth:`measure` and :meth:`fast_pinv`, which
        never require it.

        :attr:`T` (:class:`torch.tensor` or `None`): Exact inverse of the
        full :math:`N\times N` S-matrix, used by :meth:`fast_pinv` when
        :attr:`computation` is `"dense"`. `None` when `"dyadic"` (not
        needed: the fast transform is used instead).

        :attr:`computation` (str): `"dense"` or `"dyadic"`.

        :attr:`M` (int): Number of measurements :math:`M`.

        :attr:`N` (int): Number of masked pixels.

        :attr:`order` (:class:`torch.tensor`): Order vector.

        :attr:`indices` (:class:`torch.tensor`): Indices used to reorder
        the measurement vector.

    .. note::
        **Choosing** :attr:`computation`. Two ways of applying the
        S-matrix are available, trading off differently depending on
        :math:`N` and the sampling ratio :math:`M/N`:

        - `"dense"`: builds and stores the S-matrix explicitly (an
          :math:`M\times N` matrix for :attr:`H`, plus an :math:`N\times N`
          matrix :attr:`T` for the pseudo-inverse), and applies it via a
          plain matrix-vector product. Cost scales as :math:`O(MN)` per
          measurement/reconstruction. This is the only option available
          for acquisition matrices that are not built from a power-of-two
          Hadamard matrix (e.g. a generic, non-dyadic :math:`H`); it also
          becomes memory-heavy for large :math:`N` (an :math:`N\times N`
          float32 matrix already exceeds 1 GB around :math:`N=16000`, and
          building it can itself fail with an out-of-memory error before
          any measurement is even taken).

        - `"dyadic"` (**default**): uses the fast Walsh-Hadamard-based
          transform (:func:`spyrit.misc.walsh_hadamard.fwalsh_S_torch` /
          :func:`~spyrit.misc.walsh_hadamard.ifwalsh_S_torch`) instead of
          a matrix-vector product. This requires no :math:`N\times N` (or
          :math:`M\times N`) matrix to ever be stored, and costs
          :math:`O(N\log N)` regardless of :math:`M` -- but that "regardless
          of :math:`M`" is also its main limitation: unlike the dense
          path, it cannot skip work when subsampling, since it always
          computes all :math:`N` outputs (or requires all :math:`N`
          inputs for the inverse) before the top-:math:`M` measurements
          are selected. It relies on the dyadic (power-of-two) recursive
          structure of the Hadamard transform, so it is only applicable
          when :math:`N+1` is a power of two -- which is always the case
          for :class:`FreeformSmatrix`, but would not be for a
          hypothetical S-matrix-like class built on some other Hadamard
          matrix whose order is not a power of two.

        In practice (see benchmarks in the development notes), the
        crossover is around :math:`M/N \approx 0.15`-`0.20`, fairly stable
        across :math:`N` from about 1,000 to 16,000: below that sampling
        ratio, `"dense"` is faster; above it, `"dyadic"` is faster (and,
        for large :math:`N`, is often the only option that fits in
        memory at all). If in doubt, benchmark both on your actual
        :math:`N` and :math:`M`.

    .. note::
        As with :class:`HadamSmatrix2d`, the S-matrix is not orthogonal:
        the exact inverse used by :meth:`fast_pinv` is only exact when
        :attr:`M` equals :attr:`N` (no subsampling); with subsampling, it
        is an approximation. This holds for both values of
        :attr:`computation`.

    Example 1: Select the first 15 pixels on the diagonal of a batch of
    images (:attr:`N`=15, :attr:`N`+1=16=2**4). With full sampling (the default, :attr:`M`=:attr:`N`),
    :meth:`fast_pinv` exactly recovers the masked pixels, regardless of
    :attr:`computation`.

        >>> h = 32
        >>> mask = torch.tensor([[i, i] for i in range(15)]).T
        >>> meas_op = FreeformSmatrix(meas_shape=(h, h), index_mask=mask)
        >>> print(meas_op.computation)
        dyadic
        >>> print(meas_op.N, meas_op.M)
        15 15
        >>> images = torch.rand(4, h, h)
        >>> y = meas_op(images)
        >>> print(y.shape)
        torch.Size([4, 15])
        >>> x_hat = meas_op.fast_pinv(y, vectorize=True)
        >>> x_true = meas_op.vectorize(images)
        >>> print(torch.allclose(x_true, x_hat, atol=1e-4))
        True

    Example 2: With :attr:`vectorize` = :attr:`False` (the default), the
    reconstruction is expanded back to the full image shape instead,
    with unmasked pixels set to :attr:`fill_value` (0 by default).

        >>> x_hat_img = meas_op.fast_pinv(y, vectorize=False)
        >>> print(x_hat_img.shape)
        torch.Size([4, 32, 32])
        >>> print(x_hat_img[0, 20, 20].item())  # (20, 20) is not in the mask
        0.0

    Example 3: With subsampling (:attr:`M` < :attr:`N`), the reconstruction is
    only approximate (see the note above).

        >>> meas_op_sub = FreeformSmatrix(meas_shape=(h, h), M=10, index_mask=mask)
        >>> y_sub = meas_op_sub(images)
        >>> print(y_sub.shape)
        torch.Size([4, 10])
        >>> x_hat_sub = meas_op_sub.fast_pinv(y_sub, vectorize=True)
        >>> print(torch.allclose(x_true, x_hat_sub, atol=1e-4))
        False

    Example 4: The two :attr:`computation` modes give the same result
    (up to floating-point precision), as expected.

        >>> meas_op_dense = FreeformSmatrix(meas_shape=(h, h), index_mask=mask, computation="dense")
        >>> y_dense = meas_op_dense(images)
        >>> print(torch.allclose(y, y_dense, atol=1e-4))
        True
    """

    # H is exposed as a computed @property (see below): stored explicitly
    # when computation="dense", built on the fly (and not cached) when
    # computation="dyadic", to avoid materializing a potentially very
    # large N x N matrix by default.
    _store_H_as_parameter = False

    def __init__(
        self,
        meas_shape: Union[int, torch.Size, Iterable[int]] = None,
        M: int = None,
        index_mask: torch.tensor = None,
        bool_mask: torch.tensor = None,
        order: torch.tensor = None,
        computation: str = "dyadic",
        *,
        noise_model: nn.Module = nn.Identity(),
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        if computation not in ("dense", "dyadic"):
            raise ValueError(
                f"computation must be 'dense' or 'dyadic', got {computation!r}."
            )

        # Determine N (number of masked pixels) before building H, mirroring
        # the validation done inside FreeformLinear.__init__.
        if index_mask is not None:
            if index_mask.ndim != 2:
                raise ValueError("index_mask must have 2 dimensions.")
            N = index_mask.shape[1]
        elif bool_mask is not None:
            N = int(bool_mask.sum().item())
        else:
            raise ValueError("Either index_mask or bool_mask must be specified.")

        if not float(math.log2(N + 1)).is_integer():
            raise ValueError(
                f"N+1 must be a power of two for the S-matrix construction "
                f"(got N={N}, N+1={N + 1}), where N is the number of masked "
                f"pixels (not the number of measurements)."
            )

        if M is None:
            M = N

        if order is None:
            order = torch.ones(N)
        elif order.numel() != N:
            raise ValueError(
                f"order must have {N} elements (one per masked pixel), "
                f"got {order.numel()}."
            )
        indices = torch.argsort(-order.flatten(), stable=True).to(torch.int32)

        if computation == "dense":
            # S-matrix and its exact inverse, built with
            # spyrit.misc.walsh_hadamard as requested.
            S = torch.from_numpy(wh.walsh_S_matrix(N).astype(np.float32))
            T = torch.from_numpy(wh.iwalsh_S_matrix(N).astype(np.float32))
            # H = the top M rows of S, reordered by decreasing order.
            H_init = spytorch.reindex(S, indices, axis="rows", inverse_permutation=False)[
                :M, :
            ]
        else:
            # dyadic: a tiny placeholder, used only for shape validation and
            # attribute setup in FreeformLinear.__init__/Linear.__init__ --
            # never stored, and never the actual S-matrix (which is what we
            # are avoiding materializing here).
            H_init = torch.empty(M, N)
            T = None

        super().__init__(
            H_init,
            meas_shape=meas_shape,
            index_mask=index_mask,
            bool_mask=bool_mask,
            noise_model=noise_model,
            dtype=dtype,
            device=device,
        )

        self.computation = computation
        self.order = order
        # kept as a plain (int32) tensor, not a Parameter, consistent with
        # HadamSmatrix2d's convention; .indices does not automatically
        # follow subsequent .to(device) calls (matching that same
        # pre-existing convention).
        self.indices = indices.to(device=device)

        if computation == "dense":
            self._H_dense = nn.Parameter(H_init, requires_grad=False).to(
                dtype=dtype, device=device
            )
            self.T = nn.Parameter(T, requires_grad=False).to(dtype=dtype, device=device)
            self.walsh_ind = None
        else:
            self._H_dense = None
            self.T = None
            # Cache the permutation indices used internally by
            # fwalsh_S_torch/ifwalsh_S_torch: measured to matter
            # meaningfully (recomputing them every call was 20-40% slower
            # in benchmarks). This is a plain Python list (not a tensor),
            # so no device placement is needed for it.
            self.walsh_ind = wh.sequency_perm_ind(N + 1)

        self._dtype = dtype

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self.indices.device

    @property
    def H(self):
        r"""The (subsampled) :math:`M\times N` S measurement matrix.

        .. warning::
            When :attr:`computation` is `"dyadic"`, this builds the full
            :math:`N\times N` S-matrix on the fly (not cached), which can
            be memory-heavy for large :math:`N`. Prefer :meth:`measure`
            and :meth:`fast_pinv`, which never require it.
        """
        if self.computation == "dense":
            return self._H_dense
        S = torch.from_numpy(wh.walsh_S_matrix(self.N).astype(np.float32)).to(
            dtype=self.dtype, device=self.device
        )
        H = spytorch.reindex(S, self.indices, axis="rows", inverse_permutation=False)[
            : self.M, :
        ]
        return H

    def measure(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate noiseless measurements using the S-matrix.

        The mask is first applied to the input tensor, selecting the
        :math:`N` masked pixels; the result is then transformed by the
        (subsampled) S measurement matrix :math:`S_M`, either as a dense
        matrix-vector product or via the fast Walsh-Hadamard-based
        transform, depending on :attr:`self.computation` (see the class
        docstring note for the tradeoff).

        .. math::
            m = S_M x, \quad \text{where }x = \text{mask}(\tilde{x})

        .. note::
            This method does not include the noise model. See
            :meth:`forward` for noisy measurements.

        Args:
            :attr:`x` (:class:`torch.tensor`): A tensor where the
            dimensions indexed by `self.meas_dims` match the measurement
            shape `self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A tensor of shape (\*, self.M) where \*
            denotes all the dimensions of the input tensor not included in
            `self.meas_dims`.
        """
        x = self.vectorize(x)

        if self.computation == "dense":
            return torch.einsum("mn,...n->...m", self.H, x)
        else:
            # Fast transform always computes all N outputs; select/reorder
            # the top M of them (by self.order) afterward.
            y_full = wh.fwalsh_S_torch(x, self.walsh_ind)
            return y_full[..., self.indices[: self.M].long()]

    def fast_pinv(
        self, m: torch.tensor, vectorize: bool = False, fill_value: Any = 0
    ) -> torch.tensor:
        r"""Apply the pseudo-inverse of the S measurement matrix.

        Depending on :attr:`self.computation`, this uses either a dense
        matrix-vector product with the exact inverse S-matrix, or the fast
        (inverse) Walsh-Hadamard-based transform (see the class docstring
        note for the tradeoff).

        Args:
            :attr:`m` (:class:`torch.tensor`): Measurement :math:`m` of
            length :attr:`self.M`.

            :attr:`vectorize` (bool, optional): If True, returns the flat
            vector of :attr:`self.N` reconstructed masked pixels. If False
            (default), returns the reconstruction expanded back to the
            full :attr:`self.meas_shape` via :meth:`unvectorize`, with
            unmasked pixels set to :attr:`fill_value`.

            :attr:`fill_value` (Any, optional): Fill value for pixels
            outside the mask, used only when :attr:`vectorize` is False.
            Defaults to 0.

        Returns:
            :class:`torch.tensor`: The reconstructed signal, either as a
            flat vector of length :attr:`self.N` (:attr:`vectorize` =
            True) or expanded to :attr:`self.meas_shape` (:attr:`vectorize`
            = False).

        .. note::
            If the number of measurements is smaller than the number of
            masked pixels, the measurement vector is zero-padded (dense)
            or zero-scattered into the correct positions (dyadic) before
            inversion -- the two are mathematically equivalent. As with
            :meth:`HadamSmatrix2d.fast_pinv`, this reconstruction is exact
            only when :attr:`self.M` equals :attr:`self.N` (no
            subsampling); with subsampling, it is an approximation. This
            holds for both values of :attr:`computation`.
        """
        if self.computation == "dense":
            if self.N != self.M:
                m = torch.cat(
                    (
                        m,
                        torch.zeros(
                            *m.shape[:-1], self.N - self.M, device=m.device, dtype=m.dtype
                        ),
                    ),
                    -1,
                )
            m = spytorch.reindex(
                m, self.indices.to(m.device), axis="cols", inverse_permutation=False
            )
            x = torch.einsum("nm,...m->...n", self.T, m)
        else:
            # Scatter m into its correct (order-defined) positions in a
            # length-N vector, zero elsewhere, then apply the fast inverse
            # transform. Mathematically equivalent to the dense path's
            # zero-pad + reindex (verified numerically).
            m_full = torch.zeros(*m.shape[:-1], self.N, device=m.device, dtype=m.dtype)
            m_full[..., self.indices[: self.M].long()] = m
            x = wh.ifwalsh_S_torch(m_full, self.walsh_ind)

        if not vectorize:
            x = self.unvectorize(x, fill_value=fill_value)
        return x


# =============================================================================
class LinearSplit(Linear):
    r"""
    Simulate linear measurements by splitting an acquisition matrix
    :math:`H\in \mathbb{R}^{M\times N}` that contains negative values.
    In practice, only positive values can be implemented using a DMD.
    Therefore, we acquire

    .. math::
        y =\mathcal{N}\left(Ax\right),

    where :math:`\mathcal{N} \colon\, \mathbb{R}^{2M} \to \mathbb{R}^{2M}` represents a noise operator (e.g., Gaussian), :math:`A \colon\, \mathbb{R}_+^{2M\times N}` is the acquisition matrix that contains positive DMD patterns, :math:`x \in \mathbb{R}^N` is the signal of interest., :math:`2M` is the number of DMD patterns, and :math:`N` is the dimension of the signal.

    Given a matrix :math:`H`, we define the positive DMD patterns :math:`A` from the positive and negative components :math:`H`. In practice, the even rows of :math:`A` contain the positive components of :math:`H`, while odd rows of :math:`A` contain the negative components of :math:`H`

    .. math::
        \begin{cases}
            A[0::2, :] = H_{+}, \text{ with } H_{+} = \max(0,H),\\
            A[1::2, :] = H_{-}, \text{ with } H_{-} = \max(0,-H).
        \end{cases}

    .. note::
        :math:`H_{+}` and :math:`H_{-}` are such that :math:`H_{+} - H_{-} = H`.

    .. important::
        The vector :math:`x \in \mathbb{R}^N` represents a multi-dimensional array (e.g, an image :math:`X \in \mathbb{R}^{N_1 \times N_2}` with :math:`N = N_1 \times N_2`).

    Args:
        :attr:`H` (:class:`torch.tensor`): measurement matrix (linear operator)
        with shape :math:`(M, N)`. Only real values are supported.

        :attr:`meas_shape` (tuple, optional): Shape of the underliying
        multi-dimensional array :math:`X`. Must be a tuple of integers
        :math:`(N_1, ... ,N_k)` such that :math:`\prod_k N_k = N`. If not, an
        error is raised. Defaults to None.

        :attr:`meas_dims` (tuple, optional): Dimensions of :math:`X` the
        acquisition matrix applies to. Must be a tuple with the same length as
        :attr:`meas_shape`. If not, an error is raised. Defaults to the last
        dimensions of the multi-dimensional array :math:`X` (e.g., `(-2,-1)`
        when `len(meas_shape)`).

        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model :math:`\mathcal{N}`. Defaults to = `torch.nn.Identity`.

    Attributes:
        :attr:`A` (:class:`torch.tensor`): (Learnable) positive measurement
        matrix of shape :math:`(2M, N)` initialized as :math:`A`.

        :attr:`H` (:class:`torch.tensor`): (Learnable) measurement matrix of shape
        :math:`(M, N)` initialized as :math:`H`.

        :attr:`meas_shape` (tuple): Shape of the underliying
        multi-dimensional array :math:`X`.

        :attr:`meas_dims` (tuple): Dimensions the acquisition matrix applies to.

        :attr:`meas_ndim` (int): Number of dimensions the
        acquisition matrix applies to. This is `len(meas_dims)`

        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model :math:`\mathcal{N}`.

        :attr:`M` (int): Number of measurements :math:`M`.

    Examples:

        Example 1: (3, 4) signals of length 15 are measured with an acquisition matrix of shape (10, 15). This produces (3, 4) measurements of length 20.

        >>> import torch
        >>> import spyrit.core.meas as meas
        >>> H = torch.randn(10, 15)
        >>> meas_op = meas.LinearSplit(H)
        >>> x = torch.randn(3, 4, 15)
        >>> y = meas_op(x)
        >>> print(y.shape)
        torch.Size([3, 4, 20])

        Example 2: 3 signals of length (15, 4) are measured with an acquisition matrix of shape (10, 60). This produces 3 measurements of length 20. The acquisition matrix applies to both dimensions -2 and -1.

        >>> import torch
        >>> import spyrit.core.meas as meas
        >>> H = torch.randn(10, 60)
        >>> meas_op = meas.LinearSplit(H, meas_shape=(15, 4))
        >>> x = torch.randn(3, 15, 4)
        >>> y = meas_op(x)
        >>> print(y.shape)
        torch.Size([3, 20])
        >>> print(meas_op.meas_dims)
        torch.Size([-2, -1])
    """

    def __init__(
        self,
        H,
        meas_shape=None,
        meas_dims=None,
        *,
        noise_model=nn.Identity(),
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            H,
            meas_shape,
            meas_dims,
            noise_model=noise_model,
            dtype=dtype,
            device=device,
        )

        # split positive and negative components
        pos, neg = nn.functional.relu(self.H), nn.functional.relu(-self.H)
        A = torch.cat([pos, neg], 1).reshape(2 * self.M, self.N)
        # A is built from self.H which is cast to device and dtype
        self.A = nn.Parameter(A, requires_grad=False)

        # define the available matrices for reconstruction
        self._available_pinv_matrices = ["H", "A"]
        self._selected_pinv_matrix = "H"  # select default here

        # HERE: device=device, dtype=dtype

    def measure(self, x: torch.tensor):
        r"""Simulate noiseless measurements from matrix A.

        It acquires

        .. math::
            y = Ax,

        where :math:`A \in \mathbb{R}_+^{2M\times N}` is the acquisition matrix that contains positive DMD patterns, :math:`x \in \mathbb{R}^N` is the signal of interest., :math:`2M` is the number of DMD patterns, and :math:`N` is the dimension of the signal.

        Given a matrix :math:`H \in \mathbb{R}^{M\times N}`, we define the positive DMD patterns :math:`A` from the positive and negative components of :math:`H`.

        .. note::
            The acquisition matrix :math:`A` is given by :attr:`self.A`.

        Args:
            :attr:`x` (:class:`torch.tensor`): Signal :math:`x` whose
            dimensions :attr:`self.meas_dims` must have shape
            shape :attr:`self.meas_shape`.

        Returns:
            :class:`torch.tensor`: Measurement vector :math:`y` of length :attr:`2\*self.M`.

        Examples:

            Example 1: (3, 4) signals of length 15 are measured with an acquisition matrix of shape (10, 15). This produces (3, 4) measurements of length 20.

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 15)
            >>> meas_op = meas.LinearSplit(H)
            >>> x = torch.randn(3, 4, 15)
            >>> y = meas_op.measure(x)
            >>> print(y.shape)
            torch.Size([3, 4, 20])

            Example 2: 3 signals of length (15, 4) are measured with an acquisition matrix of shape (10, 60). This produces 3 measurements of length 20. The acquisition matrix applies to both dimensions -2 and -1.

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 60)
            >>> meas_op = meas.LinearSplit(H, meas_shape=(15, 4))
            >>> x = torch.randn(3, 15, 4)
            >>> y = meas_op.measure(x)
            >>> print(y.shape)
            torch.Size([3, 20])
            >>> print(meas_op.meas_dims)
            torch.Size([-2, -1])
        """
        x = self.vectorize(x)
        x = torch.einsum("mn,...n->...m", self.A, x)
        return x

    def measure_H(self, x: torch.tensor):
        r"""Simulate noiseless measurements from matrix H.

        It computes

        .. math::
            m = Hx,

        where :math:`H \in \mathbb{R}^{M\times N}` is the acquisition matrix (that may contain negative values), :math:`x \in \mathbb{R}^N` is the signal of interest, :math:`M` is the number of DMD patterns, and :math:`N` is the dimension of the signal.

        .. note::

            The acquisition matrix :math:`H` is given by :attr:`self.H`.

        Args:
            :attr:`x` (:class:`torch.tensor`): Signal :math:`x` whose
            dimensions :attr:`self.meas_dims` must have shape
            shape :attr:`self.meas_shape`.

        Returns:
            :class:`torch.tensor`: Measurement vector :math:`m` of length :attr:`self.M`.

        Examples:
            Example 1: (3, 4) signals of length 15 are measured with an acquisition matrix of shape (10, 15). This produces (3, 4) measurements of length 10.

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 15)
            >>> meas_op = meas.LinearSplit(H)
            >>> x = torch.randn(3, 4, 15)
            >>> y = meas_op.measure_H(x)
            >>> print(y.shape)
            torch.Size([3, 4, 10])

            Example 2: 3 signals of length (15, 4) are measured with an acquisition matrix of shape (10, 60). This produces 3 measurements of length 10. The acquisition matrix applies to both dimensions -2 and -1.

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 60)
            >>> meas_op = meas.LinearSplit(H, meas_shape=(15, 4))
            >>> x = torch.randn(3, 15, 4)
            >>> y = meas_op.measure_H(x)
            >>> print(y.shape)
            torch.Size([3, 10])
            >>> print(meas_op.meas_dims)
            torch.Size([-2, -1])
        """
        return super().measure(x)

    def adjoint(self, y: torch.tensor, unvectorize=False):
        r"""Apply adjoint of matrix A.

        It computes

        .. math::
            x = A^Ty,

        where :math:`A \in \mathbb{R}^{2M\times N}` is the acquisition matrix (that may contain negative values) and :math:`y \in \mathbb{R}^{2M}` is a measurement vector.

        .. note::

            The acquisition matrix :math:`A` is given by :attr:`self.A`.

        Args:
            :attr:`y` (:class:`torch.tensor`): Measurement :math:`y` whose dimensions :attr:`self.meas_dims` must have shape :attr:`self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A batch of signals :math:`x` with shape :math:`(*, N)` where :math:`*` is the same as for :attr:`m`.

        Examples:
            Example 1: (3, 4) measurements of length 20 are measured with an acquisition matrix of shape (10, 15). This produces (3, 4) signals of length 15.

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 15)
            >>> meas_op = meas.LinearSplit(H)
            >>> y = torch.randn(3, 4, 20)
            >>> x = meas_op.adjoint(y)
            >>> print(x.shape)
            torch.Size([3, 4, 15])

            Example 2: 3 measurements of length 20 are measured with an acquisition matrix of shape (10, 60). This produces 3 signals of length 60.

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 60)
            >>> meas_op = meas.LinearSplit(H, meas_shape=(15, 4))
            >>> m = torch.randn(3, 20)
            >>> x = meas_op.adjoint(m)
            >>> print(x.shape)
            torch.Size([3, 60])
        """
        y = torch.einsum("mn,...m->...n", self.A, y)
        if unvectorize:
            y = self.unvectorize(y)
        return y

    def adjoint_H(self, m: torch.tensor, unvectorize=False):
        r"""Apply adjoint of matrix H.

        It computes

        .. math::
            x = H^Tm,

        where :math:`H \in \mathbb{R}^{M\times N}` is the acquisition matrix (that may contain negative values), :math:`m \in \mathbb{R}^M` is a measurement vector.

        .. note::

            The acquisition matrix :math:`H` is given by :attr:`self.H`.

        Args:
            :attr:`m` (:class:`torch.tensor`): Measurements :math:`m` whose dimensions :attr:`self.meas_dims` must have shape :attr:`self.meas_shape`.

        Returns:
            A batch of signals :math:`x`. If :attr:`unvectorize` is :obj:`False`, :math:`x` has shape :math:`(*, N)` where :math:`*` is the same as for :attr:`m`. If :attr:`unvectorize` is :obj:`True`, :math:`x` is reshaped such that the dimensions :attr:`self.meas_dims` have shape :attr:`self.meas_shape`.

        Examples:
            Example 1: (3, 4) measurements of length 10 are measured with an acquisition matrix of shape (10, 15). This produces (3, 4) signals of length 15.

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 15)
            >>> meas_op = meas.LinearSplit(H)
            >>> m = torch.randn(3, 4, 10)
            >>> x = meas_op.adjoint_H(m)
            >>> print(x.shape)
            torch.Size([3, 4, 15])

            Example 2: 3 measurements of length 10 are measured with an acquisition matrix of shape (10, 60). This produces 3 signals of length 60.

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 60)
            >>> meas_op = meas.LinearSplit(H, meas_shape=(15, 4))
            >>> m = torch.randn(3, 10)
            >>> x = meas_op.adjoint_H(m)
            >>> print(x.shape)
            torch.Size([3, 60])

            Using unvectorize=True produces 3 signals of length (15, 4)

            >>> x = meas_op.adjoint_H(m, unvectorize=True)
            >>> print(x.shape)
            torch.Size([3, 15, 4])
        """
        return super().adjoint(m, unvectorize=unvectorize)

    def forward(self, x: torch.tensor):
        r"""Simulate noisy measurements from matrix A.

        It computes

        .. math::
            y =\mathcal{N}\left(Ax\right),

        where :math:`\mathcal{N} \colon\, \mathbb{R}^{2M} \to \mathbb{R}^{2M}` represents a noise operator (e.g., Gaussian), where :math:`A \in \mathbb{R}_+^{2M\times N}` is the acquisition matrix that contains positive DMD patterns, :math:`x \in \mathbb{R}^N` is the signal of interest., :math:`2M` is the number of DMD patterns, and :math:`N` is the dimension of the signal.

        Given a matrix :math:`H \in \mathbb{R}^{M\times N}`, we define the positive DMD patterns :math:`A` from the positive and negative components of :math:`H`.

        .. note::

            The acquisition matrix :math:`A` is given by :attr:`self.A`.

        Args:
            :attr:`x` (:class:`torch.tensor`): Signal :math:`x` whose
            dimensions :attr:`self.meas_dims` must have shape
            shape :attr:`self.meas_shape`.

        Returns:
            :class:`torch.tensor`: Measurement vector :math:`y` of length :attr:`2*self.M`.

        Examples:

            Example 1: (3, 4) signals of length 15 are measured with an acquisition matrix of shape (10, 15). This produces (3, 4) measurements of length 20.

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 15)
            >>> meas_op = meas.LinearSplit(H)
            >>> x = torch.randn(3, 4, 15)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([3, 4, 20])

            Example 2: 3 signals of length (15, 4) are measured with an acquisition matrix of shape (10, 60). This produces 3 measurements of length 20. The acquisition matrix applies to both dimensions -2 and -1.

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 60)
            >>> meas_op = meas.LinearSplit(H, meas_shape=(15, 4))
            >>> x = torch.randn(3, 15, 4)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([3, 20])
            >>> print(meas_op.meas_dims)
            torch.Size([-2, -1])
        """
        return super().forward(x)

    def forward_H(self, x: torch.tensor):
        r"""Simulate noisy measurements from matrix H.

        It computes

        .. math::
            m =\mathcal{N}\left(Hx\right),

        where :math:`\mathcal{N} \colon\, \mathbb{R}^M \to \mathbb{R}^M` represents a noise operator (e.g., Gaussian), :math:`H \in \mathbb{R}^{M\times N}` is the acquisition matrix (that may contain negative values), :math:`x \in \mathbb{R}^N` is the signal of interest, :math:`M` is the number of DMD patterns, and :math:`N` is the dimension of the signal.

        .. note::

            The acquisition matrix :math:`H` is given by :attr:`self.H`.

        Args:
            :attr:`x` (:class:`torch.tensor`): Signal :math:`x` whose
            dimensions :attr:`self.meas_dims` must have shape
            shape :attr:`self.meas_shape`.

        Returns:
            :class:`torch.tensor`: Measurement vector :math:`m` of length :attr:`self.M`.

        Examples:
            Example 1: (3, 4) signals of length 15 are measured with an acquisition matrix of shape (10, 15). This produces (3, 4) measurements of length 10.

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 15)
            >>> meas_op = meas.LinearSplit(H)
            >>> x = torch.randn(3, 4, 15)
            >>> y = meas_op.forward_H(x)
            >>> print(y.shape)
            torch.Size([3, 4, 10])

            Example 2: 3 signals of length (15, 4) are measured with an acquisition matrix of shape (10, 60). This produces 3 measurements of length 10. The acquisition matrix applies to both dimensions -2 and -1.

            >>> import spyrit.core.meas as meas
            >>> H = torch.randn(10, 60)
            >>> meas_op = meas.LinearSplit(H, meas_shape=(15, 4))
            >>> x = torch.randn(3, 15, 4)
            >>> y = meas_op.forward_H(x)
            >>> print(y.shape)
            torch.Size([3, 10])
            >>> print(meas_op.meas_dims)
            torch.Size([-2, -1])
        """
        x = self.measure_H(x)
        x = self.noise_model(x)
        return x


# =============================================================================
class FreeformLinearSplit(LinearSplit):
    r"""Simulate split measurements in a region of interest

    .. math::
        m =\mathcal{N}\left(Ax\right), \quad \text{where }x = \text{mask}(\tilde{x})

    where :math:`\mathcal{N} \colon\, \mathbb{R}^M \to \mathbb{R}^M` represents a noise operator (e.g., Gaussian), :math:`A\in\mathbb{R}_+^{2M\times N}` is the acquisition matrix, :math:`x \in \mathbb{R}^N` is the signal in the region of interest, :math:`2M` is the number of measurements, :math:`N` is the number of pixels in the region of interest, :math:`\text{mask} \colon\, \mathbb{R}^\tilde{N} \to \mathbb{R}^N` represents the masking operation, :math:`\tilde{x} \in \mathbb{R}^\tilde{N}` is the full signal, and :math:`\tilde{N}\ge N` is the dimension of the full signal :math:`\tilde{x}`.

    Example: Select one every second pixel on the diagonal of a batch of images
        >>> from spyrit.core.meas import FreeformLinearSplit
        >>> import torch
        >>> images = torch.rand(17, 3, 40, 40)
        >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
        >>> H = torch.randn(13, 20)
        >>> meas_op = FreeformLinearSplit(H, meas_shape=(40,40), index_mask=mask)
        >>> x_masked = meas_op(images)
        >>> print(x_masked.shape)
        torch.Size([17, 3, 26])
    """

    def __init__(
        self,
        H: torch.tensor,
        meas_shape: Union[int, torch.Size, Iterable[int]] = None,
        meas_dims: Union[int, torch.Size, Iterable[int]] = None,
        index_mask: torch.tensor = None,  # must have shape (len(meas_shape), H.shape[-1])
        bool_mask: torch.tensor = None,
        *,
        noise_model: nn.Module = nn.Identity(),
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            H,
            H.shape[-1],  # meas_shape,
            -1,  # meas_dims
            noise_model=noise_model,
            dtype=dtype,
            device=device,
        )

        # select mask type
        if index_mask is not None:
            if bool_mask is not None:
                warnings.warn(
                    "Both index_mask and bool_mask have been specified. Using index_mask."
                )
            self.index_mask = index_mask
            self.mask_type = "index"
            # measurement shape
            if meas_shape is None:
                raise ValueError("meas_shape must be specified when index_mask is used")
            self.meas_shape = meas_shape
            # measurement dimensions
            if meas_dims is None:
                meas_dims = list(range(-len(self.meas_shape), 0))
            if type(meas_dims) is int:
                meas_dims = [meas_dims]
            self.meas_dims = torch.Size(meas_dims)
            self.meas_ndim = len(self.meas_dims)
            self.last_dims = tuple(range(-self.meas_ndim, 0))  # for permutations

        else:
            if bool_mask is not None:
                self.bool_mask = bool_mask
                self.mask_type = "bool"
                self.meas_shape = bool_mask.shape
            else:
                raise ValueError("Either index_mask or bool_mask must be specified.")

        # check mask dimensions in the case of index mask
        if self.mask_type == "index":
            if index_mask.ndim != 2:
                raise ValueError("index_mask must have 2 dimensions.")
            if index_mask.shape[0] != len(self.meas_shape):
                raise ValueError(
                    "The first dimension of index_mask must match the number of dimensions in meas_shape."
                )
            if index_mask.shape[1] != self.N:
                raise ValueError(
                    f"The second dimension of index_mask ({index_mask.shape[1]}) must "
                    + f"match the number of measured items ({self.N})."
                )
        # check in the case of bool mask
        else:
            if bool_mask.shape != meas_shape:
                raise ValueError("bool_mask must have the same shape as meas_shape.")

    def vectorize(self, x: torch.tensor) -> torch.tensor:
        r"""Appplies the saved mask to the input tensor, where the masked
        dimensions are collapsed into one.

        This method first selects the elements from the input tensor at the
        specified dimensions `self.meas_dims` and based on the mask. The selected
        elements are then flattened into a single dimension which is the last
        dimension of the output tensor.

        Args:
            :attr:`x` (:class:`torch.tensor`): The input tensor to select the mask from. The
            dimensions indexed by `self.meas_dims` should match the measurement shape
            `self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A tensor of shape (\*, self.N) where \* denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example: Select one every second pixel on the diagonal of a batch of images
            >>> from spyrit.core.meas import FreeformLinearSplit
            >>> import torch
            >>> images = torch.rand(17, 3, 40, 40)
            >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
            >>> H = torch.randn(13, 20)
            >>> meas_op = FreeformLinearSplit(H, meas_shape=(40,40), index_mask=mask)
            >>> x_masked = meas_op.vectorize(images)
            >>> print(x_masked.shape)
            torch.Size([17, 3, 20])
        """
        x = torch.movedim(x, self.meas_dims, self.last_dims)

        if self.mask_type == "index":
            return x[(..., *self.index_mask)]

        # not tested yet
        elif self.mask_type == "bool":
            # flatten along the masked dimensions
            x = x.reshape(*x.shape[: -self.meas_ndim], self.N)
            return x[..., self.bool_mask.reshape(-1)]

        else:
            raise ValueError(
                f"mask_type must be either 'index' or 'bool', found {self.mask_type}."
            )

    def measure(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate measurements from signal/image.

        The mask is first applied to the input tensor, then the input tensor
        is multiplied by the measurement patterns.

        .. note::
            This method does not include the noise model.

        Args:
            :attr:`x` (:class:`torch.tensor`): A tensor where the dimensions indexed by
            `self.meas_dims` match the measurement shape `self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A tensor of shape (\*, self.M) where \* denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example: Select one every second pixel on the diagonal of a batch of images
            >>> from spyrit.core.meas import FreeformLinearSplit
            >>> import torch
            >>> images = torch.rand(17, 3, 40, 40)
            >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
            >>> H = torch.randn(13, 20)
            >>> meas_op = FreeformLinearSplit(H, meas_shape=(40,40), index_mask=mask)
            >>> x_masked = meas_op.measure(images)
            >>> print(x_masked.shape)
            torch.Size([17, 3, 26])
        """
        x = self.vectorize(x)
        return torch.einsum("mn,...n->...m", self.A, x)

    def measure_H(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate measurements from signal/image.

        The mask is first applied to the input tensor, then the input tensor
        is multiplied by the measurement patterns.

        .. note::
            This method does not include the noise model.

        Args:
            :attr:`x` (:class:`torch.tensor`): A tensor where the dimensions indexed by
            `self.meas_dims` match the measurement shape `self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A tensor of shape (\*, self.M) where \* denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example: Select one every second pixel on the diagonal of a batch of images
            >>> from spyrit.core.meas import FreeformLinearSplit
            >>> import torch
            >>> images = torch.rand(17, 3, 40, 40)
            >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
            >>> H = torch.randn(13, 20)
            >>> meas_op = FreeformLinearSplit(H, meas_shape=(40,40), index_mask=mask)
            >>> x_masked = meas_op.measure_H(images)
            >>> print(x_masked.shape)
            torch.Size([17, 3, 13])
        """
        x = self.vectorize(x)
        return torch.einsum("mn,...n->...m", self.H, x)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate measurements.

        The mask is first applied to the input tensor, then the input tensor
        is multiplied by the measurement patterns.

        .. note::
            This method does not include the noise model.

        Args:
            :attr:`x` (:class:`torch.tensor`): A tensor where the dimensions indexed by
            `self.meas_dims` match the measurement shape `self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A tensor of shape (\*, self.M) where \* denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example: Select one every second pixel on the diagonal of a batch of images
            >>> from spyrit.core.meas import FreeformLinearSplit
            >>> import torch
            >>> images = torch.rand(17, 3, 40, 40)
            >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
            >>> H = torch.randn(13, 20)
            >>> meas_op = FreeformLinearSplit(H, meas_shape=(40,40), index_mask=mask)
            >>> x_masked = meas_op(images)
            >>> print(x_masked.shape)
            torch.Size([17, 3, 26])
        """
        x = self.measure(x)
        x = self.noise_model(x)
        return x

    def forward_H(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate measurements.

        The mask is first applied to the input tensor, then the input tensor
        is multiplied by the measurement patterns.

        .. note::
            This method does not include the noise model.

        Args:
            :attr:`x` (:class:`torch.tensor`): A tensor where the dimensions indexed by
            `self.meas_dims` match the measurement shape `self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A tensor of shape (\*, self.M) where \* denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example: Select one every second pixel on the diagonal of a batch of images
            >>> from spyrit.core.meas import FreeformLinearSplit
            >>> import torch
            >>> images = torch.rand(17, 3, 40, 40)
            >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
            >>> H = torch.randn(13, 20)
            >>> meas_op = FreeformLinearSplit(H, meas_shape=(40,40), index_mask=mask)
            >>> x_masked = meas_op.forward_H(images)
            >>> print(x_masked.shape)
            torch.Size([17, 3, 13])
        """
        x = self.measure_H(x)
        x = self.noise_model(x)
        return x

    def unvectorize(self, x: torch.tensor, fill_value: Any = 0) -> torch.tensor:
        r"""Unflatten the last dimension of a tensor to the measurement shape at
        the measured dimensions based on the mask.

        This method expands the last dimension into the measurement shape
        `self.meas_shape`, filling the elements not in the mask with the
        `fill_value`. The expanded dimensions are then moved to their original
        positions as defined by `self.meas_dims`.

        .. note::
            This function creates a new tensor filled with the `fill_value` and
            then fills the elements in the mask with the corresponding elements.
            The output tensor is not a view of the input tensor.

        Args:
            :attr:`x` (:class:`torch.tensor`): tensor to be expanded. Its last dimension must
            contain `self.N` elements.

            :attr:`fill_value` (Any, optional): Fill value for all the indices not
            covered by the mask. Defaults to 0.

        Returns:
            :class:`torch.tensor`: A tensor where the dimensions indexed by `self.meas_dims`
            match the measurement shape `self.meas_shape`.

        See also:
            For the opposite operation use :meth:`vectorize()`.

        Example:
            >>> from spyrit.core.meas import FreeformLinearSplit
            >>> import torch
            >>> images = torch.rand(17, 3, 40, 40)
            >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
            >>> H = torch.randn(13, 20)
            >>> meas_op = FreeformLinearSplit(H, meas_shape=(40,40), index_mask=mask)
            >>> x = torch.randn(17, 3, 20)
            >>> print(meas_op.unvectorize(x).shape)
            torch.Size([17, 3, 40, 40])
        """

        if self.mask_type == "index":
            # create a new tensor with the final shape
            output = torch.full(
                (*x.shape[:-1], *self.meas_shape),
                fill_value,
                dtype=x.dtype,
                device=x.device,
            )
            output[(..., *self.index_mask)] = x

        # not tested yet
        elif self.mask_type == "bool":
            # create a new tensor with an intermediate shape
            output = torch.full(
                (*x.shape[:-1], self.N),
                fill_value,
                dtype=x.dtype,
                device=x.device,
            )
            output[..., self.bool_mask.reshape(-1)] = x
            output = output.reshape(*output.shape[:-1], *self.meas_shape)

        else:
            raise ValueError(
                f"mask_type must be either 'index' or 'bool', found {self.mask_type}."
            )

        return torch.movedim(output, self.last_dims, self.meas_dims)


# =============================================================================
class HadamSplit2d(LinearSplit):
    r"""Simulate 2D Hadamard split acquisitions.

    Considering the acquisition of :math:`2M` square DMD patterns of size :math:`h`, it computes

    .. math::
        y =\mathcal{N}\left(\mathcal{S}\left(AXA^T\right)\right),

    where :math:`\mathcal{N} \colon\, \mathbb{R}^{2M} \to \mathbb{R}^{2M}` represents a noise operator (e.g., Gaussian), :math:`\mathcal{S} \colon\, \mathbb{R}^{2h\times 2h} \to \mathbb{R}^{2M}` is a subsampling operator, :math:`A \in \mathbb{R}_+^{2h\times h}` is the acquisition matrix that contains the positive and negative components of a Hadamard matrix, :math:`X \in \mathbb{R}^{h\times h}` is the (2D) image.


    1. The matrix :math:`A` is obtained by splitting a Hadamard matrix :math:`H\in\mathbb{R}^{h\times h}` such that :math:`A[0::2, :] = H_{+}` and :math:`A[1::2, :] = H_{-}`, where :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

    .. note::

        :math:`H_{+} - H_{-} = H`.

    2. The subsampling operator keeps the pixels that correspond to the :math:`M` largest values in the order matrix :math:`O\in\mathbb{R}^{h^2 \times h^2}`.

    .. note::

        Subsampling applies to :math:`H_{+}XH_{+}^T` and :math:`H_{-}XH_{-}^T` the same way, independently.

    .. note::
            The operator :math:`\mathcal{S}` returns a vector. In the case :math:`M=h^2` (no subsampling), :math:`\mathcal{S}` is the vectorization operator.

    Args:
        :attr:`h` (int): Image size :math:`h`. Must be a power of 2.

        :attr:`order` (:class:`torch.tensor`, optional): Order matrix :math:`O` that defines the measurements to keep. The first component of :math:`y` will correspond to the index where :attr:`order` is the highest.

        :attr:`fast` (bool, optional): Whether to use the fast Hadamard transform
        algorithm. If False, it uses matrix-vector products. Defaults to True.

        :attr:`reshape_output` (bool, optional): Whether reshape the output of adjoint and pinv methods to images. If False, output are vectors.

        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model :math:`\mathcal{N}`.
        Defaults to `torch.nn.Identity()`.

        :attr:`dtype` (:class:`torch.dtype`, optional): Data type of the measurement
        matrix. Defaults to `torch.float32`.

        :attr:`device` (:obj:`torch.device`, optional): Device of the measurement matrix.
        Defaults to `torch.device("cpu")`.

    .. note:
        The argument :attr:`order` is particularly useful when rearranging the
        measurements by decreasing variance. The variance matrix can simply be
        put as `order`.

    Attributes:

        :attr:`H` (:class:`torch.tensor`): The 2D measurement matrix given by :math:`H\otimes H`.

        :attr:`A` (:class:`torch.tensor`): The 2D acquisition matrix given by :math:`A\otimes A`.

        :attr:`M` (int): Number of measurements :math:`M`.

        :attr:`N` (int): Number of pixels in the image equal to :math:`h^2`.

        :attr:`meas_shape` (torch.Size): Shape of the measurement patterns. Is
        equal to :math:`(h, h)`.

        :attr:`meas_dims` (torch.Size): Dimensions of the image the acquisition
        matrix applies to. Is equal to `(-2, -1)`.

        :attr:`H_static` (:class:`torch.tensor`): alias for :attr:`H`.

        :attr:`H_pinv` (:class:`torch.tensor`, optional): The learnable pseudo inverse
        measurement matrix :math:`H^\dagger` of shape :math:`(N, M)`.

        :attr:`order` (:class:`torch.tensor`): Order matrix :math:`O`. It
        is used by :func:`~spyrit.core.torch.sort_by_significance()`. Defaults to rectangular order (e.g., linear indices).

        :attr:`indices` (:class:`torch.tensor`): Indices used to reorder the measurement vector. It is used by the method :meth:`reindex()`.

    Example:
        >>> import spyrit.core.meas as meas
        >>> h = 32
        >>> meas_op = meas.HadamSplit2d(h, 400)
        >>> print(meas_op.H1d.shape)
        torch.Size([32, 32])
        >>> print(meas_op.M)
        400
    """

    # H is exposed as a computed @property (see below) to avoid
    # materializing the full h**2 x h**2 measurement matrix.
    _store_H_as_parameter = False

    def __init__(
        self,
        h: int,
        M: int = None,
        order: torch.tensor = None,
        fast: bool = True,
        reshape_output: bool = False,
        *,
        noise_model=nn.Identity(),
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        meas_dims = (-2, -1)
        meas_shape = (h, h)
        if M is None:
            M = h**2

        # call Linear constructor (avoid setting A)
        super(LinearSplit, self).__init__(
            torch.empty(h**2, h**2),
            meas_shape,
            meas_dims,
            noise_model=noise_model,
            dtype=dtype,
            device=device,
        )

        if order is None:
            order = torch.ones(h, h)
        # 1D version of H
        # H1d = spytorch.walsh_matrix(h).to(dtype=dtype, device=device)
        # self.H1d = nn.Parameter(H1d, requires_grad=False)
        self.H1d = nn.Parameter(spytorch.walsh_matrix(h), requires_grad=False).to(
            dtype=dtype, device=device
        )
        self.M = M  # supercharged self.M
        self.order = order
        self.indices = torch.argsort(-order.flatten(), stable=True).to(
            dtype=torch.int32, device=self.device
        )
        self.fast = fast
        self.reshape_output = reshape_output

    @property
    def dtype(self) -> torch.dtype:
        return self.H1d.dtype

    @property
    def device(self) -> torch.device:
        return self.H1d.device

    @property
    def H(self):
        H = torch.kron(self.H1d, self.H1d)
        H = self.reindex(H, "rows", False)

        # !!!!

        return H[: self.M, :]

    @property
    def A(self):
        H = self.H
        pos, neg = nn.functional.relu(H), nn.functional.relu(-H)
        return torch.cat([pos, neg], 1).reshape(2 * self.M, self.N)

    @property
    def matrix_to_inverse(self):
        return self.H

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
            :attr:`values` (:class:`torch.tensor`): The tensor to sort. Can be 1D, 2D, or any
            multi-dimensional batch of 2D tensors.

            :attr:`axis` (str, optional): The axis to sort along. Must be either 'rows' or
            'cols'. If `values` is 1D, `axis` is not used. Default is 'rows'.

            :attr:`inverse_permutation` (bool, optional): Whether to apply the permutation
            inverse. Default is False.

        Raises:
            ValueError: If `axis` is not 'rows' or 'cols'.

        Returns:
            :class:`torch.tensor`: The sorted tensor by the given indices along the
            specified axis.
        """
        return spytorch.reindex(x, self.indices.to(x.device), axis, inverse_permutation)

    def measure(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate noiseless measurements from matrix A.

        It computes

        .. math::
            y =\mathcal{S}\left(AXA^T\right),

        where :math:`\mathcal{S} \colon\, \mathbb{R}^{2h\times 2h} \to \mathbb{R}^{2M}` is the subsampling operator, :math:`A \colon\, \mathbb{R}_+^{2h\times h}` is the acquisition matrix that contains the positive and negative component of 2D Hadamard patterns, :math:`X \in \mathbb{R}^{h\times h}` is the (2D) image, :math:`2M` is the number of DMD patterns, and :math:`h` is the image size.

        Args:
            :attr:`x` (:class:`torch.tensor`): Image :math:`X` whose
            dimensions :attr:`self.meas_dims` must have shape
            shape :attr:`self.meas_shape`.

        Returns:
            Measurement vector :math:`y \in \mathbb{R}^{2M}`.

        Examples:
            Example 1: No subsampling

            >>> import torch
            >>> import spyrit.core.meas as meas
            >>> h = 32
            >>> Ord = torch.randn(h, h)
            >>> meas_op = meas.HadamSplit2d(h)
            >>> x = torch.empty(10, h, h).uniform_(0, 1)
            >>> y = meas_op.measure(x)
            >>> print(y.shape)
            torch.Size([10, 2048])

            Example 2: With subsampling

            >>> import torch
            >>> import spyrit.core.meas as meas
            >>> h = 32
            >>> Ord = torch.randn(h, h)
            >>> meas_op = meas.HadamSplit2d(h, 49)
            >>> x = torch.empty(8, 2, h, h).uniform_(0, 1)
            >>> y = meas_op.measure_H(x)
            >>> print(y.shape)
            torch.Size([8, 2, 49])
        """
        if self.fast:
            return self.fast_measure(x)
        else:
            return super().measure(x)

    def measure_H(self, x: torch.tensor):
        r"""Simulate noiseless measurements from matrix H.

        It computes

        .. math::
            m =\mathcal{S}\left(HXH^T\right),

        where :math:`\mathcal{S} \colon\, \mathbb{R}^{h\times h} \to \mathbb{R}^{M}` is the subsampling operator, :math:`H \colon\, \mathbb{R}^{h\times h}` is the Hadamard matrix, :math:`X \in \mathbb{R}^{h\times h}` is the (2D) image.

        Args:
            :attr:`x` (:class:`torch.tensor`): Image :math:`X` whose
            dimensions :attr:`self.meas_dims` must have shape
            shape :attr:`self.meas_shape`.

        Returns:
            Measurement vector :math:`m \in \mathbb{R}^{M}`.

        Examples:
            Example 1: No subsampling

            >>> import torch
            >>> import spyrit.core.meas as meas
            >>> h = 32
            >>> meas_op = meas.HadamSplit2d(h)
            >>> x = torch.empty(h, h).uniform_(0, 1)
            >>> y = meas_op.measure(x)
            >>> print(y.shape)
            torch.Size([2048])

            Example 2: With subsampling

            >>> import torch
            >>> import spyrit.core.meas as meas
            >>> h = 32
            >>> meas_op = meas.HadamSplit2d(h, 49)
            >>> x = torch.empty(8, 2, h, h).uniform_(0, 1)
            >>> y = meas_op.measure(x)
            >>> print(y.shape)
            torch.Size([8, 2, 98])
        """
        if self.fast:
            return self.fast_measure_H(x)
        else:
            return super().measure_H(x)

    def adjoint_H(self, m: torch.tensor, unvectorize=False) -> torch.tensor:
        r"""Apply the adjoint of matrix H.

        Args:
            :attr:`m` (:class:`torch.tensor`): Measurement :math:`m` length is :attr:`self.M`.

            :attr:`unvectorize` (bool): whether to apply a :meth:`unvectorize`
            operation at the end of the computation.

        Returns:
            Vectorized image vector :math:`x \in \mathbb{R}^{h^2}`

        Examples:
            Example 1: No subsampling

            >>> import torch
            >>> import spyrit.core.meas as meas
            >>> h = 32
            >>> meas_op = meas.HadamSplit2d(h)
            >>> m = torch.empty(10, h*h).uniform_(0, 1)
            >>> x = meas_op.adjoint_H(m)
            >>> print(x.shape)
            torch.Size([10, 1024])

            Example 2: With subsampling

            >>> import torch
            >>> import spyrit.core.meas as meas
            >>> h, M = 32, 49
            >>> meas_op = meas.HadamSplit2d(h, M)
            >>> m = torch.empty(8, 2, M).uniform_(0, 1)
            >>> x = meas_op.adjoint_H(m)
            >>> print(x.shape)
            torch.Size([8, 2, 1024])
        """
        if self.fast:
            # fast_pinv takes 'vectorize' as argument
            return self.fast_pinv(m, not unvectorize) * self.N
        else:
            return super().adjoint_H(m, unvectorize)

    def fast_measure(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate noiseless measurements from matrix A."""
        Hx = self.fast_measure_H(x)
        x_sum = Hx[..., None, 0]  # indexing while keeping the original shape
        y_pos, y_neg = (x_sum + Hx) / 2, (x_sum - Hx) / 2
        new_shape = y_pos.shape[:-1] + (2 * self.M,)
        y = torch.stack([y_pos, y_neg], -1).reshape(new_shape)
        return y

    def fast_measure_H(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate noiseless measurements from matrix H."""
        x = spytorch.mult_2d_separable(self.H1d, x)
        x = self.vectorize(x)
        x = x.index_select(dim=-1, index=self.indices)
        # x = self.reindex(x, "rows", False)
        return x[..., : self.M]

    def fast_pinv(self, m: torch.tensor, vectorize=False) -> torch.tensor:
        r"""Apply the pseudo inverse of H.

        Args:
            :attr:`m` (:class:`torch.tensor`): Measurement :math:`m` of length :attr:`self.M`.

            :attr:`vectorize` (bool): Whether to apply the :meth:`vectorize` method
            after computation of the pseudo inverse.

        Returns:
            :class:`torch.tensor`: Vectorized image :math:`x` of length :attr:`self.N`.

        .. note::
            We use the separability of the 2D Hadamard transform. Only multiplications
            with the "1D" Hadamard matrix (i.e., :attr:`self.H1d`) are required. If
            the number of measurements is smaller than the number of pixels,
            the measurement vector is zero-padded.

        Examples:
            Example 1: No subsampling

            >>> import torch
            >>> import spyrit.core.meas as meas
            >>> h = 32
            >>> meas_op = meas.HadamSplit2d(h)
            >>> m = torch.empty(10, h*h).uniform_(0, 1)
            >>> x = meas_op.fast_pinv(m)
            >>> print(x.shape)
            torch.Size([10, 32, 32])

            Example 2: With subsampling

            >>> import torch
            >>> import spyrit.core.meas as meas
            >>> h, M = 32, 49
            >>> meas_op = meas.HadamSplit2d(h, M)
            >>> m = torch.empty(8, 2, M).uniform_(0, 1)
            >>> x = meas_op.fast_pinv(m)
            >>> print(x.shape)
            torch.Size([8, 2, 32, 32])

            Example 3: Output are vectors, not images

            >>> import torch
            >>> import spyrit.core.meas as meas
            >>> h, M = 32, 49
            >>> meas_op = meas.HadamSplit2d(h, M)
            >>> m = torch.empty(8, 2, M).uniform_(0, 1)
            >>> x = meas_op.fast_pinv(m, vectorize=True)
            >>> print(x.shape)
            torch.Size([8, 2, 1024])
        """
        if self.N != self.M:
            m = torch.cat(
                (m, torch.zeros(*m.shape[:-1], self.N - self.M, device=m.device)),
                -1,
            )
        m = self.reindex(m, "cols", False)
        m = self.unvectorize(m)
        m = spytorch.mult_2d_separable(self.H1d, m) / self.N

        if vectorize:
            m = self.vectorize(m)
        return m

    def fast_H_pinv(self) -> torch.tensor:
        r"""Return the pseudo inverse of the matrix H"""
        return self.H.T / self.N


# =============================================================================
class HadamSmatrix2d(Linear):
    r"""Simulate 2D S-matrix acquisitions.

    An S matrix of order :math:`n-1` can obtained from any Hadamard matrix of order :math:`n`. Therefore, we define a "2D" S-transform from a 2D Hadamard transform that can be represented by a matrix :math:`H` of order :math:`n = h^2`, such that :math:`H = H_{1d}\otimes H_{1d}` with :math:`H_{1d}` a Hadamard matrix of order :math:`h`.

    Considering the acquisition of :math:`M` DMD patterns of size :math:`h \times h = N`, the class computes [CHECK what is the correct formula here!]
    .. math::
        y =\mathcal{N}\left(\mathcal{S}\left(AXA^T\right)\right),

    where :math:`\mathcal{N} \colon\, \mathbb{R}^M \to \mathbb{R}^M` represents a noise operator (e.g., Gaussian), :math:`\mathcal{S} \colon\, \mathbb{R}^{h\times h} \to \mathbb{R}^M` is a subsampling operator, :math:`A \in \mathbb{R}_+^{h\times h}` contains negative components of a Hadamard matrix, :math:`X \in \mathbb{R}^{h\times h}` is the (2D) image. The matrix :math:`A` is obtained as :math:`A = \max(0,-H)`, where :math:`H\in\mathbb{R}^{h\times h}` is a Hadamard matrix.

    **From Hadamard to S-matrix.** Let :math:`H_{1d}` be a (normalized)
    Hadamard matrix of order :math:`n+1` and :math:`J` the all-ones matrix
    of the same order. The central relationship between the Hadamard
    matrix and the S-matrix is

    .. math::
        R_{1d} = \max(0,\, -H_{1d}) = \frac{J - H_{1d}}{2}
        = \begin{pmatrix} 0 & 0 \\ 0 & S \end{pmatrix},

    where :math:`R_{1d}` is the **negative component** of :math:`H_{1d}`.
    Because :math:`H_{1d}`'s first row and column are both all-ones (the
    usual normalized/Walsh-ordering convention), :math:`R_{1d}`'s first
    row and column are *entirely* zero, and its remaining :math:`n\times
    n` bottom-right block is exactly the S-matrix :math:`S`. Computing the
    S-transform therefore reduces to computing a Hadamard transform:
    whenever the Hadamard transform can be computed fast at order
    :math:`n+1` (i.e. :math:`n+1` is a power of two), the S-transform can
    also be computed fast at order :math:`n` -- by embedding :math:`f` as
    :math:`[0,f]`, taking the negative component of its Hadamard
    transform, and dropping the (always 0) leading entry.

    This class plays the same role as :class:`HadamSplit2d`, but only
    returns the **negative** component of the 2D Hadamard split, as a
    standalone (non-interleaved) 0/1-valued measurement operator -- so it
    is directly realizable as a set of DMD patterns without any
    "unsplitting" step downstream. To get a *fast* transform when
    :math:`f` is a square :math:`h\times h` image :math:`X`, we implement
    the 2D transform in a similar manner as a **negative 2D Hadamard
    transform**, applied directly at order :math:`h` (the image size)
    rather than at order :math:`n+1` with :math:`n=h-1`. This keeps
    :math:`h` itself (rather than :math:`h+1`) a power of two -- more
    convenient for square images, e.g. :math:`h=64` -- and computation
    times remain short by exploiting the separability across rows and
    columns, exactly as :class:`HadamSplit2d` does. The price is that this
    class computes the full order-:math:`h` negative-component matrix
    (the 2D extension of :math:`R_{1d}`), not the trimmed order-
    :math:`(h-1)` S-matrix :math:`S`: it keeps the trivial all-zero
    row/column instead of dropping them (see the warning below).

    :math:`H` is built exactly as :meth:`HadamSplit2d.fast_measure` builds
    the negative component of its 2D Hadamard split, i.e. from
    :math:`H_{1d} = \texttt{spyrit.core.torch.walsh\_matrix}(h) \in
    \{-1,+1\}^{h\times h}` -- the same 1D Walsh-ordered Hadamard matrix
    used by :class:`HadamSplit2d` (see :attr:`HadamSplit2d.H1d`):

    .. math::
        H = \max\left(0,\, -\left(H_{1d}\otimes H_{1d}\right)\right)
        = \frac{J - H_{1d}\otimes H_{1d}}{2},

    where :math:`J` is now the :math:`h^2\times h^2` all-ones matrix (the
    2D counterpart of the :math:`R_{1d} = (J-H_{1d})/2` relationship
    above -- note that :math:`H` is *not* :math:`R_{1d}\otimes R_{1d}`,
    since the negative-component split does not distribute over the
    Kronecker product). Because :math:`H_{1d}\otimes H_{1d}` only takes
    values in :math:`\{-1,+1\}`,
    :math:`H` only takes values in :math:`\{0,1\}`, i.e. :math:`H` is a
    genuine 2D S-matrix -- unlike simply taking the Kronecker product of
    two 1D S-matrices, since :math:`\mathrm{relu}(-a)\,\mathrm{relu}(-b)
    \neq \mathrm{relu}(-ab)` in general.

    Since :math:`H_{1d}` is a genuine :math:`h\times h` Hadamard matrix,
    :math:`h` itself (e.g. :math:`h=64`) must be a power of two -- unlike
    a previous implementation of this class, which instead required
    :math:`h+1` to be a power of two.

    .. warning::
        Because :math:`H_{1d}`'s first row and column are both all-ones
        (the usual Walsh-ordering convention), :math:`H` has an all-zero
        row *and* an all-zero column -- the 2D counterpart of
        :math:`R_{1d}`'s :math:`\begin{pmatrix}0&0\\0&S\end{pmatrix}`
        block structure described above:

        - :math:`H`'s all-zero row means one measurement is trivially
          always zero: at natural-order index 0 (the position
          :attr:`order` keeps first, by default), :math:`y` always equals
          exactly 0, regardless of :math:`X` -- this is the "all DMD
          micromirrors off" pattern.

        - :math:`H`'s all-zero column means one pixel of :math:`X` --
          namely :math:`X[k,k]`, where :math:`k` is :attr:`self.zero_index`
          (equal to 0 unless :attr:`scramble` is True) -- has *no* effect
          whatsoever on any measurement, and is therefore fundamentally
          unrecoverable. Following the minimum-norm convention,
          :meth:`fast_pinv` reconstructs that pixel as exactly 0 (see the
          note there).

    .. note::
        :attr:`order` and :attr:`scramble` both rely on permutations, but
        they act on different things and serve different purposes -- do
        not confuse them:

        - :attr:`order` permutes the **rows** of the (subsampled) 2D
          system matrix :math:`H`, i.e. it reorders the :math:`M`
          **measurements** in the output vector :math:`y` (and selects
          which :math:`M` of the :math:`h^2` possible measurements are
          kept, if :math:`M<h^2`). It does not change what any individual
          measurement pattern looks like, only the sequence in which the
          measurements appear (e.g. by decreasing variance/significance,
          via :attr:`order`).

        - :attr:`scramble` permutes the **columns** of :math:`H_{1d}` --
          i.e. of the (1D) matrix used to build the acquisition matrix,
          before any row reordering/subsampling happens. This changes
          which pixels of the image each individual measurement pattern
          probes (and which pixel becomes unrecoverable, see
          :attr:`zero_index`), not the order in which measurements are
          returned.

        In short: :attr:`scramble` acts on :math:`H_{1d}` (columns),
        :attr:`order` acts on the sequence of measurements in :math:`y`
        (rows of :math:`H`, built from :math:`H_{1d}` after scrambling has
        already been applied, if any). The two options are independent and
        can be combined.

    If :attr:`scramble` is True, the columns of :math:`H_{1d}` are randomly
    permuted (with a fixed :attr:`seed` for reproducibility). This is useful
    e.g. to decorrelate the acquisition order from the natural Walsh
    ordering.

    .. note::
        :math:`H` is not orthogonal, so the adjoint (transpose) of the
        measurement operator and its pseudo-inverse are genuinely different
        operators here. Both, however, admit a closed form expressed only
        in terms of :math:`H_{1d}` (and its transpose), so both remain
        fast and separable across rows and columns -- no :math:`h^2\times
        h^2` matrix and no explicit matrix inversion is ever needed, even
        when :attr:`scramble` is True (see :meth:`fast_adjoint` and
        :meth:`fast_pinv` for the exact formulas).

    Args:
        :attr:`h` (int): Image size :math:`h`, which must be a power of 2.

        :attr:`M` (int, optional): Number of measurements. Defaults to
        :math:`h^2` (no subsampling).

        :attr:`order` (:class:`torch.tensor`, optional): Order matrix
        :math:`O` that defines the measurements to keep. The first
        component of :math:`y` will correspond to the index where
        :attr:`order` is the highest. Permutes the **rows** of the system
        matrix :math:`H` (i.e. the sequence of measurements in :math:`y`).
        Not to be confused with :attr:`scramble`, which permutes the
        **columns** of :math:`H_{1d}`.

        :attr:`fast` (bool, optional): Whether to use the fast, separable
        computation of the 2D S-transform. If False, it uses (memory-heavy)
        matrix-vector products with the full measurement matrix. Defaults
        to True.

        :attr:`reshape_output` (bool, optional): Whether to reshape the
        output of the adjoint and pseudo-inverse methods to images. If
        False, outputs are vectors.

        :attr:`scramble` (bool, optional): If True, the **columns** of
        :math:`H_{1d}` are randomly permuted. Defaults to False. Not to be
        confused with :attr:`order`, which permutes the **rows** of the
        system matrix :math:`H` (i.e. the sequence of measurements in
        :math:`y`), applied after scrambling.

        :attr:`seed` (int, optional): Seed used to generate the random
        column permutation when :attr:`scramble` is True, ensuring
        reproducibility. Defaults to 0.

        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model
        :math:`\mathcal{N}`. Defaults to `torch.nn.Identity()`.

        :attr:`dtype` (:class:`torch.dtype`, optional): Data type of the
        measurement matrix. Defaults to `torch.float32`.

        :attr:`device` (:obj:`torch.device`, optional): Device of the
        measurement matrix. Defaults to `torch.device("cpu")`.

    Attributes:
        :attr:`H1d` (:class:`torch.tensor`): 1D Walsh-ordered Hadamard
        matrix of shape :math:`(h,h)`, values in :math:`\{-1,+1\}` (the
        same matrix as :attr:`HadamSplit2d.H1d`). Its **columns** are
        permuted if :attr:`scramble` is True (see :attr:`column_perm`).

        :attr:`column_perm` (:class:`torch.tensor`, optional): The random
        column permutation applied to :math:`H_{1d}`. Only set as an
        attribute when :attr:`scramble` is True.

        :attr:`zero_index` (int): Row/column index :math:`k` of the pixel
        :math:`X[k,k]` that :math:`H` cannot measure (see the warning
        above). Equal to 0 unless :attr:`scramble` is True, in which case
        it is derived from :attr:`column_perm`.

        :attr:`h` (int): Image size :math:`h`.

        :attr:`H` (:class:`torch.tensor`): The 2D measurement matrix given
        by :math:`\max(0, -(H_{1d}\otimes H_{1d}))`, subsampled to
        :attr:`self.M` rows. Computed on the fly (not stored) to avoid
        materializing a :math:`h^2 \times h^2` matrix.

        :attr:`M` (int): Number of measurements :math:`M`.

        :attr:`N` (int): Number of pixels in the image, equal to :math:`h^2`.

        :attr:`meas_shape` (torch.Size): Shape of the measurement patterns.
        Equal to :math:`(h, h)`.

        :attr:`meas_dims` (torch.Size): Dimensions of the image the
        acquisition matrix applies to. Equal to `(-2, -1)`.

        :attr:`order` (:class:`torch.tensor`): Order matrix :math:`O`. Only
        affects the **row** order (sequence) of the measurements in
        :math:`y`; unrelated to :attr:`scramble`, which affects the
        **columns** of :math:`H_{1d}` instead.

        :attr:`indices` (:class:`torch.tensor`): Indices used to reorder
        the measurement vector (derived from :attr:`order`). Used by the
        method :meth:`reindex`.

    Example 1: 
        Basic construction and simulated (subsampled) measurements.
        The very first measurement is always (up to floating-point error) 0,
        since it corresponds to the trivial all-zero row of :math:`H` (see the
        warning above).

        >>> h = 64  # h must be a power of two
        >>> meas_op = HadamSmatrix2d(h, 2000)
        >>> print(meas_op.H1d.shape)
        torch.Size([64, 64])
        >>> print(meas_op.M)
        2000
        >>> x = torch.rand(4, h, h)
        >>> y = meas_op(x)
        >>> print(y.shape)
        torch.Size([4, 2000])
        >>> print(torch.allclose(y[:, 0], torch.zeros(4), atol=1e-2))
        True

    Example 2: 
        With full sampling (:attr:`M` = :math:`h^2`, the default),
        :meth:`fast_pinv` recovers the image exactly, except for the single
        unrecoverable pixel :attr:`zero_index`, which is reconstructed as 0
        (see the warning above and the note in :meth:`fast_pinv`).

        >>> h = 16
        >>> meas_op = HadamSmatrix2d(h)  # M defaults to h**2 (full sampling)
        >>> print(meas_op.M == meas_op.N)
        True
        >>> print(meas_op.zero_index)
        0
        >>> x = torch.rand(2, h, h)
        >>> y = meas_op.measure(x)
        >>> x_hat = meas_op.fast_pinv(y, vectorize=False)
        >>> x_expected = x.clone()
        >>> x_expected[:, 0, 0] = 0
        >>> print(torch.allclose(x_expected, x_hat, atol=1e-4))
        True

    Example 3: 
        With subsampling (:attr:`M` < :math:`h^2`), :meth:`fast_pinv`
        only approximates the image (see the note in :meth:`fast_pinv`), unlike
        the exact recovery obtained above with full sampling.

        >>> meas_op_sub = HadamSmatrix2d(h, M=100)
        >>> y_sub = meas_op_sub.measure(x)
        >>> x_hat_sub = meas_op_sub.fast_pinv(y_sub, vectorize=False)
        >>> print(torch.allclose(x, x_hat_sub, atol=1e-4))
        False

    Example 4: 
        :attr:`scramble` permutes the columns of :attr:`H1d`,
        breaking its symmetry and moving :attr:`zero_index` away from 0, but
        full-sampling recovery via :meth:`fast_pinv` remains exact everywhere
        except at :attr:`zero_index`.

        >>> meas_op_scrambled = HadamSmatrix2d(h, scramble=True, seed=42)
        >>> print(torch.allclose(meas_op.H1d, meas_op.H1d.T))       # unscrambled: symmetric
        True
        >>> print(torch.allclose(meas_op_scrambled.H1d, meas_op_scrambled.H1d.T))  # scrambled: not symmetric
        False
        >>> print(meas_op_scrambled.zero_index != 0)
        True
        >>> y2 = meas_op_scrambled.measure(x)
        >>> x_hat2 = meas_op_scrambled.fast_pinv(y2, vectorize=False)
        >>> k = meas_op_scrambled.zero_index
        >>> x_expected2 = x.clone()
        >>> x_expected2[:, k, k] = 0
        >>> print(torch.allclose(x_expected2, x_hat2, atol=1e-4))
        True

    Example 5: 
        :attr:`order` and :attr:`scramble` act independently (see
        note above): changing :attr:`order` selects/reorders which measurements
        are kept, but does not affect :attr:`H1d` itself.

        >>> order = torch.rand(h, h)
        >>> meas_op_ordered = HadamSmatrix2d(h, M=100, order=order)
        >>> print(torch.equal(meas_op_ordered.H1d, meas_op.H1d))
        True
    """

    # H is exposed as a computed @property (see below) to avoid
    # materializing the full h**2 x h**2 measurement matrix.
    _store_H_as_parameter = False

    def __init__(
        self,
        h: int,
        M: int = None,
        order: torch.tensor = None,
        fast: bool = True,
        reshape_output: bool = False,
        scramble: bool = False,
        seed: int = 0,
        *,
        noise_model=nn.Identity(),
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        meas_dims = (-2, -1)
        meas_shape = (h, h)
        if M is None:
            M = h**2

        # call Linear constructor (avoid setting H as a Parameter, see
        # Linear._store_H_as_parameter)
        super().__init__(
            torch.empty(h**2, h**2),
            meas_shape,
            meas_dims,
            noise_model=noise_model,
            dtype=dtype,
            device=device,
        )

        if order is None:
            order = torch.ones(h, h)

        # 1D Walsh-ordered Hadamard matrix, same as HadamSplit2d.H1d.
        # spytorch.walsh_matrix raises if h is not a power of two.
        H1d = spytorch.walsh_matrix(h)

        self.h = h
        self.scramble = scramble
        self.seed = seed
        self.zero_index = 0
        if scramble:
            # Randomly permute the columns of H1d. A fixed seed guarantees
            # the same permutation is produced every time, for
            # reproducibility. H1d_scrambled[:, j] = H1d[:, column_perm[j]].
            generator = torch.Generator().manual_seed(seed)
            self.column_perm = torch.randperm(h, generator=generator)
            H1d = H1d[:, self.column_perm]
            # The unrecoverable pixel (see class docstring) moves from 0 to
            # wherever column 0 of the unscrambled matrix was sent.
            self.zero_index = int((self.column_perm == 0).nonzero())

        self.H1d = nn.Parameter(H1d, requires_grad=False).to(dtype=dtype, device=device)

        self.M = M  # supercharged self.M
        self.order = order
        self.indices = torch.argsort(-order.flatten(), stable=True).to(
            dtype=torch.int32, device=self.device
        )
        self.fast = fast
        self.reshape_output = reshape_output

    @property
    def dtype(self) -> torch.dtype:
        return self.H1d.dtype

    @property
    def device(self) -> torch.device:
        return self.H1d.device

    @property
    def H(self):
        r"""The full (subsampled) 2D S measurement matrix, computed on the
        fly as :math:`\max(0, -(H_{1d}\otimes H_{1d}))`, reindexed and
        truncated to the first :attr:`self.M` rows (by decreasing
        :attr:`self.order`)."""
        H2D = torch.kron(self.H1d, self.H1d)
        H = nn.functional.relu(-H2D)
        H = self.reindex(H, "rows", False)
        return H[: self.M, :]

    @property
    def matrix_to_inverse(self):
        return self.H

    def reindex(
        self, x: torch.tensor, axis: str = "rows", inverse_permutation: bool = False
    ) -> torch.tensor:
        """Sorts a tensor along a specified axis using :attr:`self.indices`.
        See :meth:`HadamSplit2d.reindex` for details."""
        return spytorch.reindex(x, self.indices.to(x.device), axis, inverse_permutation)

    def measure(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate noiseless measurements.

        It computes :math:`y = \mathcal{S}(\mathrm{vec}^{-1}(H\,\mathrm{vec}(X)))`.

        Args:
            :attr:`x` (:class:`torch.tensor`): Image :math:`X` whose
            dimensions :attr:`self.meas_dims` must have shape
            :attr:`self.meas_shape`.

        Returns:
            Measurement vector :math:`y \in \mathbb{R}^{M}`.
        """
        if self.fast:
            return self.fast_measure(x)
        else:
            return super().measure(x)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate noisy measurements :math:`y = \mathcal{N}(\mathcal{S}(\mathrm{vec}^{-1}(H\,\mathrm{vec}(X))))`."""
        x = self.measure(x)
        x = self.noise_model(x)
        return x

    def fast_measure(self, x: torch.tensor) -> torch.tensor:
        r"""Simulate noiseless measurements using the separability of the
        2D S-transform.

        :math:`H` is the negative component of the 2D Hadamard split built
        from :math:`H_{1d}\otimes H_{1d}` -- exactly like
        :meth:`HadamSplit2d.fast_measure` computes its own "y_neg" half,
        but returned here on its own (not interleaved with the positive
        component). Since :math:`H = (J - H_{1d}\otimes H_{1d})/2` and
        :math:`J\,\mathrm{vec}(X) = \mathrm{sum}(X)\cdot\mathbf{1}`, this
        only requires a global sum of :math:`X` and multiplications with
        the 1D matrix :attr:`self.H1d` (applied to both the rows and the
        columns of :math:`X`).

        .. note::
            :math:`\mathrm{sum}(X)` is read off :math:`Hx[0,0]` (the DC
            term of the transform) rather than computed independently via
            a separate reduction over :math:`X`. Both are mathematically
            equal, but computing them independently (via two different
            floating-point code paths -- a direct sum vs. a chain of
            matrix products) does not generally give bit-identical
            results; the resulting (tiny) discrepancy would otherwise leak
            into every entry of :math:`y` as a uniform bias, occasionally
            making an entry that should be a small positive number (or
            exactly 0) come out slightly negative. Reading :math:`x_{sum}`
            off :math:`Hx` itself avoids this, exactly as
            :meth:`HadamSplit2d.fast_measure` does."""
        Hx = spytorch.mult_2d_separable(self.H1d, x)
        s = Hx[..., 0:1, 0:1]
        y = (s - Hx) / 2
        y = self.vectorize(y)
        y = y.index_select(dim=-1, index=self.indices)
        return y[..., : self.M]

    def adjoint(self, m: torch.tensor, unvectorize: bool = False) -> torch.tensor:
        r"""Apply the adjoint (transpose) of the measurement matrix.

        It computes :math:`x = H^Tm`, where :math:`H` is the (subsampled)
        2D S measurement matrix.

        .. note::
            This is the literal adjoint, not the pseudo-inverse: use
            :meth:`fast_pinv` for the pseudo-inverse solution. Unlike
            :class:`HadamSplit2d`, this does **not** assume :math:`H_{1d}`
            is symmetric (it is not, when :attr:`scramble` is True), and
            explicitly applies :math:`H_{1d}^T`.

        Args:
            :attr:`m` (:class:`torch.tensor`): Measurement :math:`m` of
            length :attr:`self.M`.

            :attr:`unvectorize` (bool): whether to apply
            :meth:`~spyrit.core.meas.Linear.unvectorize` at the end of the
            computation.

        Returns:
            A batch of signals :math:`x`.
        """
        if self.fast:
            return self.fast_adjoint(m, unvectorize)
        else:
            return super().adjoint(m, unvectorize)

    def fast_adjoint(self, m: torch.tensor, unvectorize: bool = False) -> torch.tensor:
        r"""Apply the adjoint of the measurement matrix using the
        separability of the 2D S-transform.

        The forward map is :math:`H = (J - H_{1d}\otimes H_{1d})/2`, and
        :math:`J` and (elementwise) division are self-adjoint, so
        :math:`H^T = (J - H_{1d}^T\otimes H_{1d}^T)/2`: the same "sum
        trick" as :meth:`fast_measure` applies, using :math:`H_{1d}^T`
        (not :math:`H_{1d}`) on both axes. When :attr:`scramble` is False,
        :math:`H_{1d}` is symmetric and this is equivalent to applying
        :math:`H_{1d}` directly.

        .. note::
            As in :meth:`fast_measure`, :math:`\mathrm{sum}(m)` is read
            off the transform :math:`H_{1d}^Tm H_{1d}^T` itself rather
            than computed independently, to avoid floating-point
            cancellation noise. Here, however, the relevant entry sits at
            :attr:`self.zero_index` (not always index 0): applying
            :math:`H_{1d}^T` to both axes means the row/column that is
            all-ones shifts from 0 to :attr:`self.zero_index` whenever
            :attr:`scramble` is True (see the class docstring). Reading
            :math:`\mathrm{sum}(m)` off that exact entry also guarantees,
            by construction (not merely numerically), that the adjoint's
            output is exactly 0 at :math:`(k,k)`, :math:`k` =
            :attr:`self.zero_index` -- unlike :meth:`fast_pinv`, no manual
            zeroing is needed here.
        """
        if self.N != self.M:
            m = torch.cat(
                (m, torch.zeros(*m.shape[:-1], self.N - self.M, device=m.device, dtype=m.dtype)),
                -1,
            )
        m = self.reindex(m, "cols", False)
        m = self.unvectorize(m)
        Htm = spytorch.mult_2d_separable(self.H1d.T, m)
        k = self.zero_index
        s = Htm[..., k : k + 1, k : k + 1]
        m = (s - Htm) / 2
        if not unvectorize:
            m = self.vectorize(m)
        return m

    def fast_pinv(self, m: torch.tensor, vectorize: bool = False) -> torch.tensor:
        r"""Apply the pseudo-inverse of the measurement matrix.

        Args:
            :attr:`m` (:class:`torch.tensor`): Measurement :math:`m` of
            length :attr:`self.M`.

            :attr:`vectorize` (bool): Whether to apply
            :meth:`~spyrit.core.meas.Linear.vectorize` after computation of
            the pseudo-inverse.

        Returns:
            :class:`torch.tensor`: Vectorized (or image-shaped) signal
            :math:`x` of length :attr:`self.N`.

        .. note::
            Since :math:`H` plays the role of the 2D extension of
            :math:`R_{1d}` (see the class docstring), and
            :math:`R_{1d}=\begin{pmatrix}0&0\\0&S\end{pmatrix}` is exactly
            block-diagonal, its pseudo-inverse is also exactly
            block-diagonal:

            .. math::
                R_{1d}^+ = \begin{pmatrix} 0 & 0 \\ 0 & S^{-1} \end{pmatrix}.

            Applied to a measurement :math:`m` (padded with a leading 0),
            this pseudo-inverse can be computed *without* ever forming
            :math:`S^{-1}` explicitly, directly from the (full,
            untrimmed) Hadamard matrix :math:`H_{1d}`:

            .. math::
                R_{1d}^+ m = -\frac{2}{n+1}\, H_{1d}^T m,

            which is exact at every entry except the leading one (always
            0 for :math:`m` in the range of :math:`R_{1d}`, but not
            automatically 0 in this formula -- see below). The 2D
            measurement matrix :math:`H` used by this class is built the
            same way, so the same identity holds with :math:`H_{1d}^T Y
            H_{1d}` in place of :math:`H_{1d}^T m` (:math:`Y` being the
            unvectorized, zero-padded measurement) and :math:`h^2=N`
            (image size) in place of :math:`n+1`:

            .. math::
                X = -\frac{2}{N}\, H_{1d}^T\, Y\, H_{1d}.

            The computation of :math:`H_{1d}^T\,Y\,H_{1d}` exploits the
            separability of the 2D transform (only multiplications with
            the 1D matrix :math:`H_{1d}`, applied to the rows and columns
            of :math:`Y`, are required -- no matrix inversion, and no
            :math:`h^2\times h^2` matrix). If the number of measurements is
            smaller than the number of pixels, the measurement vector is
            zero-padded before inversion, exactly as done in
            :meth:`HadamSplit2d.fast_pinv`.

        .. note::
            Applying :meth:`fast_pinv` to the (noiseless, full-sampling)
            measurement of an image returns that same image back
            *exactly*, at every pixel **except** :math:`X[k,k]`
            (:math:`k` = :attr:`self.zero_index`, see the warning in
            the class docstring): that single pixel is fundamentally
            unrecoverable (it never affects any measurement), so this
            method arbitrarily sets it to 0 -- the block-diagonal
            :math:`R_{1d}^+` above shows this convention is exact, not
            an approximation: :math:`X[k,k]` is genuinely a free
            parameter of the formula above, which the top-left 0
            block of :math:`R_{1d}^+` fixes at 0 (the minimum-norm,
            Moore-Penrose choice). Note that :func:`torch.linalg.pinv`
            applied directly to :attr:`self.H` is *not* a reliable way
            to check this: :math:`H`'s highly degenerate singular
            spectrum (many repeated singular values) makes generic
            SVD-based pinv numerically unstable here, unlike this
            closed-form expression.

            Because :math:`H` is rank-deficient even without subsampling,
            this reconstruction is exact (up to :math:`X[k,k]`) only when
            :attr:`self.M` equals :attr:`self.N`; with subsampling, it is
            an approximation, consistent with the convention used
            throughout :class:`HadamSplit2d`. This holds whether or not
            :attr:`scramble` is used.
        """
        if self.N != self.M:
            m = torch.cat(
                (m, torch.zeros(*m.shape[:-1], self.N - self.M, device=m.device, dtype=m.dtype)),
                -1,
            )
        m = self.reindex(m, "cols", False)
        Y = self.unvectorize(m)

        Z = spytorch.mult_2d_separable(self.H1d.T, Y)
        X = -2 * Z / self.N
        # X[..., zero_index, zero_index] can never be recovered from Y (see
        # the note above): set it to 0, the minimum-norm convention.
        X[..., self.zero_index, self.zero_index] = 0

        if vectorize:
            X = self.vectorize(X)
        return X


# =============================================================================
class DynamicLinear(Linear):
    r"""Simulates linear measurements of a moving scene

    .. math::
        m = \mathcal{N}\left( \text{diag}(H x_{t=1, ..., M})\right),

    where :math:`H\in\mathbb{R}^{M \times N}` is the acquisition matrix,
    :math:`x_{t=1, ..., M} \in \mathbb{R}^{N \times M}` is the temporal signal of interest,
    :math:`M` is both the number of measurements and the number of frames,
    :math:`N` is the dimension of the signal within the field of view,
    :math:`\text{diag}\colon\, \mathbb{R}^{M \times M} \to \mathbb{R}^M` extracts the diagonal of its input, and
    :math:`\mathcal{N} \colon\, \mathbb{R}^M \to \mathbb{R}^M` represents a noise operator (e.g., Gaussian).


    .. warning::
        The current implementation only supports 2D spatial dimensions (i.e., images).
        Consequently, meas_shape and img_shape must be tuples of two integers.

    .. warning::
        For each call, there must be **exactly** as many frames in :math:`x` as
        there are measurements in the linear operator used to initialize the class.

    Args:
        :attr:`H` (:class:`torch.tensor`): measurement matrix (linear operator)
        with shape :math:`(M, N)`. Only real values are supported.

        :attr:`time_dim` (int): dimension index in the input tensor :math:`x` that
        corresponds to time (i.e., the frames dimension).

        :attr:`meas_shape` (tuple, optional): Shape of the measurement patterns.
        Must be a tuple of two integers representing the height and width of the
        patterns. If not specified, the shape is suppposed to be a square image.
        If not, an error is raised. Defaults to None.

        :attr:`meas_dims` (tuple, optional): Dimensions of :math:`x_{t=1, ..., M}` the
        acquisition matrix applies to. Must be a tuple with the same length as
        :attr:`meas_shape`. If not, an error is raised. Defaults to the last
        dimensions of the multi-dimensional array :math:`x_{t=1, ..., M}` (e.g., `(-2,-1)`
        when `len(meas_shape)=2`).

        :attr:`img_shape` (tuple, optional): Shape of the image. Must be a tuple
        of two integers representing the height and width of the image. If not
        specified, the shape is taken as equal to `meas_shape`. Setting this
        value is particularly useful when using an extended field of view [Maitre2024_2]_.

        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model :math:`\mathcal{N}`.
        Defaults to `torch.nn.Identity()`.

        :attr:`white_acq` (torch.tensor, optional): Eventual spatial gain resulting from
        detector inhomogeneities and used for dynamic flat-field correction. It can be
        determined from a "white acquisition" without any object. If None, no correction is
        applied. Must have :attr:`self.meas_shape` shape.

    Attributes:
        :attr:`M` (int): Number of measurements.

        :attr:`N` (int): Number of pixels in the field of view.

        :attr:`L` (int): Number of pixels in the extended field of view.

        :attr:`meas_shape` (tuple): Shape of the underlying multi-dimensional
        array :math:`x` over the field of view.

        :attr:`img_shape` (tuple): Shape of the underlying multi-dimensional
        array :math:`x` over the extended field of view.

        :attr:`H` (:class:`torch.tensor`): Static measurement matrix of shape
        :math:`(M, N)` initialized as :math:`H`.

        :attr:`H_dyn` (torch.tensor): Dynamic measurement matrix :math:`H_{\rm{dyn}}` of shape.
        :math:`(M, L)`. Must be set using the method :meth:`build_dynamic_forward` before being accessed.


    Example:
        >>> import torch
        >>> from spyrit.core.meas import DynamicLinear
        >>>
        >>> x = torch.rand([1, 400, 3, 50, 50])  # dummy RGB video with 400 frames of size 50x50
        >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
        >>> meas_op = DynamicLinear(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50))
        >>> print(meas_op)
        DynamicLinear(
          (noise_model): Identity()
        )

    References:
        [Maitre2024_2]_ Maitre, T., Bretin, E., Phan, R., Ducros, N., & Sdika, M. (2024, October).
        Dynamic single-pixel imaging on an extended field of view without warping the patterns. In International
        Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 275-284).
        Cham: Springer Nature Switzerland. DOI: 10.1007/978-3-031-72104-5_27

        [Maitre2026]_ (Submitted to TIP) Maitre, T., Bretin, E., Mahieu-Williame, L., Phan, R., Sdika, M., & Ducros, N. (2025).
        Dual-arm motion-compensated single-pixel imaging. HAL Id: hal-05068181

    """

    def __init__(
        self,
        H: torch.tensor,
        time_dim: int,
        meas_shape: Union[int, torch.Size, Iterable[int]] = None,
        meas_dims: Union[int, torch.Size, Iterable[int]] = None,
        img_shape: Union[int, torch.Size, Iterable[int]] = None,
        *,
        noise_model: nn.Module = nn.Identity(),
        white_acq: torch.tensor = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            H,
            meas_shape,
            meas_dims,
            noise_model=noise_model,
            dtype=dtype,
            device=device,
        )

        self.time_dim = time_dim
        self.white_acq = white_acq
        if self.time_dim in self.meas_dims:
            raise RuntimeError(
                f"The time dimension must not be in the measurement dimensions. Found {self.time_dim} in {self.meas_dims}."
            )

        if len(self.meas_shape) != 2:
            raise NotImplementedError(
                "Currently only 2D spatial dimensions are supported."
            )

        self.img_shape = img_shape if img_shape is not None else self.meas_shape
        self.img_h, self.img_w = self.img_shape  # for legacy
        # self.h, self.w = self.meas_shape  # for legacy
        self.N = int(torch.prod(torch.tensor(self.meas_shape)))
        self.L = int(torch.prod(torch.tensor(self.img_shape)))

        # define the available matrices for reconstruction
        self._available_pinv_matrices = ["H_dyn"]
        self._selected_pinv_matrix = "H_dyn"  # select default here

    @property
    def recon_mode(self) -> str:
        """Interpolation mode used for reconstruction."""
        return self._recon_mode

    @property
    def H_dyn(self) -> torch.tensor:
        """Dynamic measurement matrix H_dyn."""
        try:
            return self._param_H_dyn.data
        except AttributeError as e:
            raise AttributeError(
                "The dynamic measurement matrix H_dyn has not been set yet. "
                + "Please call build_dynamic_forward() before accessing the attribute H_dyn."
            ) from e

    def measure(self, x):
        r"""Simulates noiseless measurements.

        .. math::
            m = \text{diag}(H x_{t=1, ..., M}),

        where :math:`H \in \mathbb{R}^{M \times N}` is the acquisition matrix,
        :math:`x_{t=1, ..., M} \in \mathbb{R}^{N \times M}` is the temporal signal of interest,
        :math:`M` is both the number of measurements and frames,
        :math:`N` is the dimension of the signal in the field of view, and
        :math:`\text{diag}\colon\, \mathbb{R}^{M \times M} \to \mathbb{R}^M` extracts the diagonal of its input.

        .. note::
            This method does not degrade measurement with noise.
            To do so, see :func:`~spyrit.core.meas.DynamicLinear.forward()`

        Args:
            :attr:`x` (:class:`torch.tensor`): A batch of temporal signals whose time
            dimension matches :attr:`self.time_dim`, and measured dimensions matches :attr:`self.meas_dims`.

        Returns:
            :class:`torch.tensor`: A batch of measurement of shape :math:`(*, M)` where * denotes
            all the dimensions of the input tensor that are not included in :attr:`self.meas_dims`.

        Example:
            >>> import torch
            >>> from spyrit.core.meas import DynamicLinear
            >>> from spyrit.core.noise import Poisson
            >>>
            >>> x = torch.rand([1, 400, 3, 50, 50])  # dummy RGB video with 400 frames of size 50x50
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinear(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinear(
              (noise_model): Poisson()
            )
            >>> m = meas_op.measure(x)  # simulate noiseless dynamic measurements
            >>> print(m.shape)
            torch.Size([1, 3, 400])

        """
        x = spytorch.center_crop(x, self.meas_shape)
        # vectorize with the time dimension being the second-to-last dimension
        x = self.vectorize(x)
        # here index m is the number of mesurements, it is also the number of frames, ie. time dimension
        x = torch.einsum("mn,...mn->...m", self.H, x)
        return x

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noisy dynamic measurements.

        .. math::
            m = \mathcal{N}\left(\text{diag}(H x_{t=1, ..., M})\right),

        where :math:`H \in \mathbb{R}^{M \times N}` is the acquisition matrix,
        :math:`x_{t=1, ..., M} \in \mathbb{R}^{N \times M}` is the temporal signal of interest,
        :math:`M` is both the number of measurements and frames,
        :math:`N` is the dimension of the signal in the field of view, and
        :math:`\text{diag}\colon\, \mathbb{R}^{M \times M} \to \mathbb{R}^M` extracts the diagonal of its input.

        .. note::
            This method degrades measurements with noise.
            To compute :math:`Hx_{t=1, ..., M}` only, see :func:`~spyrit.core.meas.DynamicLinear.measure()`.

        Args:
            :attr:`x` (:class:`torch.tensor`): A batch of temporal signals whose time
            dimension matches :attr:`self.time_dim`, and measured dimensions matches :attr:`self.meas_dims`.

        Returns:
            :class:`torch.tensor`: A batch of measurement of shape :math:`(*, M)` where * denotes
            all the dimensions of the input tensor that are not included in :attr:`self.meas_dims`.

        Example:
            >>> import torch
            >>> from spyrit.core.meas import DynamicLinear
            >>> from spyrit.core.noise import Poisson
            >>>
            >>> x = torch.rand([1, 400, 3, 50, 50])  # dummy RGB video with 400 frames of size 50x50
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinear(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinear(
              (noise_model): Poisson()
            )
            >>>
            >>> m = meas_op(x)  # simulate noisy dynamic measurements
            >>> print(m.shape)
            torch.Size([1, 3, 400])

        """
        x = self.measure(x)
        x = self.noise_model(x)
        return x

    def build_dynamic_forward(
        self,
        motion: DeformationField,
        mode: str = "bilinear",
        warping: str = "image",
        verbose: bool = False,
    ) -> None:
        r"""Builds the dynamic forward operator :math:`H_{\rm{dyn}}`.

        .. math::
            \text{diag}(H x_{t=1, ..., M}) = H_{\rm{dyn}} x,

        where
        :math:`x_{t=1, ..., M} \in \mathbb{R}^{N \times M}` is the temporal signal of interest,
        :math:`H \in \mathbb{R}^{M \times N}` is the static acquisition matrix,
        :math:`x \in \mathbb{R}^L` is the reference frame defined over an extended field-of-view, and
        :math:`H_{\rm{dyn}} \in \mathbb{R}^{M \times L}` is the dynamic forward operator that compensates the motion.

        The dynamic measurement matrix :math:`H_{\rm{dyn}}` is obtained by **motion-compensation**
        to a reference time, leveraging known deformation field.

        The output is stored in the attribute :attr:`self.H_dyn`.

        .. important::
            There are two ways of building the dynamic matrix, namely :attr:`warping='pattern'` or :attr:`warping='image'`.
            When :attr:`warping='pattern'`, the input deformation field :attr:`motion` needs to be respectively the *inverse*
            deformation field that compensates the motion.
            When :attr:`warping='image'`, the input deformation field :attr:`motion` needs to be the *direct*
            deformation field that induces the motion.

            **Reminder**: When looking at the images vectors as continuous functions from :math:`\mathbb{R}^2` to :math:`\mathbb{R}`,
            we define the **direct** deformation as the function :math:`u \colon \mathbb{Z}^3 \mapsto \mathbb{R}^2` such that,
            for :math:`k \in \{1, ..., M\}` and :math:`(i, j) \in \mathbb{Z}^2`,

            .. math::
                x_{t=k}(i, j) = x_{t=1}(u(t=k, i, j))

            The **inverse** deformation field is defined as :math:`v=u^{-1}`.

        .. note::
            Warping sharp patterns introduces a bias in the model due to interpolation artifacts.
            We recommend to exploit the image regularity by setting :attr:`warping='image'`.

        Args:
            :attr:`motion` (DeformationField): Deformation field representing the
            scene motion. Need to pass the direct deformation field when
            :attr:`warping` is set to 'image', and the inverse deformation field when
            :attr:`warping` is set to 'pattern'.

            :attr:`mode` (str): Interpolation mode for constructing the dynamic matrix. Defaults to 'bilinear'.

            :attr:`warping` (str): Choose between 'image' or 'pattern'. This parameter decides whether to warp
            the patterns or the (unknown) image to recover when building the dynamic measurement matrix.
            Defaults to 'image'.

        Returns:
            None. The dynamic measurement matrix is stored in the attribute :attr:`self.H_dyn`.

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.warp import DeformationField
            >>> from spyrit.core.meas import DynamicLinear
            >>>
            >>> def_field = DeformationField(torch.rand([400, 50, 50, 2]) * 2 - 1)  # dummy deformation field with 400 frames
            >>> x = torch.rand([1, 3, 50, 50])  # dummy RGB reference image of size 50x50
            >>> x_motion = def_field(x)  # dummy video obtained by warping x with def_field
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>>
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinear(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinear(
              (noise_model): Poisson()
            )
            >>>
            >>> meas_op.build_dynamic_forward(def_field)
            >>> print(meas_op.H_dyn.shape)
            torch.Size([400, 2500])

        References:
            [Maitre2024_2]_ Maitre, T., Bretin, E., Phan, R., Ducros, N., & Sdika, M. (2024, October).
            Dynamic single-pixel imaging on an extended field of view without warping the patterns. In International
            Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 275-284).
            Cham: Springer Nature Switzerland. DOI: 10.1007/978-3-031-72104-5_27

            [Maitre2026]_ (Submitted to TIP) Maitre, T., Bretin, E., Mahieu-Williame, L., Phan, R., Sdika, M., & Ducros, N. (2025).
            Dual-arm motion-compensated single-pixel imaging. HAL Id: hal-05068181

        """

        # Deprecate boolean 'warping' values: accept only 'image' or 'pattern' going forward.
        if isinstance(warping, bool):
            warnings.warn(
                "Passing a boolean for 'warping' is deprecated and will be removed in a future release. "
                "Please pass the string 'image' or 'pattern' instead. "
                f"Interpreting {warping!r} as {'pattern' if warping else 'image'}.",
                DeprecationWarning,
            )
            warping = "pattern" if warping else "image"

        if not isinstance(warping, str) or warping not in ("image", "pattern"):
            raise ValueError("warping must be either 'image' or 'pattern'")

        if self.img_shape != motion.img_shape:
            raise RuntimeError(
                "The measurement operator img_shape must be the same as the motion field."
            )

        if self.device != motion.device:
            raise RuntimeError(
                "The device of the motion and the measurement operator must be the same."
            )

        # store the method and mode in attribute
        self._recon_mode = mode
        self._recon_warping = warping

        try:
            del self._param_H_dyn
            del self._param_H_dyn_pinv
            warnings.warn(
                "The dynamic measurement matrix pseudo-inverse H_pinv has "
                + "been deleted. Please call self.build_dynamic_forward_pinv() to "
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

        if isinstance(self, DynamicLinearSplit):
            meas_pattern = self.A
        else:
            meas_pattern = self.H

        if self.white_acq is not None:
            # for eventual spatial gain
            meas_pattern *= self.white_acq.ravel().unsqueeze(0)

        if warping == "image":
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

            # Memory optimization: Clear CUDA cache before large allocations
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # PART 1: SEPARATE THE INTEGER AND DECIMAL PARTS OF THE FIELD
            # _________________________________________________________________
            # crop def_field to keep only measured area
            # moveaxis because crop expects (h,w) as last dimensions
            if verbose:
                print("Part 1: separating integer and decimal parts of the field")

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
            dx, dy = dx.squeeze(-1), dy.squeeze(-1)
            # dx.shape = dy.shape = (n_frames, meas_h, meas_w)

            del def_field
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # evaluate the spline at the decimal part
            dxy = torch.einsum(
                "iajk,ibjk->iabjk", self._spline(dy, mode), self._spline(dx, mode)
            ).reshape(n_frames, kernel_n_pts, self.N)
            # shape (n_frames, kernel_n_pts, meas_h*meas_w)

            # Memory optimization: explicitly delete large intermediate tensors
            del dx, dy
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # PART 2: FLATTEN THE INDICES
            # _________________________________________________________________
            # we consider an expanded grid (img_h+k)x(img_w+k), where k is
            # (kernel_width). This allows each part of the (kernel_size^2)-
            # point grid to contribute to the interpolation.
            # get coordinate of point _00
            if verbose:
                print("Part 2: flattening the indices")

            def_field_00 = def_field_floor - (kernel_size // 2 - 1)
            del def_field_floor
            # shift the grid for phantom rows/columns
            def_field_00 += kernel_width
            # create a mask indicating if either of the 2 indices is out of bounds
            # (w,h) because the def_field is in (x,y) coordinates
            maxs = torch.tensor(
                [self.img_w + kernel_width, self.img_h + kernel_width],
                device=self.device,
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
                def_field_00[..., 0]
                + def_field_00[..., 1] * (self.img_w + kernel_width),
            ).reshape(n_frames, self.N)

            del def_field_00, mask
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # PART 3: WARP H MATRIX WITH FLATTENED INDICES
            # _________________________________________________________________
            # Build 4 submatrices with 4 weights for bilinear interpolation
            if verbose:
                print("Part 3: building H_dyn matrix with flattened indices")

            meas_dxy = meas_pattern.reshape(n_frames, 1, self.N).to(dxy.dtype) * dxy

            del dxy, meas_pattern

            # Memory optimization: Check if we need chunked processing
            sparse_size = (self.img_h + kernel_width) * (self.img_w + kernel_width) + 1
            max_memory_per_tensor = 2e8  # ~200MB limit per tensor
            expected_size = (
                n_frames * kernel_n_pts * sparse_size * meas_dxy.element_size()
            )

            if expected_size > max_memory_per_tensor:
                if verbose:
                    print(
                        f"Using chunked processing to avoid OOM (tensor is expected to be {expected_size/1e9:.2f} GB)"
                    )

                # Process in smaller chunks
                chunk_size = max(
                    1,
                    int(
                        max_memory_per_tensor
                        / (kernel_n_pts * sparse_size * meas_dxy.element_size())
                    ),
                )
                if verbose:
                    print(f"Processing {n_frames} frames in chunks of {chunk_size}")
                H_dyn_chunks = []

                for i in range(0, n_frames, chunk_size):
                    end_idx = min(i + chunk_size, n_frames)
                    chunk_frames = end_idx - i
                    if verbose:
                        print(
                            f"Processing chunk {i//chunk_size + 1}/{(n_frames + chunk_size - 1)//chunk_size}: frames {i} to {end_idx-1}"
                        )

                    # Create smaller tensor for this chunk
                    meas_dxy_sorted_chunk = torch.zeros(
                        (chunk_frames, kernel_n_pts, sparse_size),
                        dtype=meas_dxy.dtype,
                        device=self.device,
                    )

                    # add at flattened_indices the values of meas_dxy for this chunk
                    meas_dxy_sorted_chunk.scatter_add_(
                        2,
                        flattened_indices[i:end_idx]
                        .unsqueeze(1)
                        .expand_as(meas_dxy[i:end_idx]),
                        meas_dxy[i:end_idx],
                    )

                    # drop last column (trash)
                    meas_dxy_sorted_chunk = meas_dxy_sorted_chunk[:, :, :-1]

                    # FOLD THE MATRIX for this chunk
                    fold = nn.Fold(
                        output_size=self.img_shape,
                        kernel_size=(kernel_size, kernel_size),
                        padding=kernel_width,
                    )
                    H_dyn_chunk = fold(meas_dxy_sorted_chunk).reshape(
                        chunk_frames, self.L
                    )
                    H_dyn_chunks.append(
                        H_dyn_chunk.clone()
                    )  # Clone to ensure memory is copied

                    # Clean up chunk memory
                    del meas_dxy_sorted_chunk, H_dyn_chunk
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                # Concatenate all chunks
                H_dyn = torch.cat(H_dyn_chunks, dim=0)
                del H_dyn_chunks
                if verbose:
                    print("Chunked processing completed successfully")

            else:
                if verbose:
                    print("Using standard processing (tensor fits in memory)")
                # Create a larger H_dyn that will be folded
                meas_dxy_sorted = torch.zeros(
                    (n_frames, kernel_n_pts, sparse_size),
                    dtype=meas_dxy.dtype,
                    device=self.device,
                )
                # add at flattened_indices the values of meas_dxy
                meas_dxy_sorted.scatter_add_(
                    2, flattened_indices.unsqueeze(1).expand_as(meas_dxy), meas_dxy
                )

                # drop last column (trash)
                meas_dxy_sorted = meas_dxy_sorted[:, :, :-1]

                # PART 4: FOLD THE MATRIX
                # _________________________________________________________________
                # define operator
                fold = nn.Fold(
                    output_size=self.img_shape,
                    kernel_size=(kernel_size, kernel_size),
                    padding=kernel_width,
                )
                H_dyn = fold(meas_dxy_sorted).reshape(n_frames, self.L)

                # Memory optimization: Clean up after folding
                del meas_dxy_sorted

            # Clean up remaining variables
            del flattened_indices, meas_dxy
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        elif warping == "pattern":
            print(
                "Be careful to use the inverse deformation field when warping patterns."
            )

            # Memory optimization: Clear cache before warping operations
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            det = self._calc_det(def_field)
            det = det.reshape((det.shape[0], -1))

            meas_pattern = meas_pattern.reshape(
                meas_pattern.shape[0], 1, self.meas_shape[0], self.meas_shape[1]
            )

            # Memory optimization: Use in-place operations when possible
            meas_pattern_ext = torch.zeros(
                (meas_pattern.shape[0], 1, self.img_shape[0], self.img_shape[1]),
                dtype=motion.field.dtype,  # Use correct dtype from start
                device=self.device,
            )

            amp_max_h = (self.img_shape[0] - self.meas_shape[0]) // 2
            amp_max_w = (self.img_shape[1] - self.meas_shape[1]) // 2
            meas_pattern_ext[
                :,
                :,
                amp_max_h : self.meas_shape[0] + amp_max_h,
                amp_max_w : self.meas_shape[1] + amp_max_w,
            ] = meas_pattern

            del meas_pattern

            H_dyn = nn.functional.grid_sample(
                meas_pattern_ext,
                motion.field,
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            )

            # Memory optimization: Clean up before final computation
            del meas_pattern_ext
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            H_dyn = H_dyn.reshape((H_dyn.shape[0], -1)) * det

            del det

        self._param_H_dyn = nn.Parameter(H_dyn, requires_grad=False).to(
            self.device
        )  # store in _param_H_dyn

        # Memory optimization: Clean up H_dyn variable (data is now in parameter)
        del H_dyn
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            if verbose:
                print(
                    f"Final memory after storing H_dyn: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
                )

    def _calc_det(self, def_field):
        r"""Computes the determinant of a deformation field.
        It is used for building the dynamic matrix with pattern warping.

        Args:
            :attr:`def_field` (:class:`torch.tensor`): a tensor of shape
            (t, h, w, 2), where t, h, w can be any dimensions.

        Returns:
            :class:`torch.tensor`: The determinant for each frame of the
            deformation field, it has shape (t, h, w).
        """
        # Memory optimization: Clear cache before computation
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # def_field of shape (n_frames, img_shape[0], img_shape[1], 2) in range [0, h-1] x [0, w-1]
        v1, v2 = def_field[:, :, :, 0], def_field[:, :, :, 1]
        n_frames = def_field.shape[0]

        # Memory-efficient gradient computation
        # Compute gradients for v1
        diff_v1_dim1 = torch.diff(v1, dim=1)
        # ones_v1_dim1 = torch.ones(n_frames, 1, v1.shape[2], device=v1.device, dtype=v1.dtype)
        last_v1_dim1 = diff_v1_dim1[:, -1:, :].clone()  # replicate last difference
        dy_v1 = torch.cat([diff_v1_dim1, last_v1_dim1], dim=1)
        del diff_v1_dim1, last_v1_dim1

        diff_v1_dim2 = torch.diff(v1, dim=2)
        # ones_v1_dim2 = torch.ones(n_frames, v1.shape[1], 1, device=v1.device, dtype=v1.dtype)
        last_v1_dim2 = diff_v1_dim2[:, :, -1:].clone()  # replicate last difference
        dx_v1 = torch.cat([diff_v1_dim2, last_v1_dim2], dim=2)
        del diff_v1_dim2, last_v1_dim2, v1

        # Compute gradients for v2
        diff_v2_dim1 = torch.diff(v2, dim=1)
        # ones_v2_dim1 = torch.ones(n_frames, 1, v2.shape[2], device=v2.device, dtype=v2.dtype)
        last_v2_dim1 = diff_v2_dim1[:, -1:, :].clone()  # replicate last difference
        dy_v2 = torch.cat([diff_v2_dim1, last_v2_dim1], dim=1)
        del diff_v2_dim1, last_v2_dim1

        diff_v2_dim2 = torch.diff(v2, dim=2)
        # ones_v2_dim2 = torch.ones(n_frames, v2.shape[1], 1, device=v2.device, dtype=v2.dtype)
        last_v2_dim2 = diff_v2_dim2[:, :, -1:].clone()  # replicate last difference
        dx_v2 = torch.cat([diff_v2_dim2, last_v2_dim2], dim=2)
        del diff_v2_dim2, last_v2_dim2, v2

        # Compute determinant
        det = dx_v1 * dy_v2 - dx_v2 * dy_v1

        # Clean up
        del dx_v1, dy_v1, dx_v2, dy_v2
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return det

    def measure_H_dyn(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noiseless dynamic measurements with the dynamic matrix

        .. math::
            m = H_{\rm{dyn}} x

        where :math:`H_{\rm{dyn}} \in \mathbb{R}^{M \times L}` is the dynamic acquisition matrix,
        :math:`x \in \mathbb{R}^L` is the reference signal of interest,
        :math:`M` is the number of measurements, and
        :math:`L` is the dimension of the signal (with extended FOV).

        .. warning::
            This supposes the dynamic measurement matrix :math:`H_{\rm{dyn}}` has been set using the
            :meth:`build_dynamic_forward()` method. An error will be raised otherwise.

        Args:
            :attr:`x` (torch.tensor): Batch of reference (static) signals. The
            dimensions indexed by :attr:`self.meas_dims` must match the measurement
            shape :attr:`self.img_shape`.

        Returns:
            torch.tensor: Measurement of the input signal. It has shape (..., M).

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.warp import DeformationField
            >>> from spyrit.core.meas import DynamicLinear
            >>>
            >>> def_field = DeformationField(torch.rand([400, 50, 50, 2]) * 2 - 1)  # dummy deformation field with 400 frames
            >>> x = torch.rand([1, 3, 50, 50])  # dummy RGB reference image of size 50x50
            >>> x_motion = def_field(x)  # dummy video obtained by warping x with def_field
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>>
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinear(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinear(
              (noise_model): Poisson()
            )
            >>>
            >>> meas_op.build_dynamic_forward(def_field)
            >>> m = meas_op.measure_H_dyn(x)  # simulate noiseless dynamic measurements from dynamic matrix
            >>> print(m.shape)
            torch.Size([1, 3, 400])

        """
        x = self.vectorize(x)  # don't need to crop because H_dyn has extended FOV
        x = torch.einsum("mn,...n->...m", self.H_dyn, x)
        return x

    def forward_H_dyn(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noisy dynamic measurements with the dynamic matrix

        .. math::
            m = \mathcal{N}\left(H_{\rm{dyn}} x \right)

        where :math:`H_{\rm{dyn}} \in \mathbb{R}^{M \times L}` is the dynamic acquisition matrix,
        :math:`x \in \mathbb{R}^L` is the reference signal of interest,
        :math:`M` is the number of measurements, and
        :math:`L` is the dimension of the signal (with extended FOV).

        .. warning::
            This supposes the dynamic measurement matrix :math:`H_{\rm{dyn}}` has been set using the
            :meth:`build_dynamic_forward()` method. An error will be raised otherwise.

        Args:
            :attr:`x` (torch.tensor): Batch of reference (static) signals. The
            dimensions indexed by :attr:`self.meas_dims` must match the measurement
            shape :attr:`self.img_shape`.

        Returns:
            torch.tensor: Measurement of the input signal. It has shape :math:`(*, M)` where :math:`*`
            denotes all the dimensions that are not included in :attr:`self.meas_dims`

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.warp import DeformationField
            >>> from spyrit.core.meas import DynamicLinear
            >>>
            >>> def_field = DeformationField(torch.rand([400, 50, 50, 2]) * 2 - 1)  # dummy deformation field with 400 frames
            >>> x = torch.rand([1, 3, 50, 50])  # dummy RGB reference image of size 50x50
            >>> x_motion = def_field(x)  # dummy video obtained by warping x with def_field
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>>
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinear(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinear(
              (noise_model): Poisson()
            )
            >>>
            >>> meas_op.build_dynamic_forward(def_field)
            >>> m = meas_op.forward_H_dyn(x)  # simulate noisy dynamic measurements from dynamic matrix
            >>> print(m.shape)
            torch.Size([1, 3, 400])

        """
        x = self.vectorize(x)  # don't need to crop because H_dyn has extended FOV
        x = torch.einsum("mn,...n->...m", self.H_dyn, x)
        x = self.noise_model(x)
        return x

    def adjoint(self, m: torch.tensor, unvectorize=False) -> torch.tensor:
        r"""Apply adjoint of matrix :math:`H_{\rm{dyn}}`.

        It computes

        .. math::
            x = H_{\rm{dyn}}^\top m,

        where :math:`H_{\rm{dyn}}^\top \in\mathbb{R}^{L \times M}` is the adjoint of the
        dynamic acquisition matrix, :math:`m \in \mathbb{R}^M` is the measurement vector.

        .. warning::
            This supposes the dynamic measurement matrix :math:`H_{\rm{dyn}}` has been
            set using the :meth:`build_dynamic_forward()` method. An error will be raised otherwise.

        Args:
            :attr:`m` (:class:`torch.tensor`): A batch of measurement
            :math:`m` of shape :math:`(*, M)` where :math:`*`  denotes all the
            dimensions that are not included in :attr:`self.meas_dims`

        Returns:
            :class:`torch.tensor`: A batch of signals :math:`x`.
            If :attr:`unvectorize` is :obj:`False`, :math:`x` has shape
            :math:`(*, N)` where :math:`*` is the same as for :attr:`m`. If
            :attr:`unvectorize` is :obj:`True`, :math:`x` is reshaped such that
            the dimensions :attr:`self.meas_dims` match the measurement shape
            :attr:`self.meas_shape`.

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.warp import DeformationField
            >>> from spyrit.core.meas import DynamicLinear
            >>>
            >>> def_field = DeformationField(torch.rand([400, 50, 50, 2]) * 2 - 1)  # dummy deformation field with 400 frames
            >>> x = torch.rand([1, 3, 50, 50])  # dummy RGB reference image of size 50x50
            >>> x_motion = def_field(x)  # dummy video obtained by warping x with def_field
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>>
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinear(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinear(
              (noise_model): Poisson()
            )
            >>>
            >>> meas_op.build_dynamic_forward(def_field)
            >>> m = meas_op(x_motion)  # simulate noisy dynamic measurements
            >>> H_dyn_adj_x = meas_op.adjoint(m)  # apply adjoint of dynamic measurement matrix
            >>> print(H_dyn_adj_x.shape)
            torch.Size([1, 3, 2500])

        """
        m = torch.einsum("mn,...m->...n", self.H_dyn, m)
        if unvectorize:
            m = self.unvectorize(m)
        return m

    def vectorize(self, input: torch.tensor) -> torch.tensor:
        r"""Flatten the measured dimensions.

        The tensor is flattened at the indicated `self.meas_dims` dimensions. The
        collapsed dimensions are then moved to the last dimension of the output tensor.
        The time dimension is moved to the second-to-last position.

        Input:
            input (:class:`torch.tensor`): A tensor whose dimensions given by :attr:`self.meas_dims`
            have shape :attr:`self.meas_shape`.

        Output:
            :class:`torch.tensor`: A tensor of shape (:attr:`*, self.M, self.meas_shape`) where * denotes
            all the dimensions of the input tensor not included in :attr:`self.meas_dims`.

        See also:
            For the opposite operation use :meth:`unvectorize()`.

        """
        # concatenate time and measurement dimensions
        time_and_meas_dims = torch.Size([self.time_dim, *self.meas_dims])
        time_and_last_dims = torch.Size(list(range(-len(self.meas_shape) - 1, 0)))
        # move only if necessary
        if time_and_meas_dims != time_and_last_dims:
            input = torch.movedim(input, time_and_meas_dims, time_and_last_dims)
        # flatten the last measured dimensions
        # input = input.reshape(*input.shape[: -self.meas_ndim], self.N)
        input = input.reshape(
            *input.shape[: -self.meas_ndim], -1
        )  # this way it works even for img_shape and meas_shape
        return input

    def unvectorize(self, input: torch.tensor) -> torch.tensor:
        r"""Unflatten the measured dimensions.

        This method expands the last dimension into the measurement or image
        shape (:attr:`self.meas_shape` or :attr:`self.img_shape`), and then moves the
        expanded dimensions to their original positions as defined by :attr:`self.meas_dims`.

        Input:
            :class:`input` (:class:`torch.tensor`): A tensor of shape (:attr:`*, self.N`)
            or (:attr:`*, self.L`) where * denotes any batch size.

        Output:
            :class:`torch.tensor`: A tensor whose dimensions given by :attr:`self.meas_dims`
            have shape :attr:`self.meas_shape` or :attr:`self.img_shape`.

        Raises:
            ValueError: If the last dimension of input is different from :attr:`self.N` or
            :attr:`self.L`

        See also:
            For the opposite operation use :meth:`vectorize()`.

        """
        if input.shape[-1] == self.N:
            unflattened_shape = self.meas_shape
        elif input.shape[-1] == self.L:
            unflattened_shape = self.img_shape
        else:
            raise ValueError("Input of unvectorize has unexpected size in its last dim")

        # unflatten the last dimension
        input = input.reshape(*input.shape[:-1], *unflattened_shape)
        # compare the dimensions
        time_and_meas_dims = torch.Size([self.time_dim, *self.meas_dims])
        time_and_last_dims = torch.Size(list(range(-len(unflattened_shape) - 1, 0)))
        # move dimensions if necessary
        if time_and_meas_dims != time_and_last_dims:
            input = torch.movedim(input, time_and_last_dims, time_and_meas_dims)
        return input

    def _spline(self, dx, mode):
        """
        Returns a 2D row-like tensor containing the values of dx evaluated at
        each B-spline (2 values for bilinear, 4 for bicubic).
        dx must be between 0 and 1.

        Shapes
            dx: (n_frames, meas_h, meas_w)
            out: (n_frames, {2,4}, meas_h, meas_w)
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
    Simulates linear measurements of a moving scene by splitting an acquisition matrix
    :math:`H \in \mathbb{R}^{M \times N}` that contains negative values.
    In practice, only positive values can be implemented using a DMD.
    Therefore, we acquire

    .. math::
        y = \mathcal{N}\left(\text{diag}(A x_{t=1,..., 2M})\right),

    where :math:`A \colon\, \mathbb{R}_+^{2M\times N}` is the acquisition
    matrix that contains positive DMD patterns,
    :math:`x_{t=1,..., 2M} \in \mathbb{R}^{N \times 2M}` is the temporal signal of interest,
    :math:`2M` is both the number of DMD patterns (positives and negatives)
    and the number of frames,
    :math:`N` is the dimension of the signal within the field of view,
    :math:`\text{diag}\colon\, \mathbb{R}^{2M \times 2M} \to \mathbb{R}^{2M}`
    extracts the diagonal of its input, and
    :math:`\mathcal{N} \colon\, \mathbb{R}^{2M} \to \mathbb{R}^{2M}`
    represents a noise operator (e.g., Gaussian).

    Given a matrix :math:`H`, we define the positive DMD patterns :math:`A`
    from the positive and negative components :math:`H`.
    In practice, the even rows of :math:`A` contain the positive components of :math:`H`,
    while odd rows of :math:`A` contain the negative components of :math:`H`.

    .. math::
        \begin{cases}
            A[0::2, :] = H_{+}, \text{ with } H_{+} = \max(0,H),\\
            A[1::2, :] = H_{-}, \text{ with } H_{-} = \max(0,-H).
        \end{cases}

    .. note::
        :math:`H_{+}` and :math:`H_{-}` are such that :math:`H_{+} - H_{-} = H`.

    .. warning::
        For each call, there must be **exactly** twice as many images in :math:`x` as
        there are measurements in the linear operator :math:`H`.

    Args:
        :attr:`H` (:class:`torch.tensor`): measurement matrix (linear operator)
        with shape :math:`(M, N)`. Only real values are supported.

        :attr:`time_dim` (int): dimension index in the input tensor :math:`x` that corresponds
        to time (i.e., the frames dimension).

        :attr:`meas_shape` (tuple, optional): Shape of the measurement patterns.
        Must be a tuple of two integers representing the height and width of the
        patterns. If not specified, the shape is suppposed to be a square image.
        If not, an error is raised. Defaults to None.

        :attr:`meas_dims` (tuple, optional): Dimensions of :math:`x_{t=1, ..., M}` the
        acquisition matrix applies to. Must be a tuple with the same length as
        :attr:`meas_shape`. If not, an error is raised. Defaults to the last
        dimensions of the multi-dimensional array :math:`x_{t=1, ..., M}` (e.g., `(-2,-1)`
        when `len(meas_shape)=2`).

        :attr:`img_shape` (tuple, optional): Shape of the image. Must be a tuple
        of two integers representing the height and width of the image. If not
        specified, the shape is taken as equal to `meas_shape`. Setting this
        value is particularly useful when using an extended field of view [Maitre2024_2]_.

        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model :math:`\mathcal{N}`.
        Defaults to `torch.nn.Identity()`.

        :attr:`white_acq` (torch.tensor, optional): Eventual spatial gain resulting from
        detector inhomogeneities and used for dynamic flat-field correction. It can be
        determined from a "white acquisition" without any object. If None, no correction is
        applied. Must have :attr:`self.meas_shape` shape.

        :attr:`dtype` (:class:`torch.dtype`, optional): Data type of the measurement
        matrix. Defaults to `torch.float32`.

        :attr:`device` (:obj:`torch.device`, optional): Device of the measurement matrix.
        Defaults to `torch.device("cpu")`.

    Attributes:
        :attr:`M` (int): Number of (pos, neg) measurements.

        :attr:`N` (int): Number of pixels in the field of view.

        :attr:`L` (int): Number of pixels in the extended field of view.

        :attr:`meas_shape` (tuple): Shape of the underlying multi-dimensional
        array :math:`x` over the field of view.

        :attr:`img_shape` (tuple): Shape of the underlying multi-dimensional
        array :math:`x` over the extended field of view.

        :attr:`H` (:class:`torch.tensor`): Static measurement matrix of shape
        :math:`(M, N)` initialized as :math:`H`.

        :attr:`A` (:class:`torch.tensor`): Splitted static measurement matrix of shape
        :math:`(2M, N)` initialized as :math:`A`.

        :attr:`H_dyn` (torch.tensor): Differential dynamic measurement matrix :math:`H_{\rm{dyn}}` of shape.
        :math:`(M, L)`. Must be set using the :meth:`build_dynamic_forward` method before being accessed.

        :attr:`A_dyn` (torch.tensor): Splitted dynamic measurement matrix :math:`A_{\rm{dyn}}` of shape.
        :math:`(2M, L)`. Must be set using the :meth:`build_dynamic_forward` method before being accessed.


    Example:
        >>> import torch
        >>> from spyrit.core.meas import DynamicLinearSplit
        >>>
        >>> x = torch.rand([1, 2*400, 3, 50, 50])  # dummy RGB video with 800 frames of size 50x50
        >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
        >>> meas_op = DynamicLinearSplit(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50))
        >>> print(meas_op)
        DynamicLinearSplit(
          (noise_model): Identity()
        )

    References:
        [Maitre2024_2]_ Maitre, T., Bretin, E., Phan, R., Ducros, N., & Sdika, M. (2024, October).
        Dynamic single-pixel imaging on an extended field of view without warping the patterns. In International
        Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 275-284).
        Cham: Springer Nature Switzerland. DOI: 10.1007/978-3-031-72104-5_27

        [Maitre2026]_ (Submitted to TIP) Maitre, T., Bretin, E., Mahieu-Williame, L., Phan, R., Sdika, M., & Ducros, N. (2025).
        Dual-arm motion-compensated single-pixel imaging. HAL Id: hal-05068181

    """

    def __init__(
        self,
        H: torch.tensor,
        time_dim: int,
        meas_shape: Union[int, torch.Size, Iterable[int]] = None,
        meas_dims: Union[int, torch.Size, Iterable[int]] = None,
        img_shape: Union[int, torch.Size, Iterable[int]] = None,
        *,
        noise_model: nn.Module = nn.Identity(),
        white_acq: torch.tensor = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        # call constructor of DynamicLinear
        super().__init__(
            H,
            time_dim,
            meas_shape,
            meas_dims,
            img_shape,
            noise_model=noise_model,
            white_acq=white_acq,
            dtype=dtype,
            device=device,
        )

        # split positive and negative components
        pos, neg = nn.functional.relu(self.H), nn.functional.relu(-self.H)
        A = torch.cat([pos, neg], 1).reshape(2 * self.M, self.N)

        # A is built from self.H which is cast to device and dtype
        self.A = nn.Parameter(A, requires_grad=False)

        # define the available matrices for reconstruction
        self._available_pinv_matrices = ["H_dyn", "A_dyn"]
        self._selected_pinv_matrix = "H_dyn"  # select default here

    @property
    def A_dyn(self) -> torch.tensor:
        """Splitted dynamic measurement matrix computed with the call to build_dynamic_forward"""
        try:
            return self._param_H_dyn.data
        except AttributeError as e:
            raise AttributeError(
                "The dynamic measurement matrix H_dyn has not been set yet. "
                + "Please call build_dynamic_forward() before accessing the attribute H_dyn_diff."
            ) from e

    @property
    def H_dyn(self) -> torch.tensor:
        """Dynamic measurement matrix H_dyn_diff that adopts the differential
        measurement strategy as described in [ref_journal],
        i.e., `H_dyn[0] - H_dyn[1]`, `H_dyn[2] - H_dyn[3]`, etc."""
        try:
            return self._param_H_dyn.data[::2] - self._param_H_dyn.data[1::2]
        except AttributeError as e:
            raise AttributeError(
                "The dynamic measurement matrix H_dyn has not been set yet. "
                + "Please call build_dynamic_forward() before accessing the attribute H_dyn_diff."
            ) from e

    def measure(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noiseless dynamic measurements from matrix A.

        It acquires

        .. math::
            y = \text{diag}(A x_{t=1, ..., 2M}),

        where :math:`A \in \mathbb{R}_+^{2M\times N}` is the acquisition matrix that contains positive DMD patterns,
        :math:`x \in \mathbb{R}^{N \times 2M}` is the temporal signal of interest,
        :math:`2M` is the number of DMD patterns and the number of frames,
        :math:`N` is the dimension of the signal, and
        :math:`\text{diag}\colon\, \mathbb{R}^{2M \times 2M} \to \mathbb{R}^{2M}`
        extracts the diagonal of its input.

        Given a matrix :math:`H \in \mathbb{R}^{M\times N}`,
        we define the positive DMD patterns :math:`A` from the positive and negative components of :math:`H`.

        .. note::
            The acquisition matrix :math:`A` is given by :attr:`self.A`.

        Args:
            :attr:`x` (:class:`torch.tensor`): Batch of temporal signals :math:`x` whose
            time dimensions :matches :attr:`self.time_dim` and measured dimensions matches
            :attr:`self.meas_dims`

        Returns:
            :class:`torch.tensor`: Measurement vector :math:`m` of length :attr:`2\*self.M`.

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.meas import DynamicLinearSplit
            >>>
            >>> x = torch.rand([1, 2*400, 3, 50, 50])  # dummy RGB video with 800 frames of size 50x50
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinearSplit(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinearSplit(
              (noise_model): Poisson()
            )
            >>>
            >>> y = meas_op.measure(x)  # simulate noiseless dynamic measurements
            >>> print(y.shape)
            torch.Size([1, 3, 800])

        """
        x = spytorch.center_crop(x, self.meas_shape)
        # vectorize with the time dimension being the second-to-last dimension
        x = self.vectorize(x)
        # here index m is the number of mesurements and the number of frames
        x = torch.einsum("mn,...mn->...m", self.A, x)
        return x

    def measure_H(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noiseless dynamic measurements from matrix H.

        It acquires

        .. math::
            m = \text{diag}(H x_{t=1, ..., M}),

        where :math:`H \in \mathbb{R}^{M\times N}` is the measurement matrix (that may contain negative values),
        :math:`x_{t=1, ..., M} \in \mathbb{R}^{N \times M}` is the temporal signal obtained from
        averaging the positive and negative frames of :math:`x_{t=1, ..., 2M}`,
        :math:`M` is the number of DMD patterns,
        :math:`N` is the dimension of the signal, and
        :math:`\text{diag}\colon\, \mathbb{R}^{M \times M} \to \mathbb{R}^{M}`
        extracts the diagonal of its input.

        .. note::
            The acquisition matrix :math:`H` is given by :attr:`self.H`.

        .. note::
            Here the number of frames is 2M and the number of measurements is M.

        Args:
            :attr:`x` (:class:`torch.tensor`): Batch of temporal signals :math:`x` whose
            time dimensions :matches :attr:`self.time_dim` and measured dimensions matches
            :attr:`self.meas_dims`

        Returns:
            :class:`torch.tensor`: Measurement vector :math:`m` of length :attr:`self.M`.

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.meas import DynamicLinearSplit
            >>>
            >>> x = torch.rand([1, 2*400, 3, 50, 50])  # dummy RGB video with 800 frames of size 50x50
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinearSplit(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinearSplit(
              (noise_model): Poisson()
            )
            >>>
            >>> m = meas_op.measure_H(x)  # simulate noiseless dynamic measurements from matrix H
            >>> print(m.shape)
            torch.Size([1, 3, 400])

        """
        x = x.movedim(self.time_dim, 0)
        x = (x[::2] + x[1::2]) / 2
        x = x.movedim(0, self.time_dim)
        return super().measure(x)

    def build_dynamic_forward(
        self,
        motion: DeformationField,
        mode: str = "bilinear",
        warping: str = "image",
        verbose: bool = False,
    ) -> None:
        r"""Builds the dynamic forward operator :math:`A_{\rm{dyn}}`.

        .. math::
            \text{diag}(A x_{t=1, ..., 2M}) = A_{\rm{dyn}} x,

        where
        :math:`x_{t=1, ..., 2M} \in \mathbb{R}^{N \times 2M}` is the temporal signal of interest,
        :math:`A \in \mathbb{R}^{2M \times N}` is the splitted static acquisition matrix,
        :math:`x \in \mathbb{R}^L` is the reference frame defined over an extended field-of-view, and
        :math:`A_{\rm{dyn}} \in \mathbb{R}^{M \times L}` is the splitted dynamic forward operator that compensates the motion.

        The dynamic measurement matrix :math:`A_{\rm{dyn}}` is obtained by **motion-compensation**
        to a reference time, leveraging known deformation field.

        The output is stored in the attribute :attr:`self.A_dyn`.

        .. important::
            There are two ways of building the dynamic matrix, namely :attr:`warping='pattern'` or :attr:`warping='image'`.
            When :attr:`warping='pattern'`, the input deformation field :attr:`motion` needs to be respectively the *inverse*
            deformation field that compensates the motion.
            When :attr:`warping='image'`, the input deformation field :attr:`motion` needs to be the *direct*
            deformation field that induces the motion.

            **Reminder**: When looking at the images vectors as continuous functions from :math:`\mathbb{R}^2` to :math:`\mathbb{R}`,
            we define the **direct** deformation as the function :math:`u \colon \mathbb{Z}^3 \mapsto \mathbb{R}^2` such that,
            for :math:`k \in \{1, ..., 2M\}` and :math:`(i, j) \in \mathbb{Z}^2`,

            .. math::
                x_{t=k}(i, j) = x_{t=1}(u(t=k, i, j))

            The *inverse* deformation field is defined as :math:`v=u^{-1}`.

        .. note::
            Warping sharp patterns introduces a bias in the model due to interpolation artifacts.
            We recommend to exploit the image regularity by setting :attr:`warping='image'`.

        .. note::
            When working with splitted measurements, it is common practice to exploit the problem's linearity by using
            a differential measurement strategy. This allows to eliminate ambient light and dark current offsets.
            The attribute :attr:`H_dyn` applies the differential strategy **after** motion compensation to avoid
            an additional error term [ref journal].

        Args:
            :attr:`motion` (DeformationField): Deformation field representing the
            scene motion. Need to pass the deformation field when
            :attr:`warping` is set to 'image', and the inverse deformation field when
            :attr:`warping` is set to 'pattern'.

            :attr:`mode` (str): Interpolation mode for constructing the dynamic matrix. Defaults to 'bilinear'.

            :attr:`warping` (str): Choose between 'image' or 'pattern'. This parameter decides whether to warp
            the patterns or the (unknown) image to recover when building the dynamic measurement matrix.
            Defaults to 'image'.

        Returns:
            None. The dynamic measurement matrix is stored in the attribute :attr:`self.A_dyn`.

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.warp import DeformationField
            >>> from spyrit.core.meas import DynamicLinearSplit
            >>>
            >>> def_field = DeformationField(torch.rand([800, 50, 50, 2]) * 2 - 1)  # dummy deformation field with 400 frames
            >>> x = torch.rand([1, 3, 50, 50])  # dummy RGB reference image of size 50x50
            >>> x_motion = def_field(x)  # dummy video obtained by warping x with def_field
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>>
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinearSplit(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinearSplit(
              (noise_model): Poisson()
            )
            >>>
            >>> meas_op.build_dynamic_forward(def_field)
            >>> print(meas_op.A_dyn.shape)
            torch.Size([800, 2500])
            >>> print(meas_op.H_dyn.shape)
            torch.Size([400, 2500])

        References:
            [Maitre2024_2]_ Maitre, T., Bretin, E., Phan, R., Ducros, N., & Sdika, M. (2024, October).
            Dynamic single-pixel imaging on an extended field of view without warping the patterns. In International
            Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 275-284).
            Cham: Springer Nature Switzerland. DOI: 10.1007/978-3-031-72104-5_27

            [Maitre2026]_ (Submitted to TIP) Maitre, T., Bretin, E., Mahieu-Williame, L., Phan, R., Sdika, M., & Ducros, N. (2025).
            Dual-arm motion-compensated single-pixel imaging. HAL Id: hal-05068181

        """

        # redefine to update doc for splitted measurements
        super().build_dynamic_forward(motion, mode, warping, verbose)

    def adjoint(self, y: torch.tensor, unvectorize=False):
        r"""Apply adjoint of matrix :math:`A_{\rm{dyn}}`.

        It computes

        .. math::
            x = A_{\rm{dyn}}^\top y,

        where :math:`A_{\rm{dyn}} \in \mathbb{R}^{2M\times L}` is the
        dynamic acquisition matrix (that may contain negative values due to warping)
        and :math:`y \in \mathbb{R}^{2M}` is a measurement vector.

        .. warning::
            This supposes the dynamic measurement matrix :math:`A_{\rm{dyn}}` has been
            set using the :meth:`build_dynamic_forward()` method. An error will be raised otherwise.

        .. note::
            The acquisition matrix :math:`A_{\rm{dyn}}` is given by :attr:`self.A_dyn`.
            It may contains negative values due to warping.

        Args:
            :attr:`y` (:class:`torch.tensor`): Measurement :math:`y` whose dimensions
            :attr:`self.meas_dims` must have shape :attr:`self.meas_shape`.

        Returns:
            :class:`torch.tensor`: A batch of signals :math:`x` with shape :math:`(*, N)`
            where :math:`*` is the same as for :attr:`m`.

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.warp import DeformationField
            >>> from spyrit.core.meas import DynamicLinearSplit
            >>>
            >>> def_field = DeformationField(torch.rand([800, 50, 50, 2]) * 2 - 1)  # dummy deformation field with 400 frames
            >>> x = torch.rand([1, 3, 50, 50])  # dummy RGB reference image of size 50x50
            >>> x_motion = def_field(x)  # dummy video obtained by warping x with def_field
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>>
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinearSplit(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinearSplit(
              (noise_model): Poisson()
            )
            >>>
            >>> meas_op.build_dynamic_forward(def_field)
            >>> y = meas_op(x_motion)  # simulate noisy dynamic measurements
            >>> A_dyn_adj_x = meas_op.adjoint(y)  # apply adjoint of dynamic measurement matrix
            >>> print(A_dyn_adj_x.shape)
            torch.Size([1, 3, 2500])

        """
        y = torch.einsum("mn,...m->...n", self.A_dyn, y)
        if unvectorize:
            y = self.unvectorize(y)
        return y

    def adjoint_H_dyn(self, m: torch.tensor, unvectorize=False):
        r"""Apply adjoint of matrix :math:`H_{\rm{dyn}}`.

        It computes

        .. math::
            x = H_{\rm{dyn}}^\top m,

        where :math:`H_{\rm{dyn}} \in \mathbb{R}^{M \times L}` is the
        dynamic acquisition matrix (that may contain negative values),
        :math:`m \in \mathbb{R}^M` is a measurement vector.

        .. warning::
            This supposes the dynamic measurement matrix :math:`H_{\rm{dyn}}` has been
            set using the :meth:`build_dynamic_forward()` method. An error will be raised otherwise.

        .. note::
            The acquisition matrix :math:`H_{\rm{dyn}}` is given by :attr:`self.H_dyn`.

        Args:
            :attr:`m` (:class:`torch.tensor`): Measurements :math:`m` whose dimensions
            :attr:`self.meas_dims` must have shape :attr:`self.meas_shape`.

        Returns:
            A batch of signals :math:`x`. If :attr:`unvectorize` is :obj:`False`, :math:`x` has
            shape :math:`(*, L)` where :math:`*` is the same as for :attr:`m`. If :attr:`unvectorize`
            is :obj:`True`, :math:`x` is reshaped such that the dimensions :attr:`self.meas_dims` have
            shape :attr:`self.img_shape`.

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.warp import DeformationField
            >>> from spyrit.core.meas import DynamicLinearSplit
            >>>
            >>> def_field = DeformationField(torch.rand([800, 50, 50, 2]) * 2 - 1)  # dummy deformation field with 400 frames
            >>> x = torch.rand([1, 3, 50, 50])  # dummy RGB reference image of size 50x50
            >>> x_motion = def_field(x)  # dummy video obtained by warping x with def_field
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>>
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinearSplit(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinearSplit(
              (noise_model): Poisson()
            )
            >>>
            >>> meas_op.build_dynamic_forward(def_field)
            >>> m = meas_op.measure_H(x_motion)  # simulate noiseless dynamic measurements from matrix H
            >>> H_dyn_adj_x = meas_op.adjoint_H_dyn(m)  # apply adjoint of dynamic measurement matrix (with differential strategy)
            >>> print(H_dyn_adj_x.shape)
            torch.Size([1, 3, 2500])

        """
        return super().adjoint(m, unvectorize=unvectorize)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noisy dynamic measurements from matrix A.

        It acquires

        .. math::
            m = \mathcal{N}\left(\text{diag}(A x_{t=1, ..., 2M})\right),

        where :math:`A \in \mathbb{R}_+^{2M\times N}` is the acquisition matrix that contains positive DMD patterns,
        :math:`x \in \mathbb{R}^{N \times 2M}` is the temporal signal of interest,
        :math:`2M` is the number of DMD patterns and the number of frames,
        :math:`N` is the dimension of the signal,
        :math:`\text{diag}\colon\, \mathbb{R}^{2M \times 2M} \to \mathbb{R}^{2M}`
        extracts the diagonal of its input, and
        :math:`\mathcal{N} \colon\, \mathbb{R}^{2M} \to \mathbb{R}^{2M}`
        represents a noise operator (e.g., Gaussian).

        Given a matrix :math:`H \in \mathbb{R}^{M\times N}`,
        we define the positive DMD patterns :math:`A` from the positive and negative components of :math:`H`.

        .. note::
            The acquisition matrix :math:`A` is given by :attr:`self.A`.

        Args:
            :attr:`x` (:class:`torch.tensor`): Video signal :math:`x` whose
            dimensions :attr:`self.meas_dims` must be of shape :attr:`self.meas_shape`
            and dimension :attr:`self.time_dim` must be of size :attr:`2 * self.M`.

        Returns:
            :class:`torch.tensor`: Measurement vector :math:`m` of length :attr:`2\*self.M`.

        Example:
            >>> import torch
            >>> from spyrit.core.meas import DynamicLinearSplit
            >>> from spyrit.core.noise import Poisson
            >>>
            >>> x = torch.rand([1, 800, 3, 50, 50])  # dummy RGB video with 400 frames of size 50x50
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinearSplit(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinearSplit(
              (noise_model): Poisson()
            )
            >>>
            >>> y = meas_op(x)  # simulate noisy dynamic measurements
            >>> print(y.shape)
            torch.Size([1, 3, 800])

        """

        # it is ok to use super().forward, because measure method has been redefined
        return super().forward(x)

    def forward_H(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noisy dynamic measurements from matrix H.

        It acquires

        .. math::
            m = \mathcal{N}\left(\text{diag}(H x_{t=1, ..., M})\right),

        where :math:`H \in \mathbb{R}^{M\times N}` is the measurement matrix (that may contain negative values),
        :math:`x_{t=1, ..., M} \in \mathbb{R}^{N \times M}` is the temporal signal obtained from
        averaging the positive and negative frames of :math:`x_{t=1, ..., 2M}`,
        :math:`M` is the number of DMD patterns,
        :math:`N` is the dimension of the signal,
        :math:`\text{diag}\colon\, \mathbb{R}^{M \times M} \to \mathbb{R}^{M}`
        extracts the diagonal of its input, and
        :math:`\mathcal{N} \colon\, \mathbb{R}^{M} \to \mathbb{R}^{M}`
        represents a noise operator (e.g., Gaussian).

        .. note::
            The acquisition matrix :math:`H` is given by :attr:`self.H`.

        .. note::
            Here the number of frames is 2M and the number of measurements is M.

        Args:
            :attr:`x` (:class:`torch.tensor`): Video signal :math:`x` whose
            dimensions :attr:`self.meas_dims` must be of shape :attr:`self.meas_shape`
            and dimension :attr:`self.time_dim` must be of size :attr:`2 * self.M`.

        Returns:
            :class:`torch.tensor`: Measurement vector :math:`m` of length :attr:`self.M`.

        Example:
            >>> import torch
            >>> from spyrit.core.meas import DynamicLinearSplit
            >>> from spyrit.core.noise import Poisson
            >>>
            >>> x = torch.rand([1, 800, 3, 50, 50])  # dummy RGB video with 400 frames of size 50x50
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinearSplit(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinearSplit(
              (noise_model): Poisson()
            )
            >>>
            >>> m = meas_op.forward_H(x)  # simulate noisy dynamic measurements
            >>> print(m.shape)
            torch.Size([1, 3, 400])

        """
        x = self.measure_H(x)
        x = self.noise_model(x)
        return x

    def measure_A_dyn(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noiseless dynamic measurements with the splitted dynamic matrix

        .. math::
            y = A_{\rm{dyn}} x

        where :math:`A_{\rm{dyn}} \in \mathbb{R}^{2 M \times L}` is the dynamic acquisition matrix,
        :math:`x \in \mathbb{R}^L` is the reference signal of interest,
        :math:`M` is the number of measurements, and
        :math:`L` is the dimension of the signal (with extended FOV).

        .. warning::
            This supposes the dynamic measurement matrix :math:`A_{\rm{dyn}}` has been set using the
            :meth:`build_dynamic_forward()` method. An error will be raised otherwise.

        Args:
            :attr:`x` (torch.tensor): Batch of reference (static) signals. The
            dimensions indexed by :attr:`self.meas_dims` must match the measurement
            shape :attr:`self.img_shape`.

        Returns:
            torch.tensor: Measurement of the input signal. It has shape :math:`(*, M)` where :math:`*`
            denotes all the dimensions that are not included in :attr:`self.meas_dims`

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.warp import DeformationField
            >>> from spyrit.core.meas import DynamicLinearSplit
            >>>
            >>> def_field = DeformationField(torch.rand([800, 50, 50, 2]) * 2 - 1)  # dummy deformation field with 400 frames
            >>> x = torch.rand([1, 3, 50, 50])  # dummy RGB reference image of size 50x50
            >>> x_motion = def_field(x)  # dummy video obtained by warping x with def_field
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>>
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinearSplit(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinearSplit(
              (noise_model): Poisson()
            )
            >>>
            >>> meas_op.build_dynamic_forward(def_field)
            >>> y = meas_op.measure_A_dyn(x)  # simulate noiseless dynamic measurements from splitted dynamic matrix A_dyn
            >>> print(y.shape)
            torch.Size([1, 3, 800])

        """
        x = self.vectorize(x)  # don't need to crop because A_dyn has extended FOV
        x = torch.einsum("mn,...n->...m", self.A_dyn, x)
        return x

    def forward_A_dyn(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noisy dynamic measurements with the splitted dynamic matrix

        .. math::
            y = \mathcal{N}\left(A_{\rm{dyn}} x \right)

        where :math:`A_{\rm{dyn}} \in \mathbb{R}^{2 M \times L}` is the dynamic acquisition matrix,
        :math:`x \in \mathbb{R}^L` is the reference signal of interest,
        :math:`M` is the number of measurements, and
        :math:`L` is the dimension of the signal (with extended FOV).

        .. warning::
            This supposes the dynamic measurement matrix :math:`A_{\rm{dyn}}` has been set using the
            :meth:`build_dynamic_forward()` method. An error will be raised otherwise.

        Args:
            :attr:`x` (torch.tensor): Batch of reference (static) signals. The
            dimensions indexed by :attr:`self.meas_dims` must match the measurement
            shape :attr:`self.img_shape`.

        Returns:
            torch.tensor: Measurement of the input signal. It has shape :math:`(*, M)` where :math:`*`
            denotes all the dimensions that are not included in :attr:`self.meas_dims`

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.warp import DeformationField
            >>> from spyrit.core.meas import DynamicLinearSplit
            >>>
            >>> def_field = DeformationField(torch.rand([800, 50, 50, 2]) * 2 - 1)  # dummy deformation field with 400 frames
            >>> x = torch.rand([1, 3, 50, 50])  # dummy RGB reference image of size 50x50
            >>> x_motion = def_field(x)  # dummy video obtained by warping x with def_field
            >>> H = torch.rand([400, 40*40])  # dummy static measurement matrix
            >>>
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicLinearSplit(H, time_dim=1, meas_shape=(40, 40), img_shape=(50, 50), noise_model=noise_op)
            >>> print(meas_op)
            DynamicLinearSplit(
              (noise_model): Poisson()
            )
            >>>
            >>> meas_op.build_dynamic_forward(def_field)
            >>> y = meas_op.forward_A_dyn(x)  # simulate noisy dynamic measurements from splitted dynamic matrix A_dyn
            >>> print(y.shape)
            torch.Size([1, 3, 800])

        """
        x = self.vectorize(x)  # don't need to crop because A_dyn has extended FOV
        x = torch.einsum("mn,...n->...m", self.A_dyn, x)
        x = self.noise_model(x)
        return x


# =============================================================================
class DynamicHadamSplit2d(DynamicLinearSplit):
    # =========================================================================
    r""" Simulate 2D Hadamard split acquisitions of a moving scene.

    We perform the acquisition of :math:`2M` square DMD patterns of size :math:`h` by exploiting the Kronecker structure of the 2D Hadamard matrix:

    Each measurement is acquired as, for :math:`k \in \{1, ..., 2M\}`:

    .. math::

        y_k = \mathcal{N}\left( \sum_{i, j} A_{1d}[r_k, i] x_{t=k}[i, j] A_{1d}[j, c_k] \right),

    where
    :math:`A_{1d} \in \mathbb{R}_+^{2h\times h}` contains the positive and negative components of a 1d Hadamard matrix,
    :math:`x_{t=k} \in \mathbb{R}^{h \times h}` is :math:`k^{\rm{th}}` frame of the video,
    :math:`(r_k, c_k) = (\left \lfloor k / h \right\rfloor, k \bmod h)` are the row and column indices of the 1d Hadamard matrix used to generate the 2d Hadamard pattern used at time :math:`t=k`,
    :math:`\mathcal{N} \colon\, \mathbb{R} \to \mathbb{R}` represents a noise operator (e.g., Gaussian).


    .. important::
        Only the forward methods benefit from the fast Hadamard transform algorithm, the adjoint methods do not because
        the dynamic forward operator (:math:`H_{\rm{dyn}}` or :math:`A_{\rm{dyn}}`) does not have Kronecker structure.


    .. note::
        The splitting of the :math:`k^{\rm{th}}` 2D pattern into its positive and negative parts is given by splitting 1D patterns as:

        .. math::
            H[k, :]^{+} = H_{1d}^{+}[r_k, :] \otimes H_{1d}^{+}[:, c_k] + H_{1d}^{-}[r_k, :] \otimes H_{1d}^{-}[:, c_k] \\
            H[k, :]^{-} = H_{1d}^{+}[r_k, :] \otimes H_{1d}^{-}[:, c_k] + H_{1d}^{-}[r_k, :] \otimes H_{1d}^{+}[:, c_k]


    Args:
        :attr:`time_dim` (int): dimension index in the input tensor :math:`x` that corresponds
        to time (i.e., the frames dimension).

        :attr:`h` (int): Image size :math:`h`. Must be a power of 2.

        :attr:`M` (int): Number of (pos, neg) measurements. If None, it is set to :math:`h^2` (no subsampling).

        :attr:`order` (:class:`torch.tensor`, optional): Order matrix :math:`O` that defines the measurements to keep. The first component of :math:`y` will correspond to the index where :attr:`order` is the highest.

        :attr:`fast` (bool, optional): Whether to use the fast Hadamard transform
        algorithm (i.e. exploit the kronecker structure). If False, it uses matrix-vector products. Defaults to True.

        :attr:`reshape_output` (bool, optional): Whether reshape the output of adjoint and pinv methods to images. If False, output are vectors.

        :attr:`img_shape` (tuple): Shape of the underlying multi-dimensional
        array :math:`x` over the extended field of view. If None, is set to :math:`(h, h)`.

        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model :math:`\mathcal{N}`.
        Defaults to `torch.nn.Identity()`.

        :attr:`white_acq` (torch.tensor, optional): Eventual spatial gain resulting from
        detector inhomogeneities and used for dynamic flat-field correction. It can be
        determined from a "white acquisition" without any object. If None, no correction is
        applied. Must have :attr:`self.meas_shape` shape.

        :attr:`dtype` (:class:`torch.dtype`, optional): Data type of the measurement
        matrix. Defaults to `torch.float32`.

        :attr:`device` (:obj:`torch.device`, optional): Device of the measurement matrix.
        Defaults to `torch.device("cpu")`.

    .. note::
        The argument :attr:`order` is particularly useful when rearranging the
        measurements by decreasing variance. The variance matrix can simply be
        put as `order`.

    Attributes:
        :attr:`M` (int): Number of (pos, neg) measurements.

        :attr:`N` (int): Number of pixels in the field of view.

        :attr:`L` (int): Number of pixels in the extended field of view.

        :attr:`meas_shape` (tuple): Shape of the measurements patterns. It is equal to :math:`(h, h)`.

        :attr:`meas_dims` (torch.Size): Dimensions of the image the acquisition
        matrix applies to. Is equal to `(-2, -1)`.

        :attr:`img_shape` (tuple): Shape of the underlying multi-dimensional
        array :math:`x` over the extended field of view.

        :attr:`H1d` (:class:`torch.tensor`): Static 1D Hadamard matrix of shape
        :math:`(h, h)`.

        :attr:`H` (:class:`torch.tensor`): Static 2D Hadamard matrix of shape
        :math:`(M, N)` given by :math:`H_{1d} \otimes H_{1d}`.

        :attr:`A` (:class:`torch.tensor`): Splitted static 2d Hadamard matrix of shape
        :math:`(2M, N)` given by :math:`A_{1d} \otimes A_{1d}`.

        :attr:`H_dyn` (torch.tensor): Differential dynamic Hadamard matrix :math:`H_{\rm{dyn}}` of shape.
        :math:`(M, L)`. Must be set using the :meth:`build_dynamic_forward` method before being accessed.

        :attr:`A_dyn` (torch.tensor): Splitted dynamic Hadamard matrix :math:`A_{\rm{dyn}}` of shape.
        :math:`(2M, L)`. Must be set using the :meth:`build_dynamic_forward` method before being accessed.

        :attr:`order` (:class:`torch.tensor`): Order matrix :math:`O`. It
        is used by :func:`~spyrit.core.torch.sort_by_significance()`. Defaults to rectangular order (e.g., linear indices).

        :attr:`indices` (:class:`torch.tensor`): Indices used to reorder the measurement vector. It is used by the method :meth:`reindex()`.

    Example:
        >>> import torch
        >>> from spyrit.core.meas import DynamicHadamSplit2d
        >>>
        >>> order = torch.rand([32,32])
        >>> # acquisition with 2 * 32 ** 2 splitted Hadamard patterns of size 32x32.
        >>> meas_op = DynamicHadamSplit2d(time_dim=1, h=32, M=32**2, order=order, img_shape=(40, 40))
        >>> print(meas_op)
        DynamicHadamSplit2d(
          (noise_model): Identity()
        )

    Reference:
        [Maitre2024_2]_ Maitre, T., Bretin, E., Phan, R., Ducros, N., & Sdika, M. (2024, October).
        Dynamic single-pixel imaging on an extended field of view without warping the patterns. In International
        Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 275-284).
        Cham: Springer Nature Switzerland. DOI: 10.1007/978-3-031-72104-5_27

        [Maitre2026]_ (Submitted to TIP) Maitre, T., Bretin, E., Mahieu-Williame, L., Phan, R., Sdika, M., & Ducros, N. (2025).
        Dual-arm motion-compensated single-pixel imaging. HAL Id: hal-05068181


    """

    def __init__(
        self,
        time_dim: int,
        h: int,
        M: int = None,
        order: torch.tensor = None,
        fast: bool = True,
        reshape_output: bool = False,
        img_shape: Union[int, torch.Size, Iterable[int]] = None,
        *,
        noise_model: nn.Module = nn.Identity(),
        white_acq: torch.tensor = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        meas_dims = (-2, -1)
        meas_shape = (h, h)
        if M is None:
            M = h**2

        self.h = h

        # call DynamicLinearSplit constructor (avoid setting A)
        super(DynamicLinearSplit, self).__init__(
            torch.empty(h**2, h**2, dtype=dtype, device=device),  # dummy H
            time_dim,
            meas_shape,
            meas_dims,
            img_shape,
            noise_model=noise_model,
            white_acq=white_acq,
            dtype=dtype,
            device=device,
        )

        if order is None:
            order = torch.ones(h, h)
        # 1D version of H
        # H1d = spytorch.walsh_matrix(h).to(dtype=dtype, device=device)
        # self.H1d = nn.Parameter(H1d, requires_grad=False)
        self.H1d = nn.Parameter(spytorch.walsh_matrix(h), requires_grad=False).to(
            dtype=dtype, device=device
        )
        self.M = M  # supercharged self.M
        self.order = order
        self.indices = torch.argsort(-order.flatten(), stable=True).to(
            dtype=torch.int32, device=self.device
        )
        self.fast = fast
        self.reshape_output = reshape_output

    @property
    def dtype(self) -> torch.dtype:
        return self.H1d.dtype

    @property
    def device(self) -> torch.device:
        return self.H1d.device

    @property
    def H(self):
        H = torch.kron(self.H1d, self.H1d)
        H = self.reindex(H, "rows", False)

        # !!!!

        return H[: self.M, :]

    @property
    def A(self):
        H = self.H
        pos, neg = nn.functional.relu(H), nn.functional.relu(-H)
        return torch.cat([pos, neg], 1).reshape(2 * self.M, self.N)

    @property
    def matrix_to_inverse(self):
        return self.H

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
            values (:class:`torch.tensor`): The tensor to sort. Can be 1D, 2D, or any
            multi-dimensional batch of 2D tensors.

            axis (str, optional): The axis to sort along. Must be either 'rows' or
            'cols'. If `values` is 1D, `axis` is not used. Default is 'rows'.

            inverse_permutation (bool, optional): Whether to apply the permutation
            inverse. Default is False.

        Raises:
            ValueError: If `axis` is not 'rows' or 'cols'.

        Returns:
            :class:`torch.tensor`: The sorted tensor by the given indices along the
            specified axis.
        """
        return spytorch.reindex(x, self.indices.to(x.device), axis, inverse_permutation)

    def measure(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noiseless measurements leveraging the Kronecker structure of the 2d splitted Hadamard transform A.

        Each measurement is acquired as, for :math:`k \in \{1, ..., 2M\}`:

        .. math::

            y_k = \sum_{i, j} A_{1d}[r_k, i] x_{t=k}[i, j] A_{1d}[j, c_k],

        where
        :math:`A_{1d} \in \mathbb{R}_+^{2h\times h}` contains the positive and negative components of a 1d Hadamard matrix,
        :math:`x_{t=k} \in \mathbb{R}^{h \times h}` is :math:`k^{\rm{th}}` frame of the video,
        :math:`(r_k, c_k) = (\left \lfloor k / h \right\rfloor, k \bmod h)` are the row and column indices of the 1d Hadamard matrix
        used to generate the 2d Hadamard pattern used at time :math:`t=k`.

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.meas import DynamicHadamSplit2d
            >>>
            >>> x = torch.rand([1, 2 * 32**2, 3, 40, 40])  # dummy RGB video with 2 * 32**2 frames of size 40x40
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicHadamSplit2d(time_dim=1, h=32, M=32**2, img_shape=(40, 40), \
            ...                               noise_model=noise_op)  # acquisition with 2*M splitted Hadamard patterns of size hxh.
            >>> print(meas_op)
            DynamicHadamSplit2d(
              (noise_model): Poisson()
            )
            >>>
            >>> y = meas_op.measure(x)  # simulate noiseless dynamic measurements
            >>> print(y.shape)
            torch.Size([1, 3, 2048])

        """

        if self.fast:
            x = spytorch.center_crop(x, self.meas_shape)

            time_and_meas_dims = torch.Size([self.time_dim, *self.meas_dims])
            time_and_last_dims = torch.Size([1, -2, -1])
            if time_and_meas_dims != time_and_last_dims:
                x = torch.movedim(x, time_and_meas_dims, time_and_last_dims)

            return self._fast_measure(x)
        else:
            return super().measure(x)

    def measure_H(self, x: torch.tensor):
        r"""Simulates noiseless measurements leveraging the Kronecker structure of the 2d Hadamard transform H.

        Each measurement is acquired as, for :math:`k \in \{1, ..., M\}`:

        .. math::

            m_k = \sum_{i, j} H_{1d}[r_k, i] x_{t=k}[i, j] H_{1d}[j, c_k],

        where
        :math:`H_{1d} \in \mathbb{R}^{h\times h}` is the 1d Hadamard matrix,
        :math:`x_{t=k} \in \mathbb{R}^{h \times h}` is :math:`k^{\rm{th}}` frame of the video,
        :math:`(r_k, c_k) = (\left \lfloor k / h \right\rfloor, k \bmod h)` are the row and column indices of
        the 1d Hadamard matrix used to generate the 2d Hadamard pattern used at time :math:`t=k`.

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.meas import DynamicHadamSplit2d
            >>>
            >>> x = torch.rand([1, 2 * 32**2, 3, 40, 40])  # dummy RGB video with 2 * 32**2 frames of size 40x40
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicHadamSplit2d(time_dim=1, h=32, M=32**2, img_shape=(40, 40), noise_model=noise_op)  # acquisition with 2*M splitted Hadamard patterns of size hxh.
            >>> print(meas_op)
            DynamicHadamSplit2d(
              (noise_model): Poisson()
            )
            >>>
            >>> m = meas_op.measure_H(x)  # simulate noiseless dynamic measurements from matrix H
            >>> print(m.shape)
            torch.Size([1, 3, 1024])

        """
        if self.fast:
            x = x.movedim(self.time_dim, 0)
            x = (x[::2] + x[1::2]) / 2
            x = x.movedim(0, self.time_dim)

            x = spytorch.center_crop(x, self.meas_shape)

            time_and_meas_dims = torch.Size([self.time_dim, *self.meas_dims])
            time_and_last_dims = torch.Size([1, -2, -1])
            if time_and_meas_dims != time_and_last_dims:
                x = torch.movedim(x, time_and_meas_dims, time_and_last_dims)

            return self._fast_measure_H(x)
        else:
            return super().measure_H(x)

    def _fast_measure(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noiseless measurements leveraging the Kronecker structure of the 2d splitted Hadamard transform A.

        Each measurement is acquired as, for :math:`k \in \{1, ..., 2M\}`:

        .. math::

            y_k = \sum_{i, j} A_{1d}[r_k, i] x_{t=k}[i, j] A_{1d}[j, c_k],

        where
        :math:`A_{1d} \in \mathbb{R}_+^{2h\times h}` contains the positive and negative components of a 1d Hadamard matrix,
        :math:`x_{t=k} \in \mathbb{R}^{h \times h}` is :math:`k^{\rm{th}}` frame of the video,
        :math:`(r_k, c_k) = (\left \lfloor k / h \right\rfloor, k \bmod h)` are the row and column indices of the 1d Hadamard matrix used to generate the 2d Hadamard pattern used at time :math:`t=k`.

        """
        pattern_indices = self.indices[: self.M]

        # Find indices to 2D coordinates in the Hadamard sampling map (for separable transform)
        row_indices = pattern_indices // self.h
        col_indices = pattern_indices % self.h

        # Extract all required rows and columns from H1d
        H1d_rows = self.H1d[row_indices, :]  # shape (M, h)
        H1d_cols = self.H1d[:, col_indices]  # shape (h, M)

        # Split the 1D patterns into positive and negative parts
        H1d_rows_pos = nn.functional.relu(H1d_rows)  # shape (M, h)
        H1d_rows_neg = nn.functional.relu(-H1d_rows)  # shape (M, h)
        H1d_cols_pos = nn.functional.relu(H1d_cols)  # shape (h, M)
        H1d_cols_neg = nn.functional.relu(-H1d_cols)  # shape (h, M)

        # For split 2D Hadamard: H_pos = H1d_row_pos \otimes H1d_col_pos + H1d_row_neg \otimes H1d_col_neg
        #                        H_neg = H1d_row_pos \otimes H1d_col_neg + H1d_row_neg \otimes H1d_col_pos

        x_pos, x_neg = x[:, ::2], x[:, 1::2]

        # Compute the four separable components
        m_pp = torch.einsum("th,btchw,wt->bct", H1d_rows_pos, x_pos, H1d_cols_pos)
        m_nn = torch.einsum("th,btchw,wt->bct", H1d_rows_neg, x_pos, H1d_cols_neg)
        m_pn = torch.einsum("th,btchw,wt->bct", H1d_rows_pos, x_neg, H1d_cols_neg)
        m_np = torch.einsum("th,btchw,wt->bct", H1d_rows_neg, x_neg, H1d_cols_pos)

        # Combine to get positive and negative measurements
        y_pos = m_pp + m_nn
        y_neg = m_pn + m_np

        # Interleave positive and negative measurements: [pos0, neg0, pos1, neg1, ...]
        y = torch.stack([y_pos, y_neg], dim=-1)  # shape (b, c, M, 2)
        y = y.reshape(*y.shape[:-2], 2 * self.M)  # shape (b, c, 2*M)

        return y

    def _fast_measure_H(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noiseless measurements leveraging the Kronecker structure of the 2d splitted Hadamard transform H.

        Each measurement is acquired as, for :math:`k \in \{1, ..., M\}`:

        .. math::

            m_k = \sum_{i, j} H_{1d}[r_k, i] x_{t=k}[i, j] H_{1d}[j, c_k],

        where
        :math:`H_{1d} \in \mathbb{R}^{h\times h}` is the 1d Hadamard matrix,
        :math:`x_{t=k} \in \mathbb{R}^{h \times h}` is :math:`k^{\rm{th}}` frame of the video,
        :math:`(r_k, c_k) = (\left \lfloor k / h \right\rfloor, k \bmod h)` are the row and column indices of the 1d Hadamard matrix used to generate the 2d Hadamard pattern used at time :math:`t=k`.

        """
        pattern_indices = self.indices[: self.M]

        # Find indices to 2D coordinates in the Hadamard sampling map (for separable transform)
        row_indices = pattern_indices // self.h
        col_indices = pattern_indices % self.h

        # Extract all required rows and columns from H1d
        H1d_rows = self.H1d[row_indices, :]  # shape (M, h)
        H1d_cols = self.H1d[:, col_indices]  # shape (h, M)

        # Vectorized separable 2D transform using the kronecker structure
        # x shape: (b, t, c, h, w) -> we want (b, c, t)
        m = torch.einsum("th,btchw,wt->bct", H1d_rows, x, H1d_cols)

        return m

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noisy measurements leveraging the Kronecker structure of the 2d splitted Hadamard transform A.

        Each measurement is acquired as, for :math:`k \in \{1, ..., 2M\}`:

        .. math::

            y_k = \mathcal{N}\left( \sum_{i, j} A_{1d}[r_k, i] x_{t=k}[i, j] A_{1d}[j, c_k] \right),

        where
        :math:`A_{1d} \in \mathbb{R}_+^{2h\times h}` contains the positive and negative components of a 1d Hadamard matrix,
        :math:`x_{t=k} \in \mathbb{R}^{h \times h}` is :math:`k^{\rm{th}}` frame of the video,
        :math:`(r_k, c_k) = (\left \lfloor k / h \right\rfloor, k \bmod h)` are the row and column indices of the 1d Hadamard matrix used to generate the 2d Hadamard pattern used at time :math:`t=k`.

        Args:
            :attr:`x` (:class:`torch.tensor`): Video signal :math:`x` whose
            dimensions :attr:`self.meas_dims` must be of shape :attr:`self.meas_shape`
            and dimension :attr:`self.time_dim` must be of size :attr:`2 * self.M`.

        Returns:
            :class:`torch.tensor`: Measurement vector :math:`m` of length :attr:`2\*self.M`.

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.meas import DynamicHadamSplit2d
            >>>
            >>> x = torch.rand([1, 2 * 32**2, 3, 40, 40])  # dummy RGB video with 2 * 32**2 frames of size 40x40
            >>> alpha = 5  # noise level
            >>> noise_op = Poisson(alpha=alpha, g=1/alpha)
            >>> meas_op = DynamicHadamSplit2d(time_dim=1, h=32, M=32**2, img_shape=(40, 40), noise_model=noise_op)  # acquisition with 2*M splitted Hadamard patterns of size hxh.
            >>> print(meas_op)
            DynamicHadamSplit2d(
              (noise_model): Poisson()
            )
            >>>
            >>> y = meas_op(x)  # simulate noisy dynamic measurements
            >>> print(y.shape)
            torch.Size([1, 3, 2048])
        """

        # it is ok to use super().forward, because measure method has been redefined
        return super().forward(x)

    def forward_H(self, x: torch.tensor) -> torch.tensor:
        r"""Simulates noisy measurements leveraging the Kronecker structure of the 2d Hadamard transform H.

        Each measurement is acquired as, for :math:`k \in \{1, ..., M\}`:

        .. math::

            m_k = \mathcal{N}\left( \sum_{i, j} H_{1d}[r_k, i] x_{t=k}[i, j] H_{1d}[j, c_k] \right),

        where
        :math:`H_{1d} \in \mathbb{R}^{h\times h}` is the 1d Hadamard matrix,
        :math:`x_{t=k} \in \mathbb{R}^{h \times h}` is :math:`k^{\rm{th}}` frame of the video,
        :math:`(r_k, c_k) = (\left \lfloor k / h \right\rfloor, k \bmod h)` are the row and column indices of the 1d Hadamard matrix used to generate the 2d Hadamard pattern used at time :math:`t=k`.


        Args:
            :attr:`x` (:class:`torch.tensor`): Video signal :math:`x` whose
            dimensions :attr:`self.meas_dims` must be of shape :attr:`self.meas_shape`
            and dimension :attr:`self.time_dim` must be of size :attr:`2 * self.M`.

        Returns:
            :class:`torch.tensor`: Measurement vector :math:`m` of length :attr:`self.M`.

        Example:
            >>> import torch
            >>> from spyrit.core.noise import Poisson
            >>> from spyrit.core.meas import DynamicHadamSplit2d
            >>>
            >>> x = torch.rand([1, 2 * 32**2, 3, 40, 40])  # dummy RGB video with 2 * 32**2 frames of size 40x40
            >>> noise_op = torch.nn.Identity()  # We can't use Poisson here because measurements from H can be negative
            >>> meas_op = DynamicHadamSplit2d(time_dim=1, h=32, M=32**2, img_shape=(40, 40), noise_model=noise_op)  # acquisition with 2*M splitted Hadamard patterns of size hxh.
            >>> print(meas_op)
            DynamicHadamSplit2d(
              (noise_model): Identity()
            )
            >>>
            >>> m = meas_op.forward_H(x)  # simulate noisy dynamic measurements from matrix H
            >>> print(m.shape)
            torch.Size([1, 3, 1024])

        """

        # it is ok to use super().forward_H, because measure_H method has been redefined
        return super().forward_H(x)
