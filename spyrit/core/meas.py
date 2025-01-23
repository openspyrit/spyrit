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
from typing import Any, Union
from collections.abc import Iterable

# import memory_profiler as mprof

import math
import torch
import torch.nn as nn

from spyrit.core.warp import DeformationField
import spyrit.core.torch as spytorch

# =============================================================================
class Linear(nn.Module):
    r"""
    Simulates linear measurements

    .. math::
        m =\mathcal{N}\left(Hx\right),
        
    where :math:`\mathcal{N} \colon\, \mathbb{R}^M \to \mathbb{R}^M` represents a noise operator (e.g., Gaussian), :math:`H \colon\, \mathbb{R}^N \to \mathbb{R}^M` is the acquisition matrix, :math:`M` is the number of measurements, :math:`N` is the dimension of the signal, and :math:`x \in \mathbb{R}^N` is the signal of interest.
    
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
        :attr:`H` (:class:`torch.tensor`): (Learnable) measurement matrix of shape
        :math:`(M, N)` initialized as :math:`H`.
        
        :attr:`meas_shape` (tuple): Shape of the underliying 
        multi-dimensional array :math:`X`.
        
        :attr:`meas_dims` (tuple): Dimensions the acquisition matrix applies to. 
        
        :attr:`meas_ndim` (int): Number of dimensions the 
        acquisition matrix applies to. This is `len(meas_dims)` 
    
        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model :math:`\mathcal{N}`.
    
        :attr:`M` (int): Number of measurements :math:`M`.
    
    Example: (to be updated!)
        Example 1:
            
        >>> H = torch.rand([400, 1600])
        >>> meas_op = Linear(H)
        >>> print(meas_op)

        Example 2:
            
        >>> H = torch.rand([400, 1600])
        >>> meas_op = Linear(H, True)
        >>> print(meas_op)
    """

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
        if meas_dims is None:
            meas_dims = list(range(len(meas_shape)))

        if type(meas_shape) is int:
            meas_shape = [meas_shape]
        if type(meas_dims) is int:
            meas_dims = [meas_dims]

        #H = H.to(device=device, dtype=dtype)

        # don't store H if we use a HadamSplit
        if not isinstance(self, HadamSplit2d):
            self.H = nn.Parameter(H, requires_grad=False)
        self.meas_shape = torch.Size(meas_shape)
        self.meas_dims = torch.Size(meas_dims)
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
        return self.H.device

    @property
    def dtype(self) -> torch.dtype:
        return self.H.dtype

    @property
    def matrix_to_inverse(self) -> str:
        return self._selected_pinv_matrix

    @property
    def get_matrix_to_inverse(self) -> torch.tensor:
        return getattr(self, self._selected_pinv_matrix)

    def measure(self, x: torch.tensor) -> torch.tensor:
        r"""Apply the measurement patterns (no noise) to the incoming tensor.

        The input tensor is multiplied by the measurement patterns.

        .. note::
            This method does not include the noise model.

        Args:
            x (torch.tensor): A tensor where the dimensions indexed by
            `self.meas_dims` match the measurement shape `self.meas_shape`.

        Returns:
            torch.tensor: A tensor of shape (*, self.M) where * denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example with a RGB 15x4 pixel image:
            >>> matrix = torch.randn(10, 60)
            >>> meas_op = Linear(matrix, meas_shape=(15, 4))
            >>> x = torch.randn(3, 15, 4)
            >>> y = meas_op.measure(x)
            >>> print(y.shape)
            torch.Size([3, 10])
        """
        x = self.vectorize(x)
        x = torch.einsum("mn,...n->...m", self.H, x)
        return x

    def forward(self, x: torch.tensor):
        r"""Forward pass (measurement + noise) of the measurement operator.

        The forward pass includes both the measurement and the noise model. It
        is equivalent to the method `measure()` followed by the noise model
        :meth:`forward()` method.

        Args:
            x (torch.tensor): A tensor where the dimensions indexed by
            `self.meas_dims` match the measurement shape `self.meas_shape`.

        Returns:
            torch.tensor: A tensor of shape (*, self.M) where * denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example:
            >>> matrix = torch.randn(10, 60)
            >>> meas_op = Linear(matrix, meas_shape=(15, 4)
        """
        x = self.measure(x)
        x = self.noise_model(x)
        return x

    def adjoint(self, y: torch.tensor, unvectorize=False):
        r"""Applies the adjoint (transpose) of the measurement matrix.

        Args:
            y (torch.tensor): A tensor of shape (*, self.M) where * denotes
            0 or more batch dimensions.

            unvectorize (bool, optional): Whether to call :meth:`unvectorize`
            after the operation. Defaults to False.

        Returns:
            torch.tensor: A tensor of shape (*, self.N) if `reshape_output` is
            False, or the :meth:`unvectorize`d version of that tensor.
        """
        y = torch.einsum("mn,...m->...n", self.H, y)
        if unvectorize:
            y = self.unvectorize(y)
        return y

    def unvectorize(self, input: torch.tensor) -> torch.tensor:
        r"""Unflatten the last dimension of a tensor to the measurement shape at
        the measured dimensions.

        This method first expands the last dimension into the measurement
        shape (`self.meas_shape`), and then moves the expanded dimensions to
        their original positions as defined by `self.meas_dims`.

        Input:
            input (torch.tensor): A tensor of shape (*, self.N) where
            * denotes any batch size and `self.N` is the number of
            measured items (pixels for instance).

        Output:
            torch.tensor: A tensor where the dimensions indexed by
            `self.meas_dims` match the measurement shape `self.meas_shape`.

        Example:
            >>> matrix = torch.randn(10, 60)
            >>> meas_op = Linear(matrix, meas_shape=(12, 5), dim=(-1,-3))
            >>> x = torch.randn(3, 7, 60)
            >>> print(meas_op.unvectorize(x).shape)
            torch.Size([3, 5, 7, 12]
        """
        # unvectorize the last dimension
        input = input.reshape(*input.shape[:-1], *self.meas_shape)
        # move the measured dimensions to their original positions
        if self.meas_dims != self.last_dims:
            input = torch.movedim(input, self.last_dims, self.meas_dims)
        return input

    def vectorize(self, input: torch.tensor) -> torch.tensor:
        r"""Flatten a tensor along the measured dimensions `self.meas_dims`.

        The tensor is flattened at the indicated `self.meas_dims` dimensions. The
        flattened dimensions are then collapsed into one, which is the last
        dimension of the output tensor.

        Input:
            input (torch.tensor): A tensor where the dimensions indexed by
            `self.meas_dims` match the measurement shape `self.meas_shape`.

        Output:
            torch.tensor: A tensor of shape (*, self.N) where * denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example:
            >>> matrix = torch.randn(10, 60)
            >>> meas_op = Linear(matrix, meas_shape=(12, 5), dim=(-1,-3))
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


# =============================================================================
class FreeformLinear(Linear):
    r"""Performs linear measurements on a subset (mask) of pixels in the image."""

    def __init__(
        self,
        H: torch.tensor,
        meas_shape: Union[int, torch.Size, Iterable[int]] = None,
        meas_dims: Union[int, torch.Size, Iterable[int]] = None,
        index_mask: torch.tensor = None,  # must have shape (len(meas_shape), H.shape[-1])
        bool_mask: torch.tensor = None,
        *,
        noise_model: bool = None,
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

        # select mask type
        if index_mask is not None:
            if bool_mask is not None:
                warnings.warn(
                    "Both index_mask and bool_mask have been specified. Using index_mask."
                )
            self.index_mask = index_mask
            self.mask_type = "index"
        else:
            if bool_mask is not None:
                self.bool_mask = bool_mask
                self.mask_type = "bool"
            else:
                raise ValueError("Either index_mask or bool_mask must be specified.")

        # check mask dimensions in the case of index mask
        if self.mask_type == "index":
            if index_mask.ndim != 2:
                raise ValueError("index_mask must have 2 dimensions.")
            if index_mask.shape[0] != len(meas_shape):
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

    def apply_mask(self, x: torch.tensor) -> torch.tensor:
        r"""Appplies the saved mask to the input tensor, where the masked
        dimensions are collapsed into one.

        This method first selects the elements from the input tensor at the
        specified dimensions `self.meas_dims` and based on the mask. The selected
        elements are then flattened into a single dimension which is the last
        dimension of the output tensor.

        Args:
            x (torch.tensor): The input tensor to select the mask from. The
            dimensions indexed by `self.meas_dims` should match the measurement shape
            `self.meas_shape`.

        Returns:
            torch.tensor: A tensor of shape (*, self.N) where * denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example: Select one every second point on the diagonal of a batch of images
            >>> images = torch.rand(17, 3, 40, 40)  # b, c, h, w
            >>> # create a (2,20) mask
            >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
            >>> H = torch.randn(13, 20)
            >>> meas_op = FreeformLinear(H, mask, meas_shape=(40,40), dim=(-1,-2))
            >>> y = meas_op.apply_mask(images)
            >>> print(y.shape)
            torch.Size([17, 3, 20])
        """
        x = torch.movedim(x, self.meas_dims, self.last_dims)

        if self.mask_type == "index":
            return x[(..., *self.index_mask)]

        elif self.mask_type == "bool":
            # flatten along the masked dimensions
            x = x.reshape(*x.shape[: -self.meas_ndim], self.N)
            return x[..., self.bool_mask.reshape(-1)]

        else:
            raise ValueError(
                f"mask_type must be either 'index' or 'bool', found {self.mask_type}."
            )

    def measure(self, x: torch.tensor) -> torch.tensor:
        r"""Apply the measurement patterns (no noise) to the incoming tensor.

        The mask is first applied to the input tensor, then the input tensor
        is multiplied by the measurement patterns.

        .. note::
            This method does not include the noise model.

        Args:
            x (torch.tensor): A tensor where the dimensions indexed by
            `self.meas_dims` match the measurement shape `self.meas_shape`.

        Returns:
            torch.tensor: A tensor of shape (*, self.M) where * denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.
        """
        x = self.mask_vectorize(x)
        return torch.einsum("mn,...n->...m", self.H, x)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Forward pass (measurement + noise) of the measurement operator.

        The mask is first applied to the input tensor, then the input tensor
        goes through the measurement model and the noise model. It
        is equivalent to the method `measure()` followed by the noise model
        :meth:`forward()` method.

        Args:
            x (torch.tensor): A tensor where the dimensions indexed by
            `self.meas_dims` match the measurement shape `self.meas_shape`.

        Returns:
            torch.tensor: A tensor of shape (*, self.M) where * denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example: Measure the upper half of 32x32 images
            >>> H = torch.randn(10, 16*32)
            >>>

        """
        super().forward(x)

    def mask_unvectorize(self, x: torch.tensor, fill_value: Any = 0) -> torch.tensor:
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
            x (torch.tensor): tensor to be expanded. Its last dimension must
            contain `self.N` elements.

            fill_value (Any, optional): Fill value for all the indices not
            covered by the mask. Defaults to 0.

        Returns:
            torch.tensor: A tensor where the dimensions indexed by `self.meas_dims`
            match the measurement shape `self.meas_shape`.
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

    def mask_vectorize(self, x: torch.tensor) -> torch.tensor:
        r"""Flatten a tensor along the measured dimensions, which are collapsed into one.

        This method first selects the elements from the input tensor at the
        specified dimensions `self.meas_dims` and based on the mask. The selected
        elements are then flattened into a single dimension which is the last
        dimension of the output tensor.

        Args:
            x (torch.tensor): The input tensor to select the mask from. The
            dimensions indexed by `self.meas_dims` should match the measurement shape
            `self.meas_shape`.

        .. note::
            This function is an alias for the method :meth:`apply_mask`.

        Returns:
            torch.tensor: A tensor of shape (*, self.N) where * denotes
            all the dimensions of the input tensor not included in `self.meas_dims`.

        Example: Select one every second point on the diagonal of a batch of images
            >>> images = torch.rand(17, 3, 40, 40)  # b, c, h, w
            >>> # create a (2,20) mask
            >>> mask = torch.tensor([[i, i] for i in range(0,40,2)]).T
            >>> H = torch.randn(13, 20)
            >>> meas_op = FreeformLinear(H, mask, meas_shape=(40,40), dim=(-1,-2))
            >>> y = meas_op.mask_vectorize(images)
            >>> print(y.shape)
            torch.Size([17, 3, 20])
        """
        return self.apply_mask(x)


# =============================================================================
class LinearSplit(Linear):
    r"""
    Simulate linear measurements by splitting an acquisition matrix :math:`H\in \mathbb{R}^{M\times N}` that contains negative values. In pratice, only positive values can be implemented using a DMD. Therefore, we acquire
 
    .. math::
        y =\mathcal{N}\left(Ax\right),
        
    where :math:`\mathcal{N} \colon\, \mathbb{R}^{2M} \to \mathbb{R}^{2M}` represents a noise operator (e.g., Gaussian), :math:`A \colon\, \mathbb{R}_+^{2M\times N}` is the acquisition matrix that contains positive DMD patterns, :math:`x \in \mathbb{R}^N` is the signal of interest., :math:`2M` is the number of DMD patterns, and :math:`N` is the dimension of the signal.
    
    Given a matrix :math:`H`, we define the positive DMD patterns :math:`A` from the positive and negative components :math:`H`. In practice, the even rows of :math:`A` contain the positive components of :math:`H`, while odd rows of :math:`A` contain the negative components of :math:`H`. Mathematically,

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
        :attr:`A` (:class:`torch.tensor`): (Learnable) measurement matrix of shape
        :math:`(2M, N)` initialized as :math:`A`.
        
        :attr:`H` (:class:`torch.tensor`): (Learnable) measurement matrix of shape
        :math:`(M, N)` initialized as :math:`H`.
        
        :attr:`meas_shape` (tuple): Shape of the underliying 
        multi-dimensional array :math:`X`.
        
        :attr:`meas_dims` (tuple): Dimensions the acquisition matrix applies to. 
        
        :attr:`meas_ndim` (int): Number of dimensions the 
        acquisition matrix applies to. This is `len(meas_dims)` 
    
        :attr:`noise_model` (see :mod:`spyrit.core.noise`): Noise model :math:`\mathcal{N}`.
    
        :attr:`M` (int): Number of measurements :math:`M`.
    
    Example: (to be updated!)
        Example 1:
            
        >>> H = torch.rand([400, 1600])
        >>> meas_op = Linear(H)
        >>> print(meas_op)

        Example 2:
            
        >>> H = torch.rand([400, 1600])
        >>> meas_op = Linear(H, True)
        >>> print(meas_op)
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

    @property
    def device(self) -> torch.device:
        if self.H.device == self.A.device:
            return self.H.device
        else:
            raise RuntimeError(
                f"device undefined, H and A are on different device (found {self.H.device} and {self.A.device} respectively)"
            )

    @property
    def dtype(self) -> torch.dtype:
        if self.H.dtype == self.A.dtype:
            return self.H.dtype
        else:
            raise RuntimeError(
                f"dtype undefined, H and A are of different dtype (found {self.H.dtype} and {self.A.dtype} respectively)"
            )

    def set_matrix_to_inverse(self, matrix_name: str) -> None:
        if matrix_name in self._available_pinv_matrices.keys():
            self._selected_pinv_matrix = matrix_name
        else:
            raise KeyError(
                f"Matrix {matrix_name} not available for pinv. Available matrices: {self._available_pinv_matrices.keys()}"
            )

    def measure(self, x: torch.tensor):
        r""" """
        x = self.vectorize(x)
        x = torch.einsum("mn,...n->...m", self.A, x)
        return x

    def measure_H(self, x: torch.tensor):
        r""" """
        x = self.vectorize(x)
        x = torch.einsum("mn,...n->...m", self.H, x)
        return x

    def adjoint(self, y: torch.tensor):
        r""" """
        y = torch.einsum("mn,...m->...n", self.A, y)
        return y

    def adjoint_H(self, y: torch.tensor):
        r""" """
        y = torch.einsum("mn,...m->...n", self.H, y)
        return y

    def forward(self, x: torch.tensor):
        r""" """
        x = self.measure(x)
        x = self.noise_model(x)
        return x

    def forward_H(self, x: torch.tensor):
        r""" """
        x = self.measure_H(x)
        x = self.noise_model(x)
        return x


# =============================================================================
class HadamSplit2d(LinearSplit):
    r""" """

    def __init__(
        self,
        M: int,
        h: int,
        order: torch.tensor = None,
        fast: bool = True,
        *,
        noise_model=nn.Identity(),
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        meas_dims = (-2, -1)
        meas_shape = (h, h)
        # 1D version of H
        self.H1d = spytorch.walsh_matrix(h).to(dtype=dtype, device=device)

        # call Linear constructor (avoid setting A)
        super(LinearSplit, self).__init__(
            torch.empty(h**2, h**2),
            meas_shape,
            meas_dims,
            noise_model=noise_model,
            dtype=dtype,
            device=device,
        )
        self.M = M
        self.order = order
        self.indices = torch.argsort(-order.flatten(), stable=True).to(
            dtype=torch.int32, device=self.device
        )
        self.fast = fast

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

    def measure(self, x: torch.tensor) -> torch.tensor:
        r""""""
        if self.fast:
            return self.fast_measure(x)
        else:
            return super().measure(x)

    def measure_H(self, x: torch.tensor):
        r""" """
        if self.fast:
            return self.fast_measure_H(x)
        else:
            return super().measure_H(x)

    def adjoint_H(self, y: torch.tensor) -> torch.tensor:
        r""""""
        if self.fast:
            return self.fast_pinv(y) * self.N
        else:
            return super().adjoint_H(y)

    def fast_measure(self, x: torch.tensor) -> torch.tensor:
        r""" """
        Hx = self.measure_H(x)
        x_sum = Hx[..., 0]
        y_pos, y_neg = (x_sum + Hx) / 2, (x_sum - Hx) / 2
        new_shape = y_pos.shape[:-1] + (2 * self.M,)
        y = torch.stack([y_pos, y_neg], -1).reshape(new_shape)
        return y

    def fast_measure_H(self, x: torch.tensor) -> torch.tensor:
        r""" """
        x = spytorch.mult_2d_separable(self.H1d, x)
        x = self.vectorize(x)
        x = x.index_select(dim=-1, index=self.indices)
        # x = self.reindex(x, "rows", False)
        return x[..., : self.M]

    def fast_pinv(self, y: torch.tensor):
        r""" """
        if self.N != self.M:
            y = torch.cat(
                (y, torch.zeros(*y.shape[:-1], self.N - self.M, device=y.device)),
                -1,
            )
        y = self.reindex(y, "cols", False)
        y = self.unvectorize(y)
        y = spytorch.mult_2d_separable(self.H1d, y)
        return y / self.N

    def fast_H_pinv(self) -> torch.tensor:
        r""" """
        return self.H.T / self.N


# =============================================================================
# class LinearSplit(Linear):

#     # `y = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}x`
#     r"""
#     Simulates split measurements.

#     Given a measurement operator :math:`H\in\mathbb{R}^{M\times N}` containing (possibly) negative values, it computes

#     .. math:: y = Ax,

#     where :math:`x` is the signal/image and  :math:`A\in\mathbb{R}_+^{2M\times N}` is the split measurement operator (associated to :math:`H`) which contains only positive values.

#     We define the split measurement operator from the positive and negative components of :math:`H`. In practice, the even rows of :math:`A` contain the positive components of :math:`H`, while odd rows of :math:`A` contain the negative components of :math:`H`. Mathematically,

#     .. math::
#         \begin{cases}
#             A[0::2, :] = H_{+}, \text{ with } H_{+} = \max(0,H),\\
#             A[1::2, :] = H_{-}, \text{ with } H_{-} = \max(0,-H).
#         \end{cases}

#     .. note::
#         :math:`H_{+}` and :math:`H_{-}` are such that :math:`H_{+} - H_{-} = H`.

#     The class is constructed from the :math:`M` by :math:`N` matrix :math:`H`,
#     where :math:`N` represents the number of pixels in the image and
#     :math:`M` the number of measurements.

#     Args:
#         :attr:`H` (:class:`torch.tensor`): measurement matrix (linear operator)
#         with shape :math:`(M, N)`. Only real values are supported.

#         :attr:`pinv` (bool): Whether to store the pseudo inverse of the
#         measurement matrix :math:`H`. If `True`, the pseudo inverse is
#         initialized as :math:`H^\dagger` and stored in the attribute
#         :attr:`H_pinv`. It is always possible to compute and store the pseudo
#         inverse later using the method :meth:`build_H_pinv`. Defaults to
#         :attr:`False`.

#         :attr:`rtol` (float, optional): Cutoff for small singular values (see
#         :mod:`torch.linalg.pinv`). Only relevant when :attr:`pinv` is
#         :attr:`True`.

#         :attr:`Ord` (torch.tensor, optional): Order matrix used to reorder the
#         rows of the measurement matrix :math:`H`. The first new row of :math:`H`
#         will correspond to the highest value in :attr:`Ord`. Must contain
#         :math:`M` values. If some values repeat, the order is kept. Defaults to
#         :attr:`None`.

#         :attr:`meas_shape` (tuple, optional): Shape of the measurement patterns.
#         Must be a tuple of two integers representing the height and width of the
#         patterns. If not specified, the shape is suppposed to be a square image.
#         If not, an error is raised. Defaults to :attr:`None`.

#     Attributes:
#         :attr:`H` (torch.tensor): The learnable measurement matrix of shape
#         :math:`(M, N)` initialized as :math:`H`.

#         :attr:`H_static` (torch.tensor): alias for :attr:`H`.

#         :attr:`P` (torch.tensor): The splitted measurement matrix of shape
#         :math:`(2M, N)`.

#         :attr:`H_pinv` (torch.tensor, optional): The learnable pseudo inverse
#         measurement matrix :math:`H^\dagger` of shape :math:`(N, M)`.

#         :attr:`M` (int): Number of measurements performed by the linear operator.

#         :attr:`N` (int): Number of pixels in the image.

#         :attr:`h` (int): Measurement pattern height.

#         :attr:`w` (int): Measurement pattern width.

#         :attr:`meas_shape` (tuple): Shape of the measurement patterns
#         (height, width). Is equal to :attr:`(self.h, self.w)`.

#         :attr:`indices` (torch.tensor): Indices used to sort the rows of H.	It
#         is used by the method :meth:`reindex()`.

#         :attr:`Ord` (torch.tensor): Order matrix used to sort the rows of H. It
#         is used by :func:`~spyrit.core.torch.sort_by_significance()`.

#     .. note::
#         If you know the pseudo inverse of :math:`H` and want to store it,
#         instantiate the class with :attr:`pinv` set to `False` and call
#         :meth:`build_H_pinv` to store the pseudo inverse.

#     Example:
#         >>> H = torch.randn(400, 1600)
#         >>> meas_op = LinearSplit(H, False)
#         >>> print(meas_op)
#         LinearSplit(
#           (M): 400
#           (N): 1600
#           (H.shape): torch.Size([400, 1600])
#           (meas_shape): (40, 40)
#           (H_pinv): False
#           (P.shape): torch.Size([800, 1600])
#         )
#     """

#     def __init__(
#         self,
#         H: torch.tensor,
#         pinv: bool = False,
#         rtol: float = None,
#         Ord: torch.tensor = None,
#         meas_shape: tuple = None,  # (height, width)
#     ):
#         super().__init__(H, pinv, rtol, Ord, meas_shape)
#         self._set_P(self.H_static)

#     def forward(self, x: torch.tensor) -> torch.tensor:
#         r"""Applies linear transform to incoming images: :math:`y = Px`.

#         This is equivalent to computing :math:`x \cdot P^T`. The input images
#         must be unvectorized. The matrix :math:`P` is obtained by splitting
#         the measurement matrix :math:`H` such that :math:`P` has a shape of
#         :math:`(2M, N)` and `P[0::2, :] = H_{+}` and `P[1::2, :] = H_{-}`,
#         where :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

#         .. warning::
#             This method uses the splitted measurement matrix :math:`P` to compute
#             the linear measurements from incoming images. If you want to apply
#             the operator :math:`H` directly, use the method :meth:`forward_H`.

#         Args:
#             :math:`x` (torch.tensor): Batch of images of shape :math:`(*, h, w)`.
#             `*` can have any number of dimensions, for instance `(b, c)` where
#             `b` is the batch size and `c` the number of channels. `h` and `w`
#             are the height and width of the images.

#         Output:
#             torch.tensor: The linear measurements of the input images. It has
#             shape :math:`(*, 2M)` where * denotes any number of dimensions and
#             `M` the number of measurements as defined by the parameter :attr:`M`,
#             which is equal to the number of rows in the measurement matrix :math:`H`
#             defined at initialization.

#         Shape:
#             :math:`x`: :math:`(*, N)` where * denotes the batch size and `N`
#             the total number of pixels in the image.

#             Output: :math:`(*, 2M)` where * denotes the batch size and `M`
#             the number of measurements as defined by the parameter :attr:`M`,
#             which is equal to the number of rows in the measurement matrix :math:`H`
#             defined at initialization.

#         Example:
#             >>> H = torch.randn(400, 1600)
#             >>> meas_op = LinearSplit(H)
#             >>> x = torch.randn(10, 40, 40)
#             >>> y = meas_op(x)
#             >>> print(y.shape)
#             torch.Size([10, 800])
#         """
#         # return x @ self.P.T.to(x.dtype)
#         return self._static_forward_with_op(x, self.P)

#     def forward_H(self, x: torch.tensor) -> torch.tensor:
#         r"""Applies linear transform to incoming images: :math:`m = Hx`.

#         This is equivalent to computing :math:`x \cdot H^T`. The input images
#         must be unvectorized.

#         .. warning::
#             This method uses the measurement matrix :math:`H` to compute the linear
#             measurements from incoming images. If you want to apply the splitted
#             operator :math:`P`, use the method :meth:`forward`.

#         Args:
#             :attr:`x` (torch.tensor): Batch of images of shape :math:`(*, h, w)`.
#             `*` can have any number of dimensions, for instance `(b, c)` where
#             `b` is the batch size and `c` the number of channels. `h` and `w`
#             are the height and width of the images.

#         Output:
#             torch.tensor: The linear measurements of the input images. It has
#             shape :math:`(*, M)` where * denotes any number of dimensions and
#             `M` the number of measurements.

#         Shape:
#             :attr:`x`: :math:`(*, h, w)` where * denotes the batch size and `h`
#             and `w` are the height and width of the images.

#             Output: :math:`(*, M)` where * denotes the batch size and `M`
#             the number of measurements.

#         Example:
#             >>> H = torch.randn(400, 1600)
#             >>> meas_op = LinearSplit(H)
#             >>> x = torch.randn(10, 40, 40)
#             >>> y = meas_op.forward_H(x)
#             >>> print(y.shape)
#             torch.Size([10, 400])
#         """
#         # call Linear.forward() method
#         return super().forward(x)

#     def _set_Ord(self, Ord):
#         super()._set_Ord(Ord)
#         self._set_P(self.H_static)


# # =============================================================================
# class HadamSplit(LinearSplit):
#     # =========================================================================
#     r"""
#     Simulates splitted measurements :math:`y = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}x`
#     with :math:`H` a Hadamard matrix.

#     Computes linear measurements from incoming images: :math:`y = Px`,
#     where :math:`P` is a linear operator (matrix) and :math:`x` is a
#     vectorized image or batch of vectorized images.

#     The matrix :math:`P` contains only positive values and is obtained by
#     splitting a Hadamard-based matrix :math:`H` such that
#     :math:`P` has a shape of :math:`(2M, N)` and `P[0::2, :] = H_{+}` and
#     `P[1::2, :] = H_{-}`, where :math:`H_{+} = \max(0,H)` and
#     :math:`H_{-} = \max(0,-H)`.

#     :math:`H` is obtained by selecting a re-ordered subsample of :math:`M` rows
#     of a "full" Hadamard matrix :math:`F` with shape :math:`(N^2, N^2)`.
#     :math:`N` must be a power of 2.

#     Args:
#         :attr:`M` (int): Number of measurements. It determines the size of the
#         Hadamard matrix subsample :math:`H`.

#         :attr:`h` (int): Measurement pattern height. The width is taken to be
#         equal to the height, so the measurement pattern is square. The Hadamard
#         matrix will have shape :math:`(h^2, h^2)`.

#         :attr:`Ord` (torch.tensor, optional): Order matrix used to reorder the
#         rows of the measurement matrix :math:`H`. The first new row of :math:`H`
#         will correspond to the highest value in :math:`Ord`. Must contain
#         :math:`M` values. If some values repeat, the order is kept. Defaults to
#         None.

#     Attributes:
#         :attr:`H` (torch.tensor): The learnable measurement matrix of shape
#         :math:`(M, N)`.

#         :attr:`H_static` (torch.tensor): alias for :attr:`H`.

#         :attr:`P` (torch.tensor): The splitted measurement matrix of shape
#         :math:`(2M, N)`.

#         :attr:`H_pinv` (torch.tensor, optional): The learnable pseudo inverse
#         measurement matrix :math:`H^\dagger` of shape :math:`(N, M)`.

#         :attr:`M` (int): Number of measurements performed by the linear operator.
#         Is equal to the parameter :attr:`M`.

#         :attr:`N` (int): Number of pixels in the image, is equal to :math:`h^2`.

#         :attr:`h` (int): Measurement pattern height.

#         :attr:`w` (int): Measurement pattern width. Is equal to :math:`h`.

#         :attr:`meas_shape` (tuple): Shape of the measurement patterns
#         (height, width). Is equal to `(self.h, self.h)`.

#         :attr:`indices` (torch.tensor): Indices used to sort the rows of H.	It
#         is used by the method :meth:`reindex()`.

#         :attr:`Ord` (torch.tensor): Order matrix used to sort the rows of H. It
#         is used by :func:`~spyrit.core.torch.sort_by_significance()`.

#     .. note::
#         The computation of a Hadamard transform :math:`Fx` benefits a fast
#         algorithm, as well as the computation of inverse Hadamard transforms.

#     .. note::
#         :math:`H = H_{+} - H_{-}`

#     Example:
#         >>> h = 32
#         >>> Ord = torch.randn(h, h)
#         >>> meas_op = HadamSplit(400, h, Ord)
#         >>> print(meas_op)
#         HadamSplit(
#           (M): 400
#           (N): 1024
#           (H.shape): torch.Size([400, 1024])
#           (meas_shape): (32, 32)
#           (H_pinv): True
#           (P.shape): torch.Size([800, 1024])
#         )
#     """

#     def __init__(
#         self,
#         M: int,
#         h: int,
#         Ord: torch.tensor = None,
#     ):

#         F = spytorch.walsh_matrix_2d(h)

#         # we pass the whole F matrix to the constructor, but override the
#         # calls self.H etc to only return the first M rows
#         super().__init__(F, pinv=False, Ord=Ord, meas_shape=(h, h))
#         self._M = M

#     @property
#     def H_pinv(self) -> torch.tensor:
#         return self._param_H_static_pinv.data / self.N

#     @H_pinv.setter
#     def H_pinv(self, value: torch.tensor) -> None:
#         self._param_H_static_pinv = nn.Parameter(
#             value.to(torch.float64), requires_grad=False
#         )

#     @H_pinv.deleter
#     def H_pinv(self) -> None:
#         del self._param_H_static_pinv

#     def forward_H(self, x: torch.tensor) -> torch.tensor:
#         r"""Optimized measurement simulation using the Fast Hadamard Transform.

#         The 2D fast Walsh-ordered Walsh-Hadamard transform is applied to the
#         incoming images :math:`x`. This is equivalent to computing :math:`x \cdot H^T`.

#         Args:
#             :math:`x` (torch.tensor): Batch of images of shape :math:`(*,h,w)`.
#             `*` denotes any dimension, for instance `(b,c)` where `b` is the
#             batch size and `c` the number of channels. `h` and `w` are the height
#             and width of the images.

#         Output:
#             torch.tensor: The linear measurements of the input images. It has
#             shape :math:`(*,M)` where * denotes any number of dimensions and
#             `M` the number of measurements.

#         Shape:
#             :math:`x`: :math:`(*,h,w)` where * denotes any dimension, for
#             instance `(b,c)` where `b` is the batch size and `c` the number of
#             channels. `h` and `w` are the height and width of the images.

#             Output: :math:`(*,M)` where * denotes denotes any number of
#             dimensions and `M` the number of measurements.
#         """
#         m = spytorch.fwht_2d(x)
#         m_flat = self.vectorize(m)
#         return self.reindex(m_flat, "cols", True)[..., : self.M]

#     def build_H_pinv(self):
#         """Build the pseudo-inverse (inverse) of the Hadamard matrix H.

#         This computes the pseudo-inverse of the Hadamard matrix H, and stores it
#         in the attribute H_pinv. In the case of an invertible matrix, the
#         pseudo-inverse is the inverse.

#         Args:
#             None.

#         Returns:
#             None. The pseudo-inverse is stored in the attribute H_pinv.
#         """
#         # the division by self.N is done in the property so as to avoid
#         # memory overconsumption
#         self.H_pinv = self.H.T

#     def pinv(self, y, *args, **kwargs):
#         y_padded = torch.zeros(y.shape[:-1] + (self.N,), device=y.device, dtype=y.dtype)
#         y_padded[..., : self.M] = y
#         return self.inverse(y_padded)

#     def inverse(self, y: torch.tensor) -> torch.tensor:
#         r"""Inverse transform of Hadamard-domain images.

#         It can be described as :math:`x = H_{had}^{-1}G y`, where :math:`y` is
#         the input Hadamard-domain measurements, :math:`H_{had}^{-1}` is the inverse
#         Hadamard transform, and :math:`G` is the reordering matrix.

#         .. note::
#             For this inverse to work, the input vector must have the same number
#             of measurements as there are pixels in the original image
#             (:math:`M = N`), i.e. no subsampling is allowed.

#         .. warning::
#             This method is deprecated and will be removed in a future version.
#             Use self.pinv instead.

#         Args:
#             :math:`y`: batch of images in the Hadamard domain of shape
#             :math:`(*,c,M)`. `*` denotes any size, `c` the number of
#             channels, and `M` the number of measurements (with `M = N`).

#         Output:
#             :math:`x`: batch of images of shape :math:`(*,c,h,w)`. `*` denotes
#             any size, `c` the number of channels, and `h`, `w` the height and
#             width of the image (with `h \times w = N = M`).

#         Shape:
#             :math:`y`: :math:`(*, c, M)` with :math:`*` any size,
#             :math:`c` the number of channels, and :math:`N` the number of
#             measurements (with `M = N`).

#             Output: math:`(*, c, h, w)` with :math:`h` and :math:`w` the height
#             and width of the image.

#         Example:
#             >>> h = 32
#             >>> Ord = torch.randn(h, h)
#             >>> meas_op = HadamSplit(400, h, Ord)
#             >>> y = torch.randn(10, h**2)
#             >>> x = meas_op.inverse(y)
#             >>> print(x.shape)
#             torch.Size([10, 32, 32])
#         """
#         # permutations
#         y = self.reindex(y, "cols", False)
#         y = self.unvectorize(y)
#         # inverse of full transform
#         x = 1 / self.N * spytorch.fwht_2d(y, True)
#         return x

#     def _pinv_mult(self, y):
#         """We use fast walsh-hadamard transform to compute the pseudo inverse.

#         Args:
#             y (torch.tensor): batch of images in the Hadamard domain of shape
#             (*,M). * denotes any size, and M the number of measurements.

#         Returns:
#             torch.tensor: batch of images in the image domain of shape (*,N).
#         """
#         # zero-pad the measurements until size N
#         y_shape = y.shape
#         y_new_shape = y_shape[:-1] + (self.N,)
#         y_new = torch.zeros(y_new_shape, device=y.device, dtype=y.dtype)
#         y_new[..., : y_shape[-1]] = y

#         # unsort the measurements
#         y_new = self.reindex(y_new, "cols", False)
#         y_new = self.unvectorize(y_new)

#         # inverse of full transform
#         return 1 / self.N * spytorch.fwht_2d(y_new, True)

#     def _set_Ord(self, Ord: torch.tensor) -> None:
#         """Set the order matrix used to sort the rows of H."""
#         # get only the indices, as done in spyrit.core.torch.sort_by_significance
#         self._indices = torch.argsort(-Ord.flatten(), stable=True).to(torch.int32)
#         # update the Ord attribute
#         self._param_Ord.data = Ord.to(self.device)


# =============================================================================
class DynamicLinear(Linear):
    r"""Simulates the measurement of a moving object :math:`y = H \cdot x(t)`.

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

        :attr:`white_acq` (torch.tensor, optional): Eventual spatial gain resulting from
        detector inhomogeneities. Must have the same shape as the measurement patterns.

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

    def __init__(
        self,
        H: torch.tensor,
        time_dim: int,
        meas_shape: Union[int, torch.Size, Iterable[int]] = None,
        meas_dims: Union[int, torch.Size, Iterable[int]] = None,
        *,
        noise_model: nn.Module = nn.Identity(),
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        white_acq: torch.tensor = None,
    ):
        super().__init__(
            H,
            meas_shape,
            meas_dims,
            noise_model=noise_model,
            dtype=dtype,
            device=device,
        )

        self.white_acq = white_acq
        self.time_dim = time_dim
        if self.time_dim in self.meas_dims:
            raise RuntimeError(
                f"The time dimension must not be in the measurement dimensions. Found {self.time_dim} in {self.meas_dims}."
            )

        # define the available matrices for reconstruction
        self._available_pinv_matrices = ["H_dyn"]
        self._selected_pinv_matrix = "H_dyn"  # select default here

    @property
    def recon_mode(self) -> str:
        """Interpolation mode used for reconstruction."""
        return self._recon_mode

    def build_H_dyn(
        self,
        motion: DeformationField,
        mode: str = "bilinear",
        warping: bool = False,
    ) -> None:
        """Build the dynamic measurement matrix `H_dyn`.

        Compute and store the dynamic measurement matrix `H_dyn` from the static
        measurement matrix `H_static` and the deformation field `motion`. The
        output is stored in the attribute `self.H_dyn`.

        This is done using the physical version explained in [MaBP24]_.

        Args:

            :attr:`motion` (DeformationField): Deformation field representing the
            motion of the image. Need to pass the inverse deformation field when
            :attr:`warping` is set to False, and the direct deformation field when
            :attr:`warping` is set to True.

            :attr:`mode` (str): Mode according to which the dynamic matrix is constructed. When warping the patterns,
            it refers to the interpolation method. When the patterns are not warped, it refers to the regularity of
            the solution that is sought after. Defaults to 'bilinear'.

            :attr:`warping` (bool): Whether to warp the patterns when building
            the dynamic measurement matrix. It's been shown [MaBP24] that warping
            the patterns induces a bias in the model. Defaults to 'False'.

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

        # store the method and mode in attribute
        self._recon_mode = mode
        self._recon_warping = warping

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

        if isinstance(self, DynamicLinearSplit):
            meas_pattern = self.P
        else:
            meas_pattern = self.H_static

        if self.white_acq is not None:
            meas_pattern *= self.white_acq.ravel().unsqueeze(
                0
            )  # for eventual spatial gain

        if not warping:
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
            ).reshape(n_frames, self.h * self.w)
            del def_field_00, mask

            # PART 3: WARP H MATRIX WITH FLATTENED INDICES
            # _________________________________________________________________
            # Build 4 submatrices with 4 weights for bilinear interpolation
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
                dtype=meas_dxy.dtype,
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

        else:
            det = motion.det()

            meas_pattern = meas_pattern.reshape(
                meas_pattern.shape[0], 1, self.meas_shape[0], self.meas_shape[1]
            )
            meas_pattern_ext = torch.zeros(
                (meas_pattern.shape[0], 1, self.img_shape[0], self.img_shape[1])
            )
            amp_max_h = (self.img_shape[0] - self.meas_shape[0]) // 2
            amp_max_w = (self.img_shape[1] - self.meas_shape[1]) // 2
            meas_pattern_ext[
                :,
                :,
                amp_max_h : self.meas_shape[0] + amp_max_h,
                amp_max_w : self.meas_shape[1] + amp_max_w,
            ] = meas_pattern
            meas_pattern_ext = meas_pattern_ext.to(dtype=motion.field.dtype)

            H_dyn = nn.functional.grid_sample(
                meas_pattern_ext,
                motion.field,
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            )
            H_dyn = det.reshape((meas_pattern.shape[0], -1)) * H_dyn.reshape(
                (meas_pattern.shape[0], -1)
            )

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

        # # =============================================================================
        # class DynamicLinearSplit(DynamicLinear):
        #     # =========================================================================
        #     r"""
        #     Simulates the measurement of a moving object using a splitted operator
        #     :math:`y = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix} \cdot x(t)`.

        #     Computes linear measurements :math:`y` from incoming images: :math:`y = Px`,
        #     where :math:`P` is a linear operator (matrix) and :math:`x` is a batch of
        #     vectorized images representing a motion picture.

        #     The matrix :math:`P` contains only positive values and is obtained by
        #     splitting a measurement matrix :math:`H` such that
        #     :math:`P` has a shape of :math:`(2M, N)` and `P[0::2, :] = H_{+}` and
        #     `P[1::2, :] = H_{-}`, where :math:`H_{+} = \max(0,H)` and
        #     :math:`H_{-} = \max(0,-H)`.

        #     The class is constructed from the :math:`M` by :math:`N` matrix :math:`H`,
        #     where :math:`N` represents the number of pixels in the image and
        #     :math:`M` the number of measurements. Therefore, the shape of :math:`P` is
        #     :math:`(2M, N)`.

        #     Args:
        #         :attr:`H` (torch.tensor): measurement matrix (linear operator) with
        #         shape :math:`(M, N)` where :math:`M` is the number of measurements and
        #         :math:`N` the number of pixels in the image. Only real values are supported.

        #         :attr:`Ord` (torch.tensor, optional): Order matrix used to reorder the
        #         rows of the measurement matrix :math:`H`. The first new row of :math:`H`
        #         will correspond to the highest value in :math:`Ord`. Must contain
        #         :math:`M` values. If some values repeat, the order is kept. Defaults to
        #         None.

        #         :attr:`meas_shape` (tuple, optional): Shape of the measurement patterns.
        #         Must be a tuple of two integers representing the height and width of the
        #         patterns. If not specified, the shape is suppposed to be a square image.
        #         If not, an error is raised. Defaults to None.

        #         :attr:`img_shape` (tuple, optional): Shape of the image. Must be a tuple
        #         of two integers representing the height and width of the image. If not
        #         specified, the shape is taken as equal to `meas_shape`. Setting this
        #         value is particularly useful when using an :ref:`extended field of view <_MICCAI24>`.

        # :attr:`white_acq` (torch.tensor, optional): Eventual spatial gain resulting from
        # detector inhomogeneities. Must have the same shape as the measurement patterns.

        # Attributes:
        #     :attr:`H_static` (torch.nn.Parameter): The learnable measurement matrix
        #     of shape :math:`(M,N)` initialized as :math:`H`.

        #         :attr:`P` (torch.nn.Parameter): The splitted measurement matrix of
        #         shape :math:`(2M, N)` such that `P[0::2, :] = H_{+}` and `P[1::2, :] = H_{-}`.

        #         :attr:`M` (int): Number of measurements performed by the linear operator.

        #         :attr:`N` (int): Number of pixels in the image.

        #         :attr:`h` (int): Measurement pattern height.

        #         :attr:`w` (int): Measurement pattern width.

        #         :attr:`meas_shape` (tuple): Shape of the measurement patterns
        #         (height, width). Is equal to `(self.h, self.w)`.

        #         :attr:`img_h` (int): Image height.

        #         :attr:`img_w` (int): Image width.

        #         :attr:`img_shape` (tuple): Shape of the image (height, width). Is equal
        #         to `(self.img_h, self.img_w)`.

        #         :attr:`H_dyn` (torch.tensor): Dynamic measurement matrix :math:`H`.
        #         Must be set using the method :meth:`build_H_dyn` before being accessed.

        #         :attr:`H` (torch.tensor): Alias for :attr:`H_dyn`.

        #         :attr:`H_dyn_pinv` (torch.tensor): Dynamic pseudo-inverse measurement
        #         matrix :math:`H_{dyn}^\dagger`. Must be set using the method
        #         :meth:`build_H_dyn_pinv` before being accessed.

        #         :attr:`H_pinv` (torch.tensor): Alias for :attr:`H_dyn_pinv`.

        #     .. warning::
        #         For each call, there must be **exactly** as many images in :math:`x` as
        #         there are measurements in the linear operator :math:`P`.

        #     Example:
        #         >>> H = torch.rand([400,1600])
        #         >>> meas_op = DynamicLinearSplit(H)
        #         >>> print(meas_op)
        #         DynamicLinearSplit(
        #           (M): 400
        #           (N): 1600
        #           (H.shape): torch.Size([400, 1600])
        #           (meas_shape): (40, 40)
        #           (H_dyn): False
        #           (img_shape): (40, 40)
        #           (H_pinv): False
        #           (P.shape): torch.Size([800, 1600])
        #         )

        #     Reference:
        #     .. _MICCAI24:
        #         [MaBP24] (MICCAI 2024 paper #883) Thomas Maitre, Elie Bretin, Romain Phan, Nicolas Ducros,
        #         Michaël Sdika. Dynamic Single-Pixel Imaging on an Extended Field of View
        #         without Warping the Patterns. 2024. hal-04533981
        #     """

        # def __init__(
        #     self,
        #     H: torch.tensor,
        #     Ord: torch.tensor = None,
        #     meas_shape: tuple = None,  # (height, width)
        #     img_shape: tuple = None,  # (height, width)
        #     white_acq: torch.tensor = None,
        # ):
        #     # call constructor of DynamicLinear
        #     super().__init__(H, Ord, meas_shape, img_shape)
        #     self._set_P(self.H_static)

        #     @property  # override _Base definition
        #     def operator(self) -> torch.tensor:
        #         return self.P

        #     def forward(self, x: torch.tensor) -> torch.tensor:
        #         r"""
        #         Simulates the measurement of a motion picture :math:`y = P \cdot x(t)`.

        #         The output :math:`y` is computed as :math:`y = Px`, where :math:`P` is
        #         the measurement matrix and :math:`x` is a batch of images.

        #         The matrix :math:`P` contains only positive values and is obtained by
        #         splitting a measurement matrix :math:`H` such that
        #         :math:`P` has a shape of :math:`(2M, N)` and `P[0::2, :] = H_{+}` and
        #         `P[1::2, :] = H_{-}`, where :math:`H_{+} = \max(0,H)` and
        #         :math:`H_{-} = \max(0,-H)`.

        #         If you want to measure with the original matrix :math:`H`, use the
        #         method :meth:`forward_H`.

        #         Args:
        #             :attr:`x`: Batch of images of shape :math:`(*, t, c, h, w)` where *
        #             denotes any dimension (e.g. the batch size), :math:`t` the number of
        #             frames, :math:`c` the number of channels, and :math:`h`, :math:`w`
        #             the height and width of the images.

        #         Output:
        #             :math:`y`: Linear measurements of the input images. It has shape
        #             :math:`(*, c, 2M)` where * denotes any number of dimensions, :math:`c`
        #             the number of channels, and :math:`M` the number of measurements.

        #         .. important::
        #             There must be as many images as there are measurements in the split
        #             linear operator, i.e. :math:`t = 2M`.

        #         Shape:
        #             :math:`x`: :math:`(*, t, c, h, w)`

        #             :math:`P` has a shape of :math:`(2M, N)` where :math:`M` is the
        #             number of measurements as defined by the first dimension of :math:`H`
        #             and :math:`N` is the number of pixels in the image.

        #             :math:`output`: :math:`(*, c, 2M)` or :math:`(*, c, t)`

        #         Example:
        #             >>> x = torch.rand([10, 800, 3, 40, 40])
        #             >>> H = torch.rand([400, 1600])
        #             >>> meas_op = DynamicLinearSplit(H)
        #             >>> y = meas_op(x)
        #             >>> print(y.shape)
        #             torch.Size([10, 3, 800])
        #         """
        #         return self._dynamic_forward_with_op(x, self.P)

        #     def forward_H(self, x: torch.tensor) -> torch.tensor:
        #         r"""
        #         Simulates the measurement of a motion picture :math:`y = H \cdot x(t)`.

        #         The output :math:`y` is computed as :math:`y = Hx`, where :math:`H` is
        #         the measurement matrix and :math:`x` is a batch of images.

        #         The matrix :math:`H` can contain positive and negative values and is
        #         given by the user at initialization. If you want to measure with the
        #         splitted matrix :math:`P`, use the method :meth:`forward`.

        #         Args:
        #             :attr:`x`: Batch of images of shape :math:`(*, t, c, h, w)` where *
        #             denotes any dimension (e.g. the batch size), :math:`t` the number of
        #             frames, :math:`c` the number of channels, and :math:`h`, :math:`w`
        #             the height and width of the images.

        #         Output:
        #             :math:`y`: Linear measurements of the input images. It has shape
        #             :math:`(*, c, M)` where * denotes any number of dimensions, :math:`c`
        #             the number of channels, and :math:`M` the number of measurements.

        #         .. important::
        #             There must be as many images as there are measurements in the original
        #             linear operator, i.e. :math:`t = M`.

        #         Shape:
        #             :math:`x`: :math:`(*, t, c, h, w)`

        #             :math:`H` has a shape of :math:`(M, N)` where :math:`M` is the
        #             number of measurements and :math:`N` is the number of pixels in the
        #             image.

        #             :math:`output`: :math:`(*, c, M)`

        #         Example:
        #             >>> x = torch.rand([10, 400, 3, 40, 40])
        #             >>> H = torch.rand([400, 1600])
        #             >>> meas_op = LinearDynamicSplit(H)
        #             >>> y = meas_op.forward_H(x)
        #             >>> print(y.shape)
        #             torch.Size([10, 3, 400])
        #         """
        #         return super().forward(x)

        #     def _set_Ord(self, Ord: torch.tensor) -> None:
        #         """Set the order matrix used to sort the rows of H."""
        #         super()._set_Ord(Ord)
        #         # update P
        #         self._set_P(self.H_static)

        # # =============================================================================
        # class DynamicHadamSplit(DynamicLinearSplit):
        #     # =========================================================================
        #     r"""
        #     Simulates the measurement of a moving object using a splitted operator
        #     :math:`y = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix} \cdot x(t)` with
        #     :math:`H` a Hadamard matrix.

        #     Computes linear measurements from incoming images: :math:`y = Px`,
        #     where :math:`P` is a linear operator (matrix) with positive entries and
        #     :math:`x` is a batch of vectorized images representing a motion picture.

        #     The matrix :math:`P` contains only positive values and is obtained by
        #     splitting a Hadamard-based matrix :math:`H` such that
        #     :math:`P` has a shape of :math:`(2M, N)` and `P[0::2, :] = H_{+}` and
        #     `P[1::2, :] = H_{-}`, where :math:`H_{+} = \max(0,H)` and
        #     :math:`H_{-} = \max(0,-H)`.

        #     :math:`H` is obtained by selecting a re-ordered subsample of :math:`M` rows
        #     of a "full" Hadamard matrix :math:`F` with shape :math:`(N^2, N^2)`.
        #     :math:`N` must be a power of 2.

        #     Args:
        #         :attr:`M` (int): Number of measurements. If :math:`M < h^2`, the
        #         measurement matrix :math:`H` is cropped to :math:`M` rows.

        #         :attr:`h` (int): Measurement pattern height, must be a power of 2. The
        #         image is assumed to be square, so the number of pixels in the image is
        #         :math:`N = h^2`.

        #         :attr:`Ord` (torch.tensor, optional): Order matrix used to reorder the
        #         rows of the measurement matrix :math:`H`. The first new row of :math:`H`
        #         will correspond to the highest value in :math:`Ord`. Must contain
        #         :math:`M` values. If some values repeat, the order is kept. Defaults to
        #         None.

        #         :attr:`img_shape` (tuple, optional): Shape of the image. Must be a tuple
        #         of two integers representing the height and width of the image. If not
        #         specified, the shape is taken as equal to `meas_shape`. Setting this
        #         value is particularly useful when using an :ref:`extended field of view <_MICCAI24>`.

        #         :attr:`white_acq` (torch.tensor, optional): Eventual spatial gain resulting from
        #         detector inhomogeneities. Must have the same shape as the measurement patterns.

        #     Attributes:
        #         :attr:`H_static` (torch.nn.Parameter): The learnable measurement matrix
        #         of shape :math:`(M,N)` initialized as :math:`H`.

        #         :attr:`P` (torch.nn.Parameter): The splitted measurement matrix of
        #         shape :math:`(2M, N)` such that `P[0::2, :] = H_{+}` and `P[1::2, :] = H_{-}`.

        #         :attr:`M` (int): Number of measurements performed by the linear operator.

        #         :attr:`N` (int): Number of pixels in the image.

        #         :attr:`h` (int): Measurement pattern height.

        #         :attr:`w` (int): Measurement pattern width.

        #         :attr:`meas_shape` (tuple): Shape of the measurement patterns
        #         (height, width). Is equal to `(self.h, self.w)`.

        #         :attr:`img_h` (int): Image height.

        #         :attr:`img_w` (int): Image width.

        #         :attr:`img_shape` (tuple): Shape of the image (height, width). Is equal
        #         to `(self.img_h, self.img_w)`.

        #         :attr:`H_dyn` (torch.tensor): Dynamic measurement matrix :math:`H`.
        #         Must be set using the method :meth:`build_H_dyn` before being accessed.

        #         :attr:`H` (torch.tensor): Alias for :attr:`H_dyn`.

        #         :attr:`H_dyn_pinv` (torch.tensor): Dynamic pseudo-inverse measurement
        #         matrix :math:`H_{dyn}^\dagger`. Must be set using the method
        #         :meth:`build_H_dyn_pinv` before being accessed.

        #         :attr:`H_pinv` (torch.tensor): Alias for :attr:`H_dyn_pinv`.

        #     .. note::
        #         The computation of a Hadamard transform :math:`Fx` benefits a fast
        #         algorithm, as well as the computation of inverse Hadamard transforms.

        #     .. note::
        #         :math:`H = H_{+} - H_{-}`

        #     Example:
        #         >>> Ord = torch.rand([32,32])
        #         >>> meas_op = HadamSplitDynamic(400, 32, Ord)
        #         >>> print(meas_op)
        #         DynamicHadamSplit(
        #           (M): 400
        #           (N): 1024
        #           (H.shape): torch.Size([400, 1024])
        #           (meas_shape): (32, 32)
        #           (H_dyn): False
        #           (img_shape): (32, 32)
        #           (H_pinv): False
        #           (P.shape): torch.Size([800, 1024])
        #         )

        #     Reference:
        #     .. _MICCAI24:
        #         [MaBP24] (MICCAI 2024 paper #883) Thomas Maitre, Elie Bretin, Romain Phan, Nicolas Ducros,
        #         Michaël Sdika. Dynamic Single-Pixel Imaging on an Extended Field of View
        #         without Warping the Patterns. 2024. hal-04533981
        #     """

        # def __init__(
        #     self,
        #     M: int,
        #     h: int,
        #     Ord: torch.tensor = None,
        #     img_shape: tuple = None,  # (height, width)
        #     white_acq: torch.tensor = None,
        # ):

        #         F = spytorch.walsh_matrix_2d(h)
        #         # empty = torch.empty(h**2, h**2)  # just to get the shape

        # we pass the whole F matrix to the constructor
        super().__init__(F, Ord, (h, h), img_shape)
        self._M = M
