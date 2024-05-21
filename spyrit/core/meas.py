"""
Measurement operators, static and dynamic.

There are six classes contained in this module, each representing a different
type of measurement operator. Three of them are static, i.e. they are used to
simulate measurements of still images, and three are dynamic, i.e. they are used
to simulate measurements of moving objects, represented as a sequence of images.

There is an additional classe :class:`_Base`, that is used for internal
purposes and should not be used directly. The inheritance tree is as follows::

                _Base
                  |
        +---------+---------+
        |                   |
        V                   V
      Linear          DynamicLinear
        |                   |
        V                   V
    LinearSplit     DynamicLinearSplit
        |                   |
        V                   V
    HadamSplit      DynamicHadamSplit

"""

import warnings
from typing import Union

import math
import torch
import torch.nn as nn

from spyrit.core.time import DeformationField
import spyrit.core.torch as spytorch


# =============================================================================
# BASE CLASS - FOR INHERITANCE ONLY (INTERAL USE)
# =============================================================================
class _Base(nn.Module):
    
    def __init__(self,
                 H_static: torch.tensor,
                 Ord: torch.tensor=None,
                 meas_shape: tuple=None,
                 ) -> None:
        super().__init__()
    
        # H should be stored in float32
        H_static = H_static.to(torch.float32)
        # store meas_shape and check it is correct
        if meas_shape is None:
            self._meas_shape = (int(math.sqrt(H_static.shape[1])),
                               int(math.sqrt(H_static.shape[1])))
        else:
            self._meas_shape = meas_shape
        if self._meas_shape[0] * self._meas_shape[1] != H_static.shape[1]:
            raise ValueError(
                f"The number of pixels in the measurement matrix H " +
                f"({H_static.shape[1]}) does not match the measurement shape " +
                f"{self._meas_shape}."
            )
        
        if Ord is not None:
            H_static, ind = spytorch.sort_by_significance(
                H_static, Ord, "rows", False, get_indices=True
            )
        else:
            ind = torch.arange(H_static.shape[0])
            Ord = torch.arange(H_static.shape[0], 0, -1)
        
        # attributes for internal use
        self._param_H_static = nn.Parameter(
            H_static.to(torch.float32), requires_grad=False
        )
        self._param_Ord = nn.Parameter(
            Ord.to(torch.float32), requires_grad=False
        )
        self._indices = ind.to(torch.int32)
        # need to store M because H_static may be cropped (see HadamSplit)
        self._M = H_static.shape[0]
        
    ### PROPERTIES ------
    @property
    def M(self) -> int:
        """Number of measurements (first dimension of H)"""
        return self._M
    @property
    def N(self) -> int:
        """Number of pixels in the image (second dimension of H)"""
        return self.H_static.shape[1]
    @property
    def h(self) -> int:
        """Image height"""
        return self.meas_shape[0]
    @property
    def w(self) -> int:
        """Image width"""
        return self.meas_shape[1]
    @property
    def meas_shape(self) -> tuple:
        """Shape of the measurement patterns (height, width). Note that
        `height * width = N`."""
        return self._meas_shape
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
        return self._param_H_static.data[:self.M, :]
    @property
    def P(self) -> torch.tensor:
        """Measurement matrix P with positive and negative components. Used in
        classes *Split and *HadamSplit."""
        return self._param_P.data[:2*self.M, :]
    ### -------------------
    
    def pinv(self, 
             x: torch.tensor, 
             reg: str=None, 
             eta: float=None
             ) -> torch.tensor:
        r"""Computes the pseudo inverse solution :math:`y = H^\dagger x`

        Args:
            :math:`x` (torch.tensor): batch of measurement vectors. If x has
            more than 1 dimension, the pseudo inverse is applied to each
            image in the batch.

        Shape:
            :math:`x`: :math:`(*, M)`

            Output: :math:`(*, N)`

        Example:
            >>> H = torch.randn([400, 1600])
            >>> meas_op = Linear(H, True)
            >>> x = torch.randn([10, 400])
            >>> y = meas_op.pinv(x)
            >>> print(y.shape)
            torch.Size([10, 1600])
        """
        # equivalent to
        # torch.linalg.solve(H_dyn.T @ H_dyn + reg, H_dyn.T @ x)
        if hasattr(self, 'H_pinv'):
            # if the pseudo inverse has been computed
            return x @ self.H_pinv.T.to(x.dtype)
        elif isinstance(self, Linear):
            # can we compute the inverse of H ?
            H_to_inv = self.H_static
        elif isinstance(self, DynamicLinear):
            H_to_inv = self.H_dyn
        else:
            raise NotImplementedError(
                "It seems you have instanciated a _Base element. This class " +
                "Should not be called on its own."
            )
        
        if reg == "L1":
            return torch.linalg.lstsq(H_to_inv, x.to(H_to_inv.dtype).T, 
                            rcond=eta, driver="gelsd").solution.to(x.dtype)
        elif reg == "L2":
            # if under- over-determined problem ?
            return torch.linalg.solve(
                H_to_inv.T @ H_to_inv 
                + eta*torch.eye(H_to_inv.shape[1]), 
                H_to_inv.T @ x.to(H_to_inv.dtype).T
            ).to(x.dtype).T
        
        elif reg is None:
            raise ValueError(
                "Regularization method not specified. Please compute " +
                "the dynamic pseudo-inverse or specify a regularization " +
                "method."
            )
        else:
            raise NotImplementedError(
                f"Regularization method ({reg}) not implemented. Please " +
                "use 'L1' or 'L2'."
            )
                
            
    def reindex(self, 
                x: torch.tensor,
                axis: str="rows",
                inverse_permutation: bool=False
                ) -> torch.tensor:
        """Reorder the rows or columns of a tensor according to the indices
        stored in the attribute self.indices. The value stored in
        `self.indices[0]` is the new index of the first row or column of the
        input tensor.
        
        Args:
            x (torch.tensor): Input tensor to be reordered.
            
            axis (str, optional): Axis along which to order the tensor. Must be
            either "rows" or "cols". Defaults to "rows".
            
            inverse_permutation (bool, optional): If True, the inverse 
            permutation is used. Defaults to False.
        
        Returns:
            torch.tensor: Tensor x with reordered rows or columns according to
            the indices.
        
        .. note::
            This method is identical to the function
            :func:`~spyrit.core.torch.reindex`.
        """
        return spytorch.reindex(
            x.to(self.indices.device),
            self.indices,
            axis,
            inverse_permutation
        )

    def _set_Ord(self, Ord: torch.tensor) -> None:
        """Set the order matrix used to sort the rows of H."""
        # unsort the rows of H
        H_natural = self.reindex(
            self.H_static, "rows", inverse_permutation=True
        )
        # resort the rows of H ; store indices in self._indices
        H_resorted, self._indices = spytorch.sort_by_significance(
            H_natural, Ord, "rows", False, get_indices=True
        )
        # update values of H, Ord
        self._param_H_static.data = H_resorted
        self._param_Ord.data = Ord

    def _set_P(self, H_static: torch.tensor) -> None:
        """Set the positive and negative components of the measurement matrix
        P from the static measurement matrix H_static. For internal use only.
        Used in classes *Split and *HadamSplit."""
        H_pos = nn.functional.relu(H_static)
        H_neg = nn.functional.relu(-H_static)
        self._param_P = nn.Parameter(
            torch.cat([H_pos, H_neg], 1).view(2 * H_static.shape[0],
                                                H_static.shape[1]),
            requires_grad=False
        )

    def _attributeslist(self) -> list:
        # change this list if you upate any attribute in any of the classes 
        # must use strings for attributes because the attributes may not exist
        _list = [
            ("M", "self.M", _Base),
            ("N", "self.N", _Base),
            ("H.shape", "self.H_static.shape", _Base),
            ("meas_shape", "self._meas_shape", _Base),
            ("H_dyn", "hasattr(self, 'H_dyn')", DynamicLinear),
            ("img_shape", "self.img_shape", DynamicLinear),
            ("H_pinv", "hasattr(self, 'H_pinv')", _Base),
            ("P.shape", "self.P.shape", Union[LinearSplit, DynamicLinearSplit]),
        ]
        return _list
        
    def __repr__(self) -> str:
        s_begin = f"{self.__class__.__name__}(\n  "
        s_fill = "\n  ".join(
            [f"({k}): {eval(v)}" 
             for k,v,t in self._attributeslist() if isinstance(self, t)]
        )
        s_end = "\n  )"
        return s_begin + s_fill + s_end


# =============================================================================
class Linear(_Base):
    # =========================================================================
    r"""
    Simulates the measurement of an still image using a measurement matrix.

    Computes linear measurements from incoming images: :math:`y = Hx`,
    where :math:`H` is a given linear operator (matrix) and :math:`x` is a
    vectorized image or batch of images.

    The class is constructed from a :math:`M` by :math:`N` matrix :math:`H`,
    where :math:`N` represents the number of pixels in the image and
    :math:`M` the number of measurements.

    Args:
        :attr:`H` (:class:`torch.tensor`): measurement matrix (linear operator)
        with shape :math:`(M, N)`.

        :attr:`pinv` (bool): Option to have access to pseudo inverse solutions. If
        `True`, the pseudo inverse is initialized as :math:`H^\dagger` and
        stored in the attribute :attr:`H_pinv`. Defaults to `False` (the pseudo
        inverse is not initiliazed).

        :attr:`rtol` (float, optional): Cutoff for small singular values (see
        :mod:`torch.linalg.pinv`). Only relevant when :attr:`pinv` is `True`.

    Attributes:
        :attr:`H` (torch.tensor): The learnable measurement matrix of shape
        :math:`(M, N)` initialized as :math:`H`

        :attr:`H_pinv` (torch.tensor, optional): The learnable adjoint measurement
        matrix of shape :math:`(N, M)` initialized as :math:`H^\dagger`.
        Only relevant when :attr:`pinv` is `True`.

        :attr:`M` (int): Number of measurements performed by the linear operator.
        It is initialized as the first dimension of :math:`H`.

        :attr:`N` (int): Number of pixels in the image. It is initialized as the
        second dimension of :math:`H`.

        :attr:`h` (int): Image height :math:`h`. The image is assumed to be
        square, i.e. :math:`h = \text{floor}(\sqrt{N})`. If not, please assign
        :attr:`h` and :attr:`w` manually.

        :attr:`w` (int): Image width :math:`w`. The image is assumed to be
        square, i.e. :math:`w = \text{floor}(\sqrt{N})`. If not, please assign
        :attr:`h` and :attr:`w` manually.

    .. note::
        If you know the pseudo inverse of :math:`H` and want to store it, it is
        best to initialize the class with :attr:`pinv` set to `False` and then
        call :meth:`set_H_pinv` to store the pseudo inverse.

    Example 1:
        >>> H = torch.rand([400, 1600])
        >>> meas_op = Linear(H, pinv=False)
        >>> print(meas_op)
        Linear(
          (Image pixels): 1600
          (H): torch.Size([400, 1600])
          (H_pinv): None
          )

    Example 2:
        >>> H = torch.rand([400, 1600])
        >>> meas_op = Linear(H, True)
        >>> print(meas_op)
        Linear(
          (Image pixels): 1600
          (H): torch.Size([400, 1600])
          (H_pinv): torch.Size([1600, 400])
          )
    """

    def __init__(self,
                 H: torch.tensor,
                 pinv: bool=False,
                 rtol: float=None,
                 Ord: torch.tensor=None,
                 meas_shape: tuple=None, # (height, width)
                 ):
        super().__init__(H, Ord, meas_shape)
        if pinv:
            self.set_H_pinv(rtol=rtol)

    @property
    def H_static(self) -> torch.tensor:
        # For name consistency across dynamic & static versions
        return self._param_H_static.data
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

    def set_H_pinv(self, rtol: float=None) -> None:
        """Used to set the pseudo inverse of the measurement matrix :math:`H`
        using `torch.linalg.pinv`.

        Args:
            rtol (float, optional): Regularization parameter (cutoff for small
            singular values, see :mod:`torch.linalg.pinv`). Defaults to None,
            in which case the default value of :mod:`torch.linalg.pinv` is used.
        
        Returns:
            None. The pseudo inverse is stored in the attribute :attr:`H_pinv`.
        """
        self.H_pinv = torch.linalg.pinv(self.H.to(torch.float64), rtol=rtol)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Applies linear transform to incoming images: :math:`y = Hx`.

        Args:
            :math:`x` (torch.tensor): Batch of vectorized (flattened) images.
            If x has more than 1 dimension, the linear measurement is applied
            to each image in the batch.

        Shape:
            :math:`x`: :math:`(*, N)` where * denotes the batch size and `N`
            the total number of pixels in the image.

            Output: :math:`(*, M)` where * denotes the batch size and `M`
            the number of measurements.

        Example:
            >>> H = torch.randn([400, 1600])
            >>> meas_op = Linear(H)
            >>> x = torch.randn([10, 1600])
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([10, 400])
        """
        # left multiplication with transpose is equivalent to right mult
        return x @ self.H.T.to(x.dtype)

    def adjoint(self, x: torch.tensor) -> torch.tensor:
        r"""Applies adjoint transform to incoming measurements :math:`y = H^{T}x`

        Args:
            :math:`x` (torch.tensor): batch of measurement vectors. If x has
            more than 1 dimension, the adjoint measurement is applied to each
            measurement in the batch.

        Shape:
            :math:`x`: :math:`(*, M)`

            Output: :math:`(*, N)`

        Example:
            >>> H = torch.randn([400, 1600])
            >>> meas_op = Linear(H)
            >>> x = torch.randn([10, 400]
            >>> y = meas_op.adjoint(x)
            >>> print(y.shape)
            torch.Size([10, 1600])
        """
        # left multiplication is equivalent to right mult with transpose
        return x @ self.H.to(x.dtype)

    def _set_Ord(self, Ord: torch.tensor) -> None:
        """Set the order matrix used to sort the rows of H."""
        super()._set_Ord(Ord)
        # delete self.H_pinv (self._param_H_static_pinv)
        try:
            del self._param_H_static_pinv
            warnings.warn(
                "The pseudo-inverse H_pinv has been deleted. Please call " +
                "set_H_pinv() to recompute it."
            )
        except AttributeError:
            pass

    
# =============================================================================
class LinearSplit(Linear):
    # =========================================================================
    r"""
    Simulates the measurement of a still image using the computed positive and
    negative components of the measurement matrix.

    Computes linear measurements from incoming images: :math:`y = Px`,
    where :math:`P` is a linear operator (matrix) and :math:`x` is a
    vectorized image or batch of vectorized images.

    The matrix :math:`P` contains only positive values and is obtained by
    splitting a measurement matrix :math:`H` such that
    :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where
    :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

    The class is constructed from the :math:`M` by :math:`N` matrix :math:`H`,
    where :math:`N` represents the number of pixels in the image and
    :math:`M` the number of measurements.

    Args:
        :attr:`H` (torch.tensor): measurement matrix (linear operator) with
        shape :math:`(M, N)`, where :math:`M` is the number of measurements and
        :math:`N` the number of pixels in the image.

        :attr:`pinv` (Any): Option to have access to pseudo inverse solutions. If
        `True`, the pseudo inverse is initialized as :math:`H^\dagger` and
        stored in the attribute :attr:`H_pinv`. Defaults to `False` (the pseudo
        inverse is not initiliazed).

        :attr:`rtol` (float, optional): Regularization parameter (cutoff for small
        singular values, see :mod:`torch.linalg.pinv`). Only relevant when
        :attr:`pinv` is `True`.

    Attributes:
        :attr:`H` (torch.nn.Parameter): The learnable measurement matrix of
        shape :math:`(M,N)`.

        :attr:`P` (torch.nn.Parameter): The splitted measurement matrix of
        shape :math:`(2M, N)` initialized as
        :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`
        where :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`

        :attr:`M` (int): Number of measurements performed by the linear operator.
        It is initialized as the first dimension of :math:`H`.

        :attr:`N` (int): Number of pixels in the image. It is initialized as the
        second dimension of :math:`H`.

        :attr:`h` (int): Image height :math:`h`. The image is assumed to be
        square, i.e. :math:`h = \text{floor}(\sqrt{N})`. If not, please assign
        :attr:`h` and :attr:`w` manually.

        :attr:`w` (int): Image width :math:`w`. The image is assumed to be
        square, i.e. :math:`w = \text{floor}(\sqrt{N})`. If not, please assign
        :attr:`h` and :attr:`w` manually.

    .. note::
        If you know the pseudo inverse of :math:`H` and want to store it, it is
        best to initialize the class with :attr:`pinv` set to `False` and then
        call :meth:`set_H_pinv` to store the pseudo inverse.

    Example:
        >>> H = torch.randn(400, 1600)
        >>> meas_op = LinearSplit(H, False)
        >>> print(meas_op)
        LinearSplit(
          (Image pixels): 1600
          (H): torch.Size([400, 1600])
          (P): torch.Size([800, 1600])
          (H_pinv): None
          )
    """

    def __init__(self,
                 H: torch.tensor,
                 pinv: bool=False,
                 rtol: float=None,
                 Ord: torch.tensor=None,
                 meas_shape: tuple=None, # (height, width)
                 ):
        super().__init__(H, pinv, rtol, Ord, meas_shape)
        self._set_P(self.H_static)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Applies linear transform to incoming images: :math:`y = Px`.

        This method uses the splitted measurement matrix :math:`P` to compute
        the linear measurements from incoming images. :math:`P` contains only
        positive values and is obtained by splitting a given measurement matrix
        :math:`H` such that :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`,
        where :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

        Args:
            :math:`x` (torch.tensor): Batch of vectorized (flattened) images. If
            x has more than 1 dimension, the linear measurement is applied to
            each image in the batch.

        Shape:
            :math:`x`: :math:`(*, N)` where * denotes the batch size and `N`
            the total number of pixels in the image.

            Output: :math:`(*, 2M)` where * denotes the batch size and `M`
            the number of measurements.

        Example:
            >>> H = torch.randn(400, 1600)
            >>> meas_op = LinearSplit(H)
            >>> x = torch.randn(10, 1600)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([10, 800])
        """
        return x @ self.P.T.to(x.dtype)

    def forward_H(self, x: torch.tensor) -> torch.tensor:
        r"""Applies linear transform to incoming images: :math:`m = Hx`.

        This method uses the measurement matrix :math:`H` to compute the linear
        measurements from incoming images.

        Args:
            :attr:`x` (torch.tensor): Batch of vectorized (flatten) images. If
            x has more than 1 dimension, the linear measurement is applied to
            each image in the batch.

        Shape:
            :attr:`x`: :math:`(*, N)` where * denotes the batch size and `N`
            the total number of pixels in the image.

            Output: :math:`(*, M)` where * denotes the batch size and `M`
            the number of measurements.

        Example:
            >>> H = torch.randn(400, 1600)
            >>> meas_op = LinearSplit(H)
            >>> x = torch.randn(10, 1600)
            >>> y = meas_op.forward_H(x)
            >>> print(y.shape)
            torch.Size([10, 400])
        """
        # call Linear.forward() method
        return super().forward(x)


# =============================================================================
class HadamSplit(LinearSplit):
    # =========================================================================
    r"""
    Simulates the measurement of a still image using the positive and
    negative components of a Hadamard matrix.

    Computes linear measurements from incoming images: :math:`y = Px`,
    where :math:`P` is a linear operator (matrix) with positive entries and
    :math:`x` is a vectorized image or a batch of images.

    The class relies on a Hadamard-based matrix :math:`H` with shape :math:`(M,N)`
    where :math:`N` represents the number of pixels in the image and
    :math:`M \le N` the number of measurements. :math:`H` is obtained by
    selecting a re-ordered subsample of :math:`M` rows of a "full" Hadamard
    matrix :math:`F` with shape :math:`(N^2, N^2)`. :math:`N` must be a power
    of 2.

    The matrix :math:`P` is then obtained by splitting the matrix :math:`H`
    such that :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where
    :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

    Args:
        :attr:`M` (int): Number of measurements

        :attr:`h` (int): Image height :math:`h`, must be a power of 2. The
        image is assumed to be square, so the number of pixels in the image is
        :math:`N = h^2`.

        :attr:`Ord` (torch.tensor): Order matrix with shape :math:`(h, h)` used to
        compute the permutation matrix :math:`G^{T}` with shape :math:`(N, N)`
        (see the :mod:`~spyrit.misc.sampling` submodule)

    Attributes:
        :attr:`H` (torch.nn.Parameter): The measurement matrix of shape
        :math:`(M, h^2)`. It is initialized as a re-ordered subsample of the
        rows of the "full" Hadamard matrix :math:`F` with shape :math:`(N^2, N^2)`.

        :attr:`H_pinv` (torch.nn.Parameter): The pseudo inverse of the measurement
        matrix of shape :math:`(h^2, M)`. It is initialized as
        :math:`H^\dagger = \frac{1}{N}H^{T}` where :math:`N = h^2`.

        :attr:`P` (torch.nn.Parameter): The splitted measurement matrix of
        shape :math:`(2M, h^2)` initialized as
        :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`
        where :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

        :attr:`Perm` (torch.nn.Parameter): The permutation matrix :math:`G^{T}`
        that is used to re-order the subsample of rows of the "full" Hadamard
        matrix :math:`F` according to descreasing value of the order matrix
        :math:`Ord`. It has shape :math:`(N, N)` where :math:`N = h^2`.

        :attr:`M` (int): Number of measurements performed by the linear operator.

        :attr:`N` (int): Number of pixels in the image. It is initialized as
        :math:`h^2`.

        :attr:`h` (int): Image height :math:`h`.

        :attr:`w` (int): Image width :math:`w`. The image is assumed to be
        square, i.e. :math:`w = h`.

    .. note::
        The computation of a Hadamard transform :math:`Fx` benefits a fast
        algorithm, as well as the computation of inverse Hadamard transforms.

    .. note::
        The matrix H has shape :math:`(M,N)` with :math:`N = h^2`.

    .. note::
        :math:`H = H_{+} - H_{-}`

    Example:
        >>> h = 32
        >>> Ord = torch.randn(h, h)
        >>> meas_op = HadamSplit(400, h, Ord)
        >>> print(meas_op)
        HadamSplit(
          (Image pixels): 1024
          (H): torch.Size([400, 1024])
          (P): torch.Size([800, 1024])
          (Perm): torch.Size([1024, 1024])
          (H_pinv): torch.Size([1024, 400])
          )
    """

    def __init__(self,
                 M: int,
                 h: int,
                 Ord: torch.tensor=None,
                 ):
        
        F = spytorch.walsh2_matrix(h)
        # we pass the whole F matrix to the constructor, but override the 
        # calls self.H etc to only return the first M rows
        super().__init__(F, pinv=False, Ord=Ord, meas_shape=(h,h))
        self._M = M
        # set H_pinv as it is the transpose of H / self.N
        self.H_pinv = self.H.T / self.N
        
    def inverse(self, x: torch.tensor) -> torch.tensor:
        r"""Inverse transform of Hadamard-domain images
        :math:`x = H_{had}^{-1}G y` is a Hadamard matrix.

        Args:
            :math:`x`:  batch of images in the Hadamard domain

        Shape:
            :math:`x`: :math:`(b*c, N)` with :math:`b` the batch size,
            :math:`c` the number of channels, and :math:`N` the number of
            pixels in the image.

            Output: math:`(b*c, N)`

        .. note::
            This method only works if the Hadamard matrix used to initialize
            the class is square, i.e. :attr:`M` = :attr:`h^2`.

        Example:
            >>> h = 32
            >>> Ord = torch.randn(h, h)
            >>> meas_op = HadamSplit(400, h, Ord)
            >>> y = torch.randn(10, h**2)
            >>> x = meas_op.inverse(y)
            >>> print(x.shape)
            torch.Size([10, 1024])
        """
        # permutations
        # todo: check walsh2_S_fold_torch to speed up
        b, N = x.shape

        x = self.reindex(x, "cols", False)  # new way
        # x = x @ self.Perm.T               # old way

        x = x.view(b, 1, self.h, self.w)
        # inverse of full transform
        # todo: initialize with 1D transform to speed up
        x = 1 / self.N * spytorch.walsh2_torch(x)
        return x.view(b, N)
    
    def _set_Ord(self, Ord: torch.tensor) -> None:
        """Set the order matrix used to sort the rows of H."""
        super()._set_Ord(Ord)
        # update P
        self._set_P(self.H_static)


# =============================================================================
class DynamicLinear(_Base):
    # =========================================================================
    r"""
    Simulates the measurement of a moving object using a measurement matrix.

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

    Attributes:
        :attr:`H` (torch.nn.Parameter): The learnable measurement matrix of
        shape :math:`(M,N)` initialized as :math:`H`.

        :attr:`M` (int): Number of measurements performed by the linear operator.
        It is initialized as the first dimension of :math:`H`.

        :attr:`N` (int): Number of pixels in the image. It is initialized as the
        second dimension of :math:`H`.

        :attr:`h` (int): Image height :math:`h`. The image is assumed to be
        square, i.e. :math:`h = \text{floor}(\sqrt{N})`. If not, please assign
        :attr:`h` and :attr:`w` manually.

        :attr:`w` (int): Image width :math:`w`. The image is assumed to be
        square, i.e. :math:`w = \text{floor}(\sqrt{N})`. If not, please assign
        :attr:`h` and :attr:`w` manually.

    Example:
        >>> H = torch.rand([400, 1600])
        >>> meas_op = DynamicLinear(H)
        >>> print(meas_op)
        DynamicLinear(
          (Image pixels): 1600
          (H): torch.Size([400, 1600])
          )
    """

    def __init__(self, 
                 H: torch.tensor, 
                 Ord: torch.tensor=None,
                 meas_shape: tuple=None,  # (height, width)
                 img_shape: tuple=None,  # (height, width)
                 ):
        super().__init__(H, Ord, meas_shape)

        if img_shape is not None:
            self._img_shape = img_shape
            if img_shape[0] < self.meas_shape[0] or img_shape[1] < self.meas_shape[1]:
                raise ValueError(
                    "The image shape must be at least as large as the measurement " +
                    f"shape. Got image shape {img_shape} and measurement shape " +
                    f"{self.meas_shape}."
                )
        else:
            self._img_shape = self.meas_shape
    
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
                "The dynamic measurement matrix H has not been set yet. " +
                "Please call build_H_dyn() before accessing the attribute " +
                "H_dyn (or H)."
            ) from e
    @property
    def H_pinv(self) -> torch.tensor:
        """Dynamic pseudo-inverse H_pinv. Equal to self.H_dyn_pinv."""
        return self.H_dyn_pinv
    @property
    def H_dyn_pinv(self) -> torch.tensor:
        """Dynamic pseudo-inverse H_pinv."""
        try:
            return self._param_H_dyn_pinv.data
        except AttributeError as e:
            raise AttributeError(
                "The dynamic pseudo-inverse H_pinv has not been set yet. " +
                "Please call build_H_dyn_pinv() before accessing the attribute "+
                "H_dyn_pinv (or H_pinv)."
            ) from e
    @H_dyn_pinv.deleter
    def H_dyn_pinv(self) -> None:
        del self._param_H_dyn_pinv
    
    def build_H_dyn(self, motion:DeformationField, mode: str='bilinear') -> None:
        """Compute and store the dynamic measurement matrix `H` from the static
        measurement matrix `H_static` and the deformation field `motion`. The 
        output is stored in the attribute `self.H`.
        
        Args:
            motion (DeformationField): Deformation field representing the 
            motion of the image.
        """
        
        try:
            del self._param_H_dyn
            del self._param_H_dyn_pinv
            warnings.warn(
                "The dynamic measurement matrix pseudo-inverse H_pinv has " +
                "been deleted. Please call build_H_dyn_pinv() to recompute it."
            , UserWarning)
        except AttributeError:
            pass
        
        if hasattr(self, "_param_P"):
            meas_pattern = self.P
        else:
            meas_pattern = self.H_static
        # get H_static, pad it to make it the size of the image
        H_padded = spytorch.center_pad(
            meas_pattern.reshape(-1, *self._meas_shape),
            self.img_shape
        )
        
        # get deformation field from motion
        # scale from [-1;1] x [-1;1] to [0;width-1] x [0;height-1]
        scale_factor = torch.tensor(self.img_shape) - 1
        def_field = (motion.field + 1) / 2 * scale_factor
        
        if mode == 'bilinear':
            # get the integer part of the field for the 4 nearest neighbours
            #   00    point      01
            #    +------+--------+
            #    |      |        |
            #    |      |        |
            #    +------+--------+ point
            #    |      |        |
            #    +------+--------+
            #   10               11
            def_field_00 = def_field.floor().to(torch.int16)
            def_field_01 = def_field_00 + torch.tensor([0, 1]).to(torch.int16)
            def_field_10 = def_field_00 + torch.tensor([1, 0]).to(torch.int16)
            def_field_11 = def_field_00 + torch.tensor([1, 1]).to(torch.int16)
            # stack them all up for vectorized operations
            def_field_stacked = torch.stack(
                [def_field_00, def_field_01, def_field_10, def_field_11]
            )
        
            # compute the weights for the bilinear interpolation
            dx, dy = torch.split((def_field - def_field_00), [1, 1], dim=-1)
            dx, dy = dx.squeeze(-1), dy.squeeze(-1)
            # stack for the 4 nearest neighbours, must match def_field order
            dxy = torch.stack([
                (1 - dx) * (1 - dy),
                (1 - dx) * dy,
                dx * (1 - dy),
                dx * dy
            ])
            
            # combine with H_padded
            H_dxy = H_padded.to(torch.float64) * dxy
            
            # label each frame in the deformation field
            frames_index = torch.arange(
                meas_pattern.shape[0]
            ).view(1, meas_pattern.shape[0], 1, 1, 1)
            frames_index = frames_index.expand(
                4, -1, self.img_w, self.img_h, -1)
            # stack the frames with the def_field_stacked
            def_field_stacked_framed = torch.cat(
                (frames_index, def_field_stacked), dim=-1
            )
            
            # keep indices that are within the image AND for which 
            # the weights are non-zero
            maxs = torch.tensor([self.img_h, self.img_w])
            keep = ((def_field_stacked >= 0).all(dim=-1) 
                    & (def_field_stacked < maxs).all(dim=-1)
                    & (H_dxy != 0))
            
            # get the values and indices for the non-zero weights to add
            # to the dynamic measurement matrix. Linearize the indices
            vals = H_dxy[keep]
            indices = def_field_stacked_framed[keep].T
            
            H_shape_multiplier = torch.tensor([
                [H_padded.shape[1] * H_padded.shape[2]], 
                [1],                # accounts for x dimension
                [H_padded.shape[2]] # accounts for y dimension
            ])
            indices_linearized = (indices * H_shape_multiplier).sum(dim=0)
            
            # add in dynamic measurement matrix
            H_dyn = torch.zeros(H_padded.numel()).to(vals.dtype)
            H_dyn = H_dyn.put_(indices_linearized, vals, accumulate=True)
            H_dyn = H_dyn.view(meas_pattern.shape[0], self.img_w*self.img_h)
            # what is the dtype?
            
            self._param_H_dyn = nn.Parameter(H_dyn, requires_grad=False)
            
        elif mode == 'bicubic':
            raise NotImplementedError(
                "Bicubic interpolation is not yet implemented. It will be " +
                "available in a future release."
            )
        
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Please use either 'bilinear' or " +
                "'bicubic'."
            )
         
    def build_H_dyn_pinv(self, reg: str="L1", eta: float=None) -> None:
        """Computes the pseudo-inverse of the dynamic measurement matrix
        `H_dyn` and stores it in the attribute `H_dyn_pinv`. 
        
        This method supposes that the dynamic measurement matrix `H_dyn` has
        already been set using the method `build_H_dyn()`.

        Raises:
            AttributeError: If the dynamic measurement matrix `H_dyn` has not
            been set yet.
        """
        # later do with regularization parameter
        try:
            H_dyn = self.H_dyn.to(torch.float64)
        except AttributeError as e:
            raise AttributeError(
                "The dynamic measurement matrix H has not been set yet. " +
                "Please call build_H_dyn() before computing the pseudo-inverse."
            ) from e
        
        if reg == "L1":
            pinv = torch.linalg.pinv(H_dyn, atol=eta)
            
        elif reg == "L2":
            if H_dyn.shape[0] >= H_dyn.shape[1]:
                pinv = torch.linalg.inv(
                    H_dyn.T @ H_dyn + eta * torch.eye(H_dyn.shape[1])
                ) @ H_dyn.T
            else:
                pinv = H_dyn.T @ torch.linalg.inv(
                    H_dyn @ H_dyn.T + eta * torch.eye(H_dyn.shape[0])
                )
                
        elif reg =="H1":
            raise NotImplementedError(
                "H1 regularization has not yet been implemented. It will be " +
                "available in a future release."
            )
            # is the problem over- or under-determined?
            if H_dyn.shape[0] >= H_dyn.shape[1]:
                # Boundary condition matrices
                Dx, Dy = spytorch.finite_diff_mat(H_dyn.shape[1], 
                                                  boundary='neumann')
                D2 = Dx.T @ Dx + Dy.T @ Dy
                pinv = torch.linalg.inv(H_dyn.T @ H_dyn + eta * D2) @ H_dyn.T
            else:
                Dx, Dy = spytorch.finite_diff_mat(H_dyn.shape[0],
                                                    boundary='neumann')
                D2 = Dx.T @ Dx + Dy.T @ Dy
                print(D2.shape, H_dyn.T.shape)
                pinv = H_dyn.T @ torch.linalg.inv(H_dyn @ H_dyn.T + eta * D2)
                
        else:
            raise NotImplementedError(
                f"Regularization method '{reg}' is not implemented. Please " +
                "choose either 'L1' or 'L2'." #, or 'H1'."
            )
        
        self._param_H_dyn_pinv = nn.Parameter(pinv, requires_grad=False)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates the measurement of a motion picture.

        The output :math:`y` is computed as :math:`y = Hx`, where :math:`H` is
        the measurement matrix and :math:`x` is a batch of vectorized (flattened)
        images.

        .. warning::
            There must be **exactly** as many images as there are measurements
            in the linear operator used to initialize the class, i.e.
            `H.shape[-2] == x.shape[-2]`

        Args:
            :math:`x`: Batch of vectorized (flattened) images.

            :math:`img_shape` (tuple): Image size :math:`(width, height)` as it
            is represented in :math:`x`. :math:`width \times height` must be
            equal to :math:`N`, the last dimension of :math:`x`.

        Shape:
            :math:`x`: :math:`(*, M, N)`, where * denotes the batch size and
            :math:`(M, N)` is the shape of the measurement matrix :math:`H`.
            :math:`M` is the number of measurements (and frames) and :math:`N`
            the number of pixels in the image.
            
            :math:`img_shape`: :math:`(2,)`

            :math:`output`: :math:`(*, M)`

        Example:
            >>> x = torch.rand([10, 400, 1600])
            >>> H = torch.rand([400, 1600])
            >>> meas_op = DynamicLinear(H)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([10, 400])
        """
        return self._forward_with_static_op(x, self.H_static)
    
    def _set_Ord(self, Ord: torch.tensor) -> None:
        """Set the order matrix used to sort the rows of H."""
        super()._set_Ord(Ord)
        # delete self.H (self._param_H_dyn)
        try:
            del self._param_H_dyn
            warnings.warn(
                "The dynamic measurement matrix H has been deleted. " +
                "Please call build_H_dyn() to recompute it."
            )
        except AttributeError:
            pass
        # delete self.H_pinv (self._param_H_dyn_pinv)
        try:
            del self._param_H_dyn_pinv
            warnings.warn(
                "The dynamic pseudo-inverse H_pinv has been deleted. " +
                "Please call build_H_dyn_pinv() to recompute it."
            )
        except AttributeError:
            pass
    
    def _forward_with_static_op(self,
                                x: torch.tensor, 
                                op: torch.tensor
                                ) -> torch.tensor:
        r"""Simulates an acquisition with a specific linear operator.

        Args:
            x (torch.tensor): Image tensor of shape :math:`(*, M, N)` where * 
            denotes any dimension, :math:`M` is the number of measurements and
            :math:`N` the number of pixels in the image.
            
            op (torch.tensor): Operator tensor of shape :math:`(M, N)` where
            :math:`M` is the number of measurements and :math:`N` the number of
            pixels in the image.

        Size:
            x: :math:`(*, M, N)`
            
            op: :math:`(M, N)`
            
            output: :math:`(*, M)`

        Returns:
            torch.tensor: Line-by-line dot product of the operator and the
            input image tensor.
        """
        x_cropped = spytorch.center_crop(x, self.meas_shape, self.img_shape)
        try:
            return torch.einsum("ij,...ij->...i", op.to(x.dtype), x_cropped)
        except RuntimeError as e:
            if "subscript i" in str(e):
                raise RuntimeError(
                    "The number of measurements in the operator must match " +
                    "the number of measurements in the input image. Got " +
                    f"{op.shape[0]} measurements in the operator and " +
                    f"{x_cropped.shape[-2]} measurements in the input image."
                ) from e
            elif "subscript j" in str(e):
                raise RuntimeError(
                    "The number of pixels in the operator must match the " +
                    "number of pixels in the input image. Got " +
                    f"{op.shape[1]} pixels in the operator and " +
                    f"{x_cropped.shape[-1]} pixels in the input image."
                ) from e
            else:
                raise e
        

# =============================================================================
class DynamicLinearSplit(DynamicLinear):
    # =========================================================================
    r"""
    Simulates the measurement of a moving object using the positive and
    negative components of the measurement matrix.

    Computes linear measurements :math:`y` from incoming images: :math:`y = Px`,
    where :math:`P` is a linear operator (matrix) and :math:`x` is a batch of
    vectorized images representing a motion picture.

    The matrix :math:`P` contains only positive values and is obtained by
    splitting a given measurement matrix :math:`H` such that
    :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where
    :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

    The class is constructed from the :math:`M` by :math:`N` matrix :math:`H`,
    where :math:`N` represents the number of pixels in the image and
    :math:`M` the number of measurements.

    Args:
        :math:`H` (torch.tensor): measurement matrix (linear operator) with
        shape :math:`(M, N)` where :math:`M` is the number of measurements and
        :math:`N` the number of pixels in the image.

    Attributes:
        :attr:`H` (torch.nn.Parameter): The learnable measurement matrix of
        shape :math:`(M,N)`.

        :attr:`P` (torch.nn.Parameter): The splitted measurement matrix of
        shape :math:`(2M, N)` initialized as
        :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`
        where :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`

        :attr:`M` (int): Number of measurements performed by the linear operator.
        It is initialized as the first dimension of :math:`H`.

        :attr:`N` (int): Number of pixels in the image. It is initialized as the
        second dimension of :math:`H`.

        :attr:`h` (int): Image height :math:`h`. The image is assumed to be
        square, i.e. :math:`h = \text{floor}(\sqrt{N})`. If not, please assign
        :attr:`h` and :attr:`w` manually.

        :attr:`w` (int): Image width :math:`w`. The image is assumed to be
        square, i.e. :math:`w = \text{floor}(\sqrt{N})`. If not, please assign
        :attr:`h` and :attr:`w` manually.

    .. warning::
        For each call, there must be **exactly** as many images in :math:`x` as
        there are measurements in the linear operator used to initialize the class.

    Example:
        >>> H = torch.rand([400,1600])
        >>> meas_op = DynamicLinearSplit(H)
        >>> print(meas_op)
        DynamicLinearSplit(
            (Image pixels): 1600
            (H): torch.Size([400, 1600])
            (P): torch.Size([800, 1600])
            )
    """

    def __init__(self, 
                 H: torch.tensor, 
                 Ord: torch.tensor=None,
                 meas_shape: tuple=None, # (height, width)
                 img_shape: tuple=None,  # (height, width)
                 ):
        # call constructor of DynamicLinear
        super().__init__(H, Ord, meas_shape, img_shape)
        # initialize P
        self._set_P(self.H_static)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates the measurement of a motion picture using :math:`P`.

        The output :math:`y` is computed as :math:`y = Px`, where :math:`P` is
        the measurement matrix and :math:`x` is a batch of vectorized (flattened)
        images.

        :math:`P` contains only positive values and is obtained by
        splitting a given measurement matrix :math:`H` such that
        :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where
        :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

        The matrix :math:`H` can contain positive and negative values and is
        given by the user at initialization.

        .. warning::
            There must be **exactly** as many images as there are measurements
            in the linear operator used to initialize the class, i.e.
            `P.shape[-2] == x.shape[-2]`

        Args:
            :math:`x`: Batch of vectorized (flattened) images of shape
            :math:`(*, 2M, N)` where * denotes the batch size, :math:`2M` the
            number of measurements in the measurement matrix :math:`P` and
            :math:`N` the number of pixels in the image.

        Shape:
            :math:`x`: :math:`(*, 2M, N)`

            :math:`P` has a shape of :math:`(2M, N)` where :math:`M` is the
            number of measurements as defined by the first dimension of :math:`H`
            and :math:`N` is the number of pixels in the image.

            :math:`output`: :math:`(*, 2M)`

        Example:
            >>> x = torch.rand([10, 800, 1600])
            >>> H = torch.rand([400, 1600])
            >>> meas_op = DynamicLinearSplit(H)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([10, 800])
        """
        return self._forward_with_static_op(x, self.P)

    def forward_H(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates the measurement of a motion picture using :math:`H`.

        The output :math:`y` is computed as :math:`y = Hx`, where :math:`H` is
        the measurement matrix and :math:`x` is a batch of vectorized (flattened)
        images. The positive and negative components of the measurement matrix
        are **not** used in this method.

        The matrix :math:`H` can contain positive and negative values and is
        given by the user at initialization.

        .. warning::
            There must be **exactly** as many images as there are measurements
            in the linear operator used to initialize the class, i.e.
            `H.shape[-2:] == x.shape[-2:]`

        Args:
            :math:`x`: Batch of vectorized (flatten) images of shape
            :math:`(*, M, N)` where * denotes the batch size, and :math:`(M, N)`
            is the shape of the measurement matrix :math:`H`.

        Shape:
            :math:`x`: :math:`(*, M, N)`

            :math:`H` has a shape of :math:`(M, N)` where :math:`M` is the
            number of measurements and :math:`N` is the number of pixels in the
            image.

            :math:`output`: :math:`(*, M)`

        Example:
            >>> x = torch.rand([10, 400, 1600])
            >>> H = torch.rand([400, 1600])
            >>> meas_op = LinearDynamicSplit(H)
            >>> y = meas_op.forward_H(x)
            >>> print(y.shape)
            torch.Size([10, 400])
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
    Simulates the measurement of a moving object using the positive and
    negative components of a Hadamard matrix.

    Computes linear measurements from incoming images: :math:`y = Px`,
    where :math:`P` is a linear operator (matrix) with positive entries and
    :math:`x` is a batch of vectorized images representing a motion picture.

    The class relies on a Hadamard-based matrix :math:`H` with shape :math:`(M,N)`
    where :math:`N` represents the number of pixels in the image and
    :math:`M \le N` the number of measurements. :math:`H` is obtained by
    selecting a re-ordered subsample of :math:`M` rows of a "full" Hadamard
    matrix :math:`F` with shape :math:`(N^2, N^2)`. :math:`N` must be a power
    of 2.

    The matrix :math:`P` is then obtained by splitting the matrix :math:`H`
    such that :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where
    :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

    Args:
        :attr:`M` (int): Number of measurements

        :attr:`h` (int): Image height :math:`h`, must be a power of 2. The
        image is assumed to be square, so the number of pixels in the image is
        :math:`N = h^2`.

        :attr:`Ord` (torch.tensor): Order matrix with shape :math:`(h, h)` used to
        select the rows of the full Hadamard matrix :math:`F`
        compute the permutation matrix :math:`G^{T}` with shape :math:`(N, N)`
        (see the :mod:`~spyrit.misc.sampling` submodule)

    Attributes:
        :attr:`H` (torch.nn.Parameter): The measurement matrix of shape
        :math:`(M, h^2)`. It is initialized as a re-ordered subsample of the
        rows of the "full" Hadamard matrix :math:`F` with shape :math:`(N^2, N^2)`.

        :attr:`H_pinv` (torch.nn.Parameter): The pseudo inverse of the measurement
        matrix of shape :math:`(h^2, M)`. It is initialized as
        :math:`H^\dagger = \frac{1}{N}H^{T}` where :math:`N = h^2`.

        :attr:`P` (torch.nn.Parameter): The splitted measurement matrix of
        shape :math:`(2M, h^2)` initialized as
        :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`
        where :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

        :attr:`Perm` (torch.nn.Parameter): The permutation matrix :math:`G^{T}`
        that is used to re-order the subsample of rows of the "full" Hadamard
        matrix :math:`F` according to descreasing value of the order matrix
        :math:`Ord`. It has shape :math:`(N, N)` where :math:`N = h^2`.

        :attr:`M` (int): Number of measurements performed by the linear operator.

        :attr:`N` (int): Number of pixels in the image. It is initialized as
        :math:`h^2`.

        :attr:`h` (int): Image height :math:`h`.

        :attr:`w` (int): Image width :math:`w`. The image is assumed to be
        square, i.e. :math:`w = h`.

    .. warning::
        For each call, there must be **exactly** as many images in :math:`x` as
        there are measurements in the linear operator used to initialize the class.

    .. note::
        The computation of a Hadamard transform :math:`Fx` benefits a fast
        algorithm, as well as the computation of inverse Hadamard transforms.

    .. note::
        The matrix :math:`H` has shape :math:`(M, N)` with :math:`N = h^2`.

    .. note::
        :math:`H = H_{+} - H_{-}`

    Example:
        >>> Ord = torch.rand([32,32])
        >>> meas_op = HadamSplitDynamic(400, 32, Ord)
        >>> print(meas_op)
        HadamSplitDynamic(
          (Image pixels): 1024
          (H): torch.Size([400, 1024])
          (P): torch.Size([800, 1024])
          (Perm): torch.Size([1024, 1024])
          )
    """

    def __init__(self, 
                 M: int, 
                 h: int, 
                 Ord: torch.tensor=None,
                 img_shape: tuple=None,  # (height, width)
                 ):
        
        F = spytorch.walsh2_matrix(h)        
        # we pass the whole F matrix to the constructor
        super().__init__(F, Ord, (h,h), img_shape)
        self._M = M


# -----------------------------------------------------------------------------

# REGULARIZATION TO BE IMPLEMENTED
# if reg == 'L2':
#     ans = torch.linalg.solve(
#         H_dyn.T @ H_dyn + eta * torch.eye(H_dyn.shape[1]),
#         H_dyn.T @ x.T,
#         driver = 'gelsd').T
    
# elif reg == 'H1':
#     a, b = spytorch.finite_diff_mat(meas_op.img_h, 'neumann')
#     D2 = (a.T @ a + b.T @ b).to_dense()
#     ans = torch.linalg.solve(H_dyn.T @ H_dyn + eta * D2,
#                             H_dyn.T @ x.T).T

# else:
#     print("No valid regularization specified, using least squares")
#     ans = torch.linalg.lstsq(H_dyn, x.T,
#                         rcond=0.05, driver='gelss').solution.T

# return ans.to(orig_dtype)

# BICUBIC INTERPOLATION TO BE IMPLEMENTED

# elif mode == 'bicubic':
# #     # get the integer part of the field for the 16 nearest neighbours
# #     #      00          01   point   02          03
# #     #       +-----------+-----+-----+-----------+
# #     #       |           |           |           |
# #     #       |           |     |     |           |
# #     #       |        11 |           | 12        |
# #     #    10 +-----------+-----+-----+-----------+ 13
# #     #       |           |     |     |           |
# #     # point + - - - - - + - - + - - + - - - - - +
# #     #       |           |     |     |           |
# #     #    20 +-----------+-----+-----+-----------+ 23
# #     #       |        21 |     |     | 22        |
# #     #       |           |           |           |
# #     #       |           |     |     |           |
# #     #       +-----------+-----+-----+-----------+
# #     #      30          31           32          33
    
#     def_field_00 = def_field.floor().to(torch.int32) - 1
#     increments = torch.tensor(
#         [[i,j] for i in range(4) for j in range(4)]
#     ).to(torch.int32) # has order 00, 01, 02, 03, 10, 11, ...
#     def_field_stacked = def_field_00.repeat(16, *[1]*def_field.dim())
#     def_field_stacked += increments.expand_as(def_field_stacked)
