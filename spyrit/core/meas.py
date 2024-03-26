import warnings

import torch
import torch.nn as nn
import numpy as np
import scipy.stats

from spyrit.misc.walsh_hadamard import walsh2_torch, walsh2_matrix
from spyrit.misc.sampling import Permutation_Matrix, sort_by_significance
from spyrit.core.time import DeformationField, AffineDeformationField


# =============================================================================
class DynamicLinear(nn.Module):
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
        >>> H = np.random.random([400, 1600])
        >>> meas_op = DynamicLinear(H)
        >>> print(meas_op)
        DynamicLinear(
          (Image pixels): 1600
          (H): torch.Size([400, 1600])
          )
    """

    def __init__(self, H: torch.tensor):
        super().__init__()

        # convert H from numpy to torch tensor if needed
        # convert to float 32 for memory efficiency
        if isinstance(H, np.ndarray):
            H = torch.from_numpy(H)
            warnings.warn(
                "Using a numpy array is deprecated. Please use a torch tensor instead.",
                DeprecationWarning,
            )
        H = H.type(torch.float32)
        # nn.Parameter are sent to the device when using .to(device),
        self.H = nn.Parameter(H, requires_grad=False)

        self.M = H.shape[0]
        self.N = H.shape[1]
        self.h = int(self.N**0.5)
        self.w = int(self.N**0.5)
        if self.h * self.w != self.N:
            warnings.warn(
                f"N ({H.shape[1]}) is not a square. Please assign self.h and self.w manually."
            )

    def get_H(self) -> torch.tensor:
        r"""Returns the attribute measurement matrix :math:`H`.

        Shape:
            Output: :math:`(M, N)`

        Example:
            >>> H1 = np.random.random([400, 1600])
            >>> meas_op = Linear(H1)
            >>> H2 = meas_op.get_H()
            >>> print(H2.shape)
            torch.Size([400, 1600])
        """
        return self.H.data

    def get_H_pinv(self) -> torch.tensor:
        r"""Returns the pseudo inverse of the measurement matrix :math:`H`.

        Shape:
            Output: :math:`(N, M)`

        Example:
            >>> H1 = np.random.random([400, 1600])
            >>> meas_op = Linear(H1, True)
            >>> H2 = meas_op.get_H_pinv()
            >>> print(H2.shape)
            torch.Size([1600, 400])
        """
        try:
            return self.H_pinv.data
        except AttributeError as e:
            if "has no attribute 'H_pinv'" in str(e):
                raise AttributeError(
                    "The pseudo inverse has not been initialized. Please set it using self.set_H_pinv()."
                )
            else:
                raise e

    def set_H_pinv(self, reg: float = 1e-15, pinv: torch.tensor = None) -> None:
        r"""
        Stores in self.H_pinv the pseudo inverse of the measurement matrix :math:`H`.

        If :attr:`pinv` is given, it is directly stored as the pseudo inverse.
        The validity of the pseudo inverse is not checked. If :attr:`pinv` is
        :obj:`False`, the pseudo inverse is computed from the existing
        measurement matrix :math:`H` with regularization parameter :attr:`reg`.

        Args:
            :attr:`reg` (float, optional): Cutoff for small singular values.

            :attr:`H_pinv` (torch.tensor, optional): If given, the tensor is
            directly stored as the pseudo inverse. No checks are performed.
            Otherwise, the pseudo inverse is computed from the existing
            measurement matrix :math:`H`.

        .. note:
            Only one of :math:`H_pinv` and :math:`reg` should be given. If both
            are given, :math:`H_pinv` is used and :math:`reg` is ignored.

        Shape:
            :attr:`H_pinv`: :math:`(N, M)`, where :math:`N` is the number of
            pixels in the image and :math:`M` the number of measurements.

        Example:
            >>> H1 = torch.rand([400, 1600])
            >>> H2 = torch.linalg.pinv(H1)
            >>> meas_op = Linear(H1)
            >>> meas_op.set_H_pinv(H2)
        """
        if pinv is not None:
            H_pinv = pinv.type(torch.FloatTensor)  # to float32
        else:
            H_pinv = torch.linalg.pinv(self.get_H(), rcond=reg)
        self.H_pinv = nn.Parameter(H_pinv, requires_grad=False)

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

        Shape:
            :math:`x`: :math:`(*, M, N)`, where * denotes the batch size and
            :math:`(M, N)` is the shape of the measurement matrix :math:`H`.
            :math:`M` is the number of measurements (and frames) and :math:`N`
            the number of pixels in the image.

            :math:`output`: :math:`(*, M)`

        Example:
            >>> x = torch.rand([10, 400, 1600])
            >>> H = np.random.random([400, 1600])
            >>> meas_op = DynamicLinear(H)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([10, 400])
        """
        try:
            return torch.einsum("ij,...ij->...i", self.get_H(), x)
        except RuntimeError as e:
            if "which does not broadcast with previously seen size" in str(e):
                raise ValueError(
                    f"The shape of the input x ({x.shape}) does not match the "
                    + f"shape of the measurement matrix H ({self.get_H().shape})."
                )
            else:
                raise e

    def __str__(self):
        s_begin = f"{self.__class__.__name__}(\n  "
        s_fill = "\n  ".join([f"({k}): {v}" for k, v in self._attributeslist()])
        s_end = "\n  )"
        return s_begin + s_fill + s_end

    def _attributeslist(self):
        return [("Image pixels", self.N), ("H", self.H.shape)]


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
        :math:`H` (np.ndarray): measurement matrix (linear operator) with
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
        >>> H = np.array(np.random.random([400,1600]))
        >>> meas_op = DynamicLinearSplit(H)
        >>> print(meas_op)
        DynamicLinearSplit(
            (Image pixels): 1600
            (H): torch.Size([400, 1600])
            (P): torch.Size([800, 1600])
            )
    """

    def __init__(self, H: torch.tensor):
        # initialize self.H
        if isinstance(H, np.ndarray):
            H = torch.from_numpy(H)
            warnings.warn(
                "Using a numpy array is deprecated. Please use a torch tensor instead.",
                DeprecationWarning,
            )
        H = H.type(torch.float32)

        super().__init__(H)

        # initialize self.P = [ H^+ ]
        #                     [ H^- ]
        zero = torch.zeros(1)
        H_pos = torch.maximum(zero, H)
        H_neg = torch.maximum(zero, -H)
        # concatenate side by side, then reshape vertically
        P = torch.cat([H_pos, H_neg], 1).view(2 * self.M, self.N)
        P = P.type(torch.FloatTensor)  # cast to float 32
        self.P = nn.Parameter(P, requires_grad=False)

    def get_P(self) -> torch.tensor:
        r"""Returns the attribute measurement matrix :math:`P`.

        Shape:
            Output: :math:`(2M, N)`, where :math:`(M, N)` is the shape of the
            measurement matrix :math:`H` given at initialization.

        Example:
            >>> H = np.random.random([400, 1600])
            >>> meas_op = LinearDynamicSplit(H)
            >>> P = meas_op.get_P()
            >>> print(P.shape)
            torch.Size([800, 1600])
        """
        return self.P.data

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
            >>> H = np.random.random([400, 1600])
            >>> meas_op = DynamicLinearSplit(H)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([10, 800])
        """
        try:
            return torch.einsum("ij,...ij->...i", self.get_P(), x)
        except RuntimeError as e:
            if "which does not broadcast with previously seen size" in str(e):
                raise ValueError(
                    f"The shape of the input x ({x.shape}) does not match the "
                    + f"shape of the measurement matrix P ({self.get_P().shape})."
                )
            else:
                raise e

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
            >>> H = np.random.random([400, 1600])
            >>> meas_op = LinearDynamicSplit(H)
            >>> y = meas_op.forward_H(x)
            >>> print(y.shape)
            torch.Size([10, 400])
        """
        return super().forward(x)

    def _attributeslist(self):
        return super()._attributeslist() + [("P", self.P.shape)]


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

        :attr:`Ord` (np.ndarray): Order matrix with shape :math:`(h, h)` used to
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
        >>> Ord = np.random.random([32,32])
        >>> meas_op = HadamSplitDynamic(400, 32, Ord)
        >>> print(meas_op)
        HadamSplitDynamic(
          (Image pixels): 1024
          (H): torch.Size([400, 1024])
          (P): torch.Size([800, 1024])
          (Perm): torch.Size([1024, 1024])
          )
    """

    def __init__(self, M: int, h: int, Ord: np.ndarray):
        F = walsh2_matrix(h)  # full matrix
        H = sort_by_significance(F, Ord, "rows", False)[:M, :]  # much faster
        w = h  # we assume a square image

        super().__init__(torch.from_numpy(H))

        # overwrite self.h and self.w   /!\   is it necessary?
        self.h = h
        self.w = w

        #######################################################################
        # these lines can be deleted in a future version, along with the
        # method self.get_Perm()
        #######################################################################
        Perm = Permutation_Matrix(Ord)
        Perm = torch.from_numpy(Perm).float()  # float32
        self.Perm = nn.Parameter(Perm.T, requires_grad=False)

    def get_Perm(self) -> torch.tensor:
        warnings.warn(
            "The attribute 'Perm' will be removed in a future version.",
            DeprecationWarning,
        )
        return self.Perm.data


# =============================================================================
class Linear(DynamicLinear):
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

        :attr:`reg` (float, optional): Regularization parameter (cutoff for small
        singular values, see :mod:`numpy.linal.pinv`). Only relevant when
        :attr:`pinv` is `True`.

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
        >>> H = np.random.random([400, 1600])
        >>> meas_op = Linear(H, pinv=False)
        >>> print(meas_op)
        Linear(
          (Image pixels): 1600
          (H): torch.Size([400, 1600])
          (H_pinv): None
          )

    Example 2:
        >>> H = np.random.random([400, 1600])
        >>> meas_op = Linear(H, True)
        >>> print(meas_op)
        Linear(
          (Image pixels): 1600
          (H): torch.Size([400, 1600])
          (H_pinv): torch.Size([1600, 400])
          )
    """

    def __init__(self, H: np.ndarray, pinv=False, reg: float = 1e-15):
        super().__init__(H)
        if pinv:
            self.set_H_pinv(reg=reg)

    def get_H_T(self) -> torch.tensor:
        r"""
        Returns the transpose of the measurement matrix :math:`H`.

        Shape:
            Output: :math:`(N, M)`, where :math:`N` is the number of pixels in
            the image and :math:`M` the number of measurements.

        Example:
            >>> H1 = np.random.random([400, 1600])
            >>> meas_op = Linear(H1)
            >>> H2 = meas_op.get_H_T()
            >>> print(H2.shape)
            torch.Size([400, 1600])
        """
        return self.H.T

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
        return torch.matmul(x, self.get_H().T)

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
        return torch.matmul(x, self.get_H_T().T)

    def pinv(self, x: torch.tensor) -> torch.tensor:
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
        # Pmat.transpose()*f
        return torch.matmul(x, self.get_H_pinv().T)

    def _attributeslist(self):
        return super()._attributeslist() + [
            ("H_pinv", self.H_pinv.shape if hasattr(self, "H_pinv") else None)
        ]


# =============================================================================
class LinearSplit(Linear, DynamicLinearSplit):
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

        :attr:`reg` (float, optional): Regularization parameter (cutoff for small
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

    def __init__(self, H: np.ndarray, pinv=False, reg: float = 1e-15):
        # initialize from DynamicLinearSplit __init__
        super(Linear, self).__init__(H)
        if pinv:
            self.set_H_pinv(reg)

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
        return torch.matmul(x, self.get_P().T)

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
        return super(LinearSplit, self).forward(x)


# =============================================================================
class HadamSplit(LinearSplit, DynamicHadamSplit):
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

        :attr:`Ord` (np.ndarray): Order matrix with shape :math:`(h, h)` used to
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

    def __init__(self, M: int, h: int, Ord: np.ndarray):
        # initialize from DynamicHadamSplit (the MRO is not trivial here)
        super(Linear, self).__init__(M, h, Ord)
        self.set_H_pinv(pinv=1 / self.N * self.get_H_T())

        # store Ord as attribute for use of self.inverse() method
        Ord = torch.from_numpy(Ord).float()  # float32
        self.Ord = nn.Parameter(Ord, requires_grad=False)

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

        # False because self.Perm is already permuted vvvvv
        x = sort_by_significance(x, self.Ord, "cols", False)  # new way
        # x = x @ self.Perm.T                               # old way

        x = x.view(b, 1, self.h, self.w)
        # inverse of full transform
        # todo: initialize with 1D transform to speed up
        x = 1 / self.N * walsh2_torch(x)
        return x.view(b, N)

    def _attributeslist(self):
        return super()._attributeslist() + [("Perm", self.Ord.shape)]


# =============================================================================
# Functions
# =============================================================================


def set_dyn_pinv(
    meas_op: DynamicLinear,
    motion: DeformationField,
    interp_mode: str = "bilinear",
    # regularizer: str=None,
    reg: float = 1e-15,
) -> None:

    # get full image shape, defined by motion
    Nx_img, Ny_img = motion.Nx, motion.Ny
    n_frames = motion.n_frames

    # get measurement matrix shape
    Nx_meas = meas_op.w
    Ny_meas = meas_op.h
    n_meas = meas_op.M

    if Nx_meas * Ny_meas != meas_op.N:
        raise ValueError(
            f"The image size in the measurement operator is not a square. "
            + "Please assign self.h and self.w manually."
        )

    # get the measurement matrix
    H = meas_op.get_H().view(n_meas, 1, Nx_meas, Ny_meas)

    # if the image is larger than the measurement matrix, we need to pad the
    # measurement matrix with zeros
    if (Nx_img > Nx_meas) or (Ny_img > Ny_meas):
        pad_left = (Ny_img - Ny_meas) // 2
        pad_right = Ny_img - Ny_meas - pad_left
        pad_top = (Nx_img - Nx_meas) // 2
        pad_bottom = Nx_img - Nx_meas - pad_top

        pad = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0)
        H = pad(H)

    H_dyn_physical = motion(
        H, 0, meas_op.M, mode=interp_mode
    )  #########################
    # is it the reciprocal motion we are looking for ?

    # find pseudo inverse according to regularizer
    # first no regularizer
    H_pinv = torch.linalg.pinv(H_dyn_physical, rcond=reg)
    meas_op.set_H_pinv(pinv=H_pinv)
