import warnings
import torch
import torch.nn as nn
import numpy as np
from spyrit.misc.walsh_hadamard import walsh2_torch, walsh2_matrix
from spyrit.misc.sampling import Permutation_Matrix


# =============================================================================
class DynamicLinear(nn.Module):
    # =========================================================================
    r"""
    Simulates the measurement of a moving object using the positive and
    negative components of the measurement matrix.

    Computes linear measurements from incoming images: :math:`y = Hx`,
    where :math:`H` is a linear operator (matrix) and :math:`x` is a
    batch of vectorized images representing a motion picture.

    The class is constructed from a matrix :math:`H` of shape :math:`(M, N)`,
    where :math:`N` represents the number of pixels in the image and
    :math:`M` the number of measurements and the number of frames in the
    animated object.

    .. warning::
        For each call, there must be **exactly** as many images in :math:`x` as
        there are measurements in the linear operator used to initialize the class.
        If not, an error will be raised.

    Args:
        :attr:`H`: measurement matrix (linear operator) with shape :math:`(M, N)`.

        :attr:`pinv`: Option to have access to pseudo inverse solutions.
        Defaults to `None` (the pseudo inverse is not initiliazed).

        :attr:`reg` (optional): Regularization parameter (cutoff for small
        singular values, see :mod:`numpy.linal.pinv`). Only relevant when
        :attr:`pinv` is not `None`.

    Attributes:
          :attr:`H`: The learnable measurement matrix of shape
          :math:`(M,N)` initialized as :math:`H`

          :attr:`H_pinv` (optional): The learnable adjoint measurement
          matrix of shape :math:`(N,M)` initialized as :math:`H^\dagger`.
          Only relevant when :attr:`pinv` is not `None`.

    Example 1:
        >>> H = np.random.random([400, 1000])
        >>> meas_op = LinearDynamic(H)
        >>> print(meas_op)
        LinearDynamic(
          (H): Linear(in_features=1000, out_features=400, bias=False)
          )

    Example 2:
        >>> H = np.random.random([400, 1000])
        >>> meas_op = LinearDynamic(H, True)
        >>> print(meas_op)
        LinearDynamic(
          (H): Linear(in_features=1000, out_features=400, bias=False)
          (H_pinv): Linear(in_features=400, out_features=1000, bias=False)
        )
    """

    def __init__(self, H: np.ndarray | torch.tensor, pinv=None, reg: float = 1e-15):
        super().__init__()

        # nn.Parameter are sent to the device when using .to(device),
        # contrary to attributes
        H = torch.tensor(H, dtype=torch.float32)
        self.H = nn.Parameter(H, requires_grad=False)

        self.M = H.shape[0]
        self.N = H.shape[1]
        self.h = int(self.N**0.5)
        self.w = int(self.N**0.5)
        if self.h * self.w != self.N:
            warnings.warn(
                "N is not a square. Please assign self.h and self.w manually."
            )
        if pinv is not None:
            H_pinv = torch.linalg.pinv(H, rcond=reg)
            self.H_pinv = nn.Parameter(H_pinv, requires_grad=False)
        else:
            print("Pseudo inverse will not be instanciated")

    def get_H(self) -> torch.tensor:
        r"""Returns the measurement matrix :math:`H`.

        Shape:
            Output: :math:`(M, N)`

        Example:
            >>> H1 = np.random.random([400, 1000])
            >>> meas_op = Linear(H1)
            >>> H2 = meas_op.get_H()
            >>> print('Matrix shape:', H2.shape)
            Matrix shape: torch.Size([400, 1000])
        """
        return self.H.data

    def get_H_T(self) -> torch.tensor:
        r"""
        Returns the transpose of the measurement matrix :math:`H`.

        Shape:
            Output: :math:`(N, M)`

        Example:
            >>> H1 = np.random.random([400, 1000])
            >>> meas_op = Linear(H1)
            >>> H2 = meas_op.get_H_T()
            >>> print('Transpose shape:', H2.shape)
            Transpose shape: torch.Size([400, 1000])
        """
        return self.H.T

    def get_H_pinv(self) -> torch.tensor:
        r"""Returns the pseudo inverse of the measurement matrix :math:`H`.

        Shape:
            Output: :math:`(N, M)`

        Example:
            >>> H1 = np.random.random([400, 1000])
            >>> meas_op = Linear(H1, True)
            >>> H2 = meas_op.get_H_pinv()
            >>> print('Pseudo inverse shape:', H2.shape)
            Pseudo inverse shape: torch.Size([1000, 400])
        """
        return self.H_pinv.data

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates the measurement of a motion picture.

        The output :math:`y` is computed as :math:`y = Hx`, where :math:`H` is
        the measurement matrix and :math:`x` is a batch of vectorized (flattened)
        images.

        .. warning::
            There must be **exactly** as many images as there are measurements
            in the linear operator used to initialize the class, i.e.
            `H.shape[-2:] == x.shape[-2:]

        Args:
            :math:`x`: Batch of vectorized (flattened) images.

        Shape:
            :math:`x`: :math:`(*, M, N)`
            :math:`output`: :math:`(*, M)`

        Example:
            >>> x = torch.rand([10, 400, 1000], dtype=torch.float)
            >>> H = np.random.random([400, 1000])
            >>> meas_op = LinearDynamic(H)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([10, 400])
        """
        return torch.einsum("ij,...ij->...i", self.get_H(), x)


# =============================================================================
class DynamicLinearSplit(DynamicLinear):
    # =========================================================================
    r"""
    Used to simulate the measurement of a moving object using the positive and
    negative components of the measurement matrix.

    Computes linear measurements from incoming images: :math:`y = Px`,
    where :math:`P` is a linear operator (matrix) and :math:`x` is a batch of
    vectorized images representing a motion picture.

    The matrix :math:`P` contains only positive values and is obtained by
    splitting a measurement matrix :math:`H` such that
    :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where
    :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

    The class is constructed from the :math:`M` by :math:`N` matrix :math:`H`,
    where :math:`N` represents the number of pixels in the image and
    :math:`M` the number of measurements.

    .. warning::
        For each call, there must be **exactly** as many images in :math:`x` as
        there are measurements in the linear operator used to initialize the class.
        If not, an error will be raised.

    Args:
        :math:`H` (np.ndarray): measurement matrix (linear operator) with
        shape :math:`(M, N)`.

    Example:
        >>> H = np.array(np.random.random([400,1000]))
        >>> meas_op = LinearDynamicSplit(H)
    """

    def __init__(self, H: np.ndarray, pinv=None, reg: float = 1e-15):
        # initialize self.H and self.H_pinv
        super().__init__(H, pinv, reg)
        # initialize self.P = [ H^+ ]
        #                     [ H^- ]
        zero = torch.zeros(1)
        H_pos = torch.maximum(zero, H)
        H_neg = torch.maximum(zero, -H)
        # concatenate side by side, then reshape vertically
        P = torch.cat([H_pos, H_neg], 1).view(2 * self.M, self.N)
        self.P = nn.Parameter(P, requires_grad=False)

    def get_P(self) -> torch.tensor:
        r"""Returns the measurement matrix :math:`P`.

        Shape:
            Output: :math:`(2M, N)`

        Example:
            >>> P = meas_op.get_P()
            >>> print('Matrix shape:', P.shape)
            Matrix shape: torch.Size([800, 1000])
        """
        return self.P.data

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates the measurement of a motion picture using :math:`P`.

        The output :math:`y` is computed as :math:`y = Px`, where :math:`P` is
        the measurement matrix and :math:`x` is a batch of vectorized (flattened)
        images.

        :math:`P` contains only positive values and is obtained by
        splitting a measurement matrix :math:`H` such that
        :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where
        :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

        .. warning::
            There must be **exactly** as many images as there are measurements
            in the linear operator used to initialize the class, i.e.
            `P.shape[-2:] == x.shape[-2:]

        Args:
            :math:`x`: Batch of vectorized (flatten) images.

        Shape:
            :math:`P` has a shape of :math:`(2M, N)` where :math:`M` is the
            number of measurements as defined by the first dimension of :math:`H`
            and :math:`N` is the number of pixels in the image.

            :math:`x`: :math:`(*, 2M, N)`

            :math:`output`: :math:`(*, 2M)`

        Example:
            >>> x = torch.rand([10, 400, 1000], dtype=torch.float)
            >>> H = np.random.random([400, 1000])
            >>> meas_op = LinearDynamicSplit(H)
            >>> y = meas_op(x)
            >>> print(y.shape)
            torch.Size([10, 800])
        """
        return torch.einsum("ij,...ij->...i", self.get_P(), x)

    def forward_H(self, x: torch.tensor) -> torch.tensor:
        r"""
        Simulates the measurement of a motion picture using :math:`H`.

        The output :math:`y` is computed as :math:`y = Hx`, where :math:`H` is
        the measurement matrix and :math:`x` is a batch of vectorized (flattened)
        images. The positive and negative components of the measurement matrix
        are **not** used in this method.

        .. warning::
            There must be **exactly** as many images as there are measurements
            in the linear operator used to initialize the class, i.e.
            `H.shape[-2:] == x.shape[-2:]

        Args:
            :math:`x`: Batch of vectorized (flatten) images.

        Shape:
            :math:`H` has a shape of :math:`(M, N)` where :math:`M` is the
            number of measurements and :math:`N` is the number of pixels in the
            image.

            :math:`x`: :math:`(*, M, N)`

            :math:`output`: :math:`(*, M)`

        Example:
            >>> x = torch.rand([10, 400, 1000], dtype=torch.float)
            >>> H = np.random.random([400, 1000])
            >>> meas_op = LinearDynamicSplit(H)
            >>> y = meas_op.forward_H(x)
            >>> print(y.shape)
            torch.Size([10, 400])
        """
        return super.forward(x)


# =============================================================================
class DynamicHadamSplit(DynamicLinearSplit):
    # =========================================================================
    r"""
    Simulates the measurement of a moving object using the positive and
    negative components of a Hadamard matrix.

    Computes linear measurements from incoming images: :math:`y = Px`,
    where :math:`P` is a linear operator (matrix) with positive entries and
    :math:`x` is a batch of vectorized images representing a motion picture.

    The class relies on a matrix :math:`H` with
    shape :math:`(M,N)` where :math:`N` represents the number of pixels in the
    image and :math:`M \le N` the number of measurements. The matrix :math:`P`
    is obtained by splitting the matrix :math:`H` such that
    :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where
    :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

    The matrix :math:`H` is obtained by retaining the first :math:`M` rows of
    a permuted Hadamard matrix :math:`GF`, where :math:`G` is a
    permutation matrix with shape with shape :math:`(M,N)` and :math:`F` is a
    "full" Hadamard matrix with shape :math:`(N,N)`. The computation of a
    Hadamard transform :math:`Fx` benefits a fast algorithm, as well as the
    computation of inverse Hadamard transforms.

    .. warning::
        For each call, there must be **exactly** as many images in :math:`x` as
        there are measurements in the linear operator used to initialize the class.
        If not, an error will be raised.

    Args:
        :attr:`M` (int): Number of measurements

        :attr:`h` (int): Image height :math:`h`. The image is assumed to be square.

        :attr:`Ord` (np.ndarray): Order matrix with shape :math:`(h,h)` used to
        compute the permutation matrix :math:`G^{T}` with shape :math:`(N, N)`
        (see the :mod:`~spyrit.misc.sampling` submodule)

    .. note::
        The matrix H has shape :math:`(M,N)` with :math:`N = h^2`.

    .. note::
        :math:`H = H_{+} - H_{-}`

    Example:
        >>> Ord = np.random.random([32,32])
        >>> meas_op = HadamSplitDynamic(400, 32, Ord)
    """

    def __init__(self, M: int, h: int, Ord: np.ndarray):
        F = walsh2_matrix(h)  # full matrix
        Perm = Permutation_Matrix(Ord)
        F = Perm @ F  # If Perm is not learnt, could be computed mush faster
        H = F[:M, :]
        w = h  # we assume a square image

        super().__init__(H)

        Perm = torch.tensor(Perm, dtype=torch.float32)
        self.Perm = nn.Parameter(Perm, requires_grad=False)
        # overwrite self.h and self.w
        self.h = h
        self.w = w


# =============================================================================
class Linear(DynamicLinear):
    # =========================================================================
    r"""
    Simulates the measurement of an image using a measurement operator.

    Computes linear measurements from incoming images: :math:`y = Hx`,
    where :math:`H` is a linear operator (matrix) and :math:`x` is a
    vectorized image or a batch of images.

    The class is constructed from a :math:`M` by :math:`N` matrix :math:`H`,
    where :math:`N` represents the number of pixels in the image and
    :math:`M` the number of measurements.

    Args:
        :attr:`H`: measurement matrix (linear operator) with shape :math:`(M, N)`.

        :attr:`pinv`: Option to have access to pseudo inverse solutions.
        Defaults to `None` (the pseudo inverse is not initiliazed).

        :attr:`reg` (optional): Regularization parameter (cutoff for small
        singular values, see :mod:`numpy.linal.pinv`). Only relevant when
        :attr:`pinv` is not `None`.


    Attributes:
          :attr:`H`: The learnable measurement matrix of shape
          :math:`(M,N)` initialized as :math:`H`

          :attr:`H_adjoint`: The learnable adjoint measurement matrix
          of shape :math:`(N,M)` initialized as :math:`H^\top`

          :attr:`H_pinv` (optional): The learnable adjoint measurement
          matrix of shape :math:`(N,M)` initialized as :math:`H^\dagger`.
          Only relevant when :attr:`pinv` is not `None`.

    Example 1:
        >>> H = np.random.random([400, 1000])
        >>> meas_op = Linear(H)
        >>> print(meas_op)
        Linear(
          (H): Linear(in_features=1000, out_features=400, bias=False)
          )

    Example 2:
        >>> H = np.random.random([400, 1000])
        >>> meas_op = Linear(H, True)
        >>> print(meas_op)
        Linear(
          (H): Linear(in_features=1000, out_features=400, bias=False)
          (H_pinv): Linear(in_features=400, out_features=1000, bias=False)
        )
    """

    def __init__(self, H: np.ndarray, pinv=None, reg: float = 1e-15):
        super().__init__(H, pinv, reg)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Applies linear transform to incoming images: :math:`y = Hx`.

        Args:
            :math:`x`: Batch of vectorized (flatten) images.

        Shape:
            :math:`x`: :math:`(*, N)` where * denotes the batch size and `N`
            the total number of pixels in the image.

            Output: :math:`(*, M)` where * denotes the batch size and `M`
            the number of measurements.

        Example:
            >>> x = torch.rand([10,1000], dtype=torch.float)
            >>> y = meas_op(x)
            >>> print('forward:', y.shape)
            forward: torch.Size([10, 400])

        """
        # left multiplication with transpose is equivalent to right mult
        return x @ self.get_H_T()

    def adjoint(self, x: torch.tensor) -> torch.tensor:
        r"""Applies adjoint transform to incoming measurements :math:`y = H^{T}x`

        Args:
            :math:`x`:  batch of measurement vectors.

        Shape:
            :math:`x`: :math:`(*, M)`

            Output: :math:`(*, N)`

        Example:
            >>> x = torch.rand([10,400], dtype=torch.float)
            >>> y = meas_op.adjoint(x)
            >>> print('adjoint:', y.shape)
            adjoint: torch.Size([10, 1000])
        """
        # left multiplication is equivalent to right mult with transpose
        return x @ self.get_H()

    def pinv(self, x: torch.tensor) -> torch.tensor:
        r"""Computes the pseudo inverse solution :math:`y = H^\dagger x`

        Args:
            :math:`x`:  batch of measurement vectors.

        Shape:
            :math:`x`: :math:`(*, M)`

            Output: :math:`(*, N)`

        Example:
            >>> x = torch.rand([10,400], dtype=torch.float)
            >>> y = meas_op.pinv(x)
            >>> print('pinv:', y.shape)
            adjoint: torch.Size([10, 1000])
        """
        # Pmat.transpose()*f
        return x @ self.get_H_pinv().T


# =============================================================================
class LinearSplit(Linear, DynamicLinearSplit):
    # =========================================================================
    r"""
    Simulates the measurement of an image using the computed positive and
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
        :math:`H` (np.ndarray): measurement matrix (linear operator) with
        shape :math:`(M, N)`.

    Example:
        >>> H = np.array(np.random.random([400,1000]))
        >>> meas_op =  LinearSplit(H)
    """

    def __init__(self, H: np.ndarray, pinv=None, reg: float = 1e-15):
        # initialize from DynamicLinearSplit __init__
        super(Linear, self).__init__(H, pinv, reg)

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Applies linear transform to incoming images: :math:`y = Px`.

        Args:
            :math:`x`: Batch of vectorized (flatten) images.

        Shape:
            :math:`x`: :math:`(*, N)` where * denotes the batch size and `N`
            the total number of pixels in the image.

            Output: :math:`(*, 2M)` where * denotes the batch size and `M`
            the number of measurements.

        Example:
            >>> x = torch.rand([10,1000], dtype=torch.float)
            >>> y = meas_op(x)
            >>> print('Output:', y.shape)
            Output: torch.Size([10, 800])
        """
        # x.shape[b*c,N]
        # output shape : [b*c, 2*M]
        return x @ self.get_P().T

    def forward_H(self, x: torch.tensor) -> torch.tensor:
        r"""Applies linear transform to incoming images: :math:`m = Hx`.

        Args:
            :math:`x`: Batch of vectorized (flatten) images.

        Shape:
            :math:`x`: :math:`(*, N)` where * denotes the batch size and `N`
            the total number of pixels in the image.

            Output: :math:`(*, M)` where * denotes the batch size and `M`
            the number of measurements.

        Example:
            >>> x = torch.rand([10,1000], dtype=torch.float)
            >>> y = meas_op.forward_H(x)
            >>> print('Output:', y.shape)
            output shape: torch.Size([10, 400])

        """
        # call Linear.forward() method
        return super(LinearSplit, self).forward(x)


# =============================================================================
class HadamSplit(LinearSplit, DynamicHadamSplit):
    # =========================================================================
    r"""
    Simulates the measurement of a moving object using the positive and
    negative components of a Hadamard matrix.

    Computes linear measurements from incoming images: :math:`y = Px`,
    where :math:`P` is a linear operator (matrix) with positive entries and
    :math:`x` is a vectorized image or a batch of images.

    The class relies on a matrix :math:`H` with
    shape :math:`(M,N)` where :math:`N` represents the number of pixels in the
    image and :math:`M \le N` the number of measurements. The matrix :math:`P`
    is obtained by splitting the matrix :math:`H` such that
    :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`, where
    :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

    The matrix :math:`H` is obtained by retaining the first :math:`M` rows of
    a permuted Hadamard matrix :math:`GF`, where :math:`G` is a
    permutation matrix with shape with shape :math:`(M,N)` and :math:`F` is a
    "full" Hadamard matrix with shape :math:`(N,N)`. The computation of a
    Hadamard transform :math:`Fx` benefits a fast algorithm, as well as the
    computation of inverse Hadamard transforms.

    Args:
        :attr:`M` (int): Number of measurements

        :attr:`h` (int): Image height :math:`h`. The image is assumed to be square.

        :attr:`Ord` (np.ndarray): Order matrix with shape :math:`(h,h)` used to
        compute the permutation matrix :math:`G^{T}` with shape :math:`(N, N)`
        (see the :mod:`~spyrit.misc.sampling` submodule)

    .. note::
        The matrix H has shape :math:`(M,N)` with :math:`N = h^2`.

    .. note::
        :math:`H = H_{+} - H_{-}`

    Example:
        >>> Ord = np.random.random([32,32])
        >>> meas_op = HadamSplit(400, 32, Ord)
    """

    def __init__(self, M: int, h: int, Ord: np.ndarray):
        # initialize from DynamicHadamSplit __init__
        super(LinearSplit, self).__init__(M, h, Ord)

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

            >>> y = torch.rand([85,32*32], dtype=torch.float)
            >>> x = meas_op.inverse(y)
            >>> print('Inverse:', x.shape)
            Inverse: torch.Size([85, 1024])
        """
        # permutations
        # todo: check walsh2_S_fold_torch to speed up
        b, N = x.shape
        x = self.Perm(x)
        x = x.view(b, 1, self.h, self.w)
        # inverse of full transform
        # todo: initialize with 1D transform to speed up
        x = 1 / self.N * walsh2_torch(x)
        return x.view(b, N)

    def pinv(self, x: torch.tensor) -> torch.tensor:
        r"""Pseudo inverse transform of incoming mesurement vectors :math:`x`

        Args:
            :attr:`x`:  batch of measurement vectors.

        Shape:
            x: :math:`(*, M)`

            Output: :math:`(*, N)`

        Example:
            >>> y = torch.rand([85,400], dtype=torch.float)
            >>> x = meas_op.pinv(y)
            >>> print(x.shape)
            torch.Size([85, 1024])
        """
        #
        return self.adjoint(x) / self.N
