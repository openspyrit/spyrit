"""
Contains pytorch-based functions used in spyrit.core modules.

The goal of this module is to provide a set of functions that use various
pytorch functionalities and optimizations to perform the necessary operations
in the spyrit.core modules. It mirrors the the spyrit.misc most used
functions, but using pytorch tensors instead of numpy arrays.
"""

# import warnings

import math
import torch
import torch.nn as nn
import torchvision

import spyrit.misc.walsh_hadamard as wh
import spyrit.core.warp as warp


# =============================================================================
# Walsh / Hadamard -related functions
# =============================================================================
def assert_power_of_2(n, raise_error=True):
    r"""Asserts that n is a power of 2.

    Args:
        n (int): The number to check.

        raise_error (bool, optional): Whether to raise an error if n is not a
        power of 2 or not. Default is True.

    Raises:
        ValueError: If n is not a power of 2 and if raise_error is True.

    Returns:
        bool: True if n is a power of 2, False otherwise.

    Example:
        >>> from spyrit.core import torch
        >>> torch.assert_power_of_2(64)
        True
    """
    if n < 1:
        if raise_error:
            raise ValueError("n must be a positive integer.")
        return False
    if n & (n - 1) == 0:
        return True
    if raise_error:
        raise ValueError("n must be a power of 2.")
    return False


def sequency_perm(X, ind=None):
    r"""Permute the last dimension of a tensor. By defaults this allows the sequency order to be obtained from the natural order.

    Args:
        :attr:`X` (torch.tensor): input of shape (*,n)

        :attr:`ind` : list of index length n. Defaults to indices to get sequency order.

    Returns:
        torch.tensor: output of shape (*,n).

    Note:
        Same as :func:`spyrit.misc.walsh_hadamard.sequency_perm()` for torch tensors.

    Example :
        >>> import torch
        >>> import spyrit.core.torch as st
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1, -2])
        >>> x = x[None, None, :]
        >>> x = st.sequency_perm(x)
        >>> print(x)
        tensor([[[ 1,  7,  1,  0, -1, -2,  5,  3]]])
        >>> print(x.shape)
        torch.Size([1, 1, 8])
    """
    if ind is None:
        ind = wh.sequency_perm_ind(X.shape[-1])

    Y = X[..., ind]
    return Y


def walsh_matrix(n):
    r"""Returns a 1D Walsh-ordered Hadamard.

    Args:
        :attr:`n` (:obj:`int`): Order of the transform :math:`n`, which must be a power of two.

    Raises:
        ValueError: If :attr:`n` is not a positive integer that is a power of 2.

    Returns:
        torch.tensor:  Matrix :math:`H` with shape :math:`(n,n)`.
    """
    assert_power_of_2(n, raise_error=True)

    # define recursive function
    def recursive_walsh(k):
        if k >= 3:
            j = k // 2
            a = recursive_walsh(j)
            out = torch.empty((k, k), dtype=torch.float32)

            # generate upper half of the matrix
            out[:j, ::2] = a
            out[:j, 1::2] = a
            # by symmetry, fill in lower left corner
            out[j:, :j] = out[:j, j:].T
            # fill in lower right corner
            alternate = torch.tensor([1, -1]).repeat(j // 2)
            out[j:, j:] = alternate * (out[:j, j:]).flip(0)
            return out
        elif k == 2:
            return torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        else:
            return torch.tensor([[1.0]])

    return recursive_walsh(n)


def walsh_matrix_2d(n):
    r"""2D Walsh-ordered Hadamard matrix.

    This is the matrix :math:`A\in\mathbb{R}^{n^2 \times n^2}` such that :math:`Ax` represents the 2D Hadamard transform of the vectorised image :math:`x`.

    Args:
        :attr:`n` (:obj:`int`): Order of the transform :math:`n`, which must be a power of two.

    Raises:
        ValueError: If :attr:`n` is not a positive integer that is a power of 2.

    Returns:
        :class:`torch.Tensor`: Matrix :math:`A` with shape :math:`(n^2,n^2)`.
    """
    H1d = walsh_matrix(n)
    return torch.kron(H1d, H1d)


def walsh2_torch(img, H=None):
    # r"""Deprecated function. Use `fwht_2d` instead."""
    # raise NotImplementedError("This function is deprecated. Use `fwht_2d` instead.")
    r"""Return 2D Walsh-ordered Hadamard transform of an image

    This applies the 1D transform :math:`H \in \mathbb{R}^{n \times n}` to the rows and to the columns of batches of images :math:`X\in \mathbb{R}^{n \times n}`

    .. math::

        Y = H X H^T.

    Args:
        :attr:`img` (:class:`torch.tensor`): Batch of images :math:`X` with shape :math:`(*,n,n)`.

        :attr:`H` (:class:`torch.tensor`, optional): 1D Walsh-ordered Hadamard matrix with shape :math:`(n,n)`.

    Returns:
        :class:`torch.tensor`: Transformed image :math:`Y` with shape :math:`(*, n, n)` where :math:`*` is the same number as for :attr:`img`.

    See Also:
        :func:`~spyrit.core.torch.fwht_2d` implements the same transform with a different algorithm.

    Example:
        Example 1: Basic example

        >>> img = torch.randn(256, 1, 64, 64)
        >>> had = walsh2_torch(img)

        Example 2: Same on CPU

        >>> img = torch.randn(256, 1, 64, 64)
        >>> img = img.to(device='cpu')
        >>> had = walsh2_torch(img)
        >>> print(had.device)
        cpu

        Example 3: On GPU using :class:`torch.float64`

        >>> img = torch.randn(256, 1, 64, 64)
        >>> img = img.to(device='cpu', dtype=torch.float64)
        >>> had = walsh2_torch(img)
        >>> print(had.device,'+',had.dtype)
        cpu + torch.float64

    """
    if H is None:
        H = walsh_matrix(img.shape[-1])

    H = H.to(device=img.device, dtype=img.dtype)  # move in if?

    return mult_2d_separable(H, img)


def mult_1d(H: torch.tensor, x: torch.tensor, dim: int = -1) -> torch.tensor:
    r"""Multiply a matrix to batches of (1D) vectors.

    This computes matrix-vector products to a batch of vectors :math:`x`.

    Args:
        H (torch.tensor): Matrix with shape :math:`(a,b)`. The matrix :math:`H` multiplies to one of the dimensions of the batch of vectors.

        x (torch.tensor): Batch of vectors. The :attr:`dim`-th dimension of the tensor
        must have length :math:`b`.

        dim (int, optional): The dimension along which multiplication applies.
        Default is -1.

    Returns:
        torch.tensor: Transformed tensor. Has the same shape as the input tensor
        except for the :attr:`dim`-th dimension which has :math:`a` elements.
    """
    if dim != -1 and dim != x.ndim - 1:
        x = torch.moveaxis(x, dim, -1)
    x = torch.einsum("ij,...j->...i", H, x)
    if dim != -1 and dim != x.ndim - 1:
        x = torch.moveaxis(x, -1, dim)
    return x


def mult_2d_separable(H: torch.tensor, x: torch.tensor) -> torch.tensor:
    r"""Applies separable transform to batches of (2D) images.

    This applies the same transform :math:`H` to the rows and columns of a batch of images :math:`X`

    .. math::

        Y = H X H^T.

    Args:
        H (:class:`torch.tensor`): Matrix :math:`H` with shape :math:`(a, b)`.

        x (:class:`torch.tensor`): Input tensor to transform with shape
        :math:`(*, b, b)` where :math:`*` represents any number of batch dimensions.

    Returns:
        :class:`torch.tensor`: Transformed image :math:`Y` with shape :math:`(*, a, a)` where :math:`*` is the same number of batch dimensions as the input tensor.
    """
    x = H @ x @ H.T
    return x


def fwht(x, order=True, dim=-1):
    r"""Fast Walsh-Hadamard transform of x

    Args:
        x (torch.tensor): *-by-n input signal, where n is a power of two.

        order (bool or list, optional): True for sequency order (default), False
        for natural order. When order is a list, it defines the permutation
        indices to use. Default is True.

        dim (int, optional): The dimension along which to apply the transform.
        Default is -1.

    Returns:
        torch.tensor: *-by-n transformed signal

    Example:
        Example 1: Fast sequency-ordered (i.e., Walsh) Hadamard transform

        >>> import torch
        >>> import spyrit.core.torch as st
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1, -2])
        >>> x = x[None,:]
        >>> y = st.fwht(x)
        >>> print(y)
        tensor([[14, -8, -8, 18, -4, -2, -6,  4]])

        Example 2: Fast Hadamard transform

        >>> import torch
        >>> import spyrit.core.torch as st
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1, -2])
        >>> x = x[None,:]
        >>> y = st.fwht(x, False)
        >>> print(y)
        tensor([[14,  4, 18, -4, -8, -6, -8, -2]])

        Example 3: Permuted fast Hadamard transform

        >>> import numpy as np
        >>> import torch
        >>> import spyrit.core.torch as st
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1, -2])
        >>> ind = [1, 0, 3, 2, 7, 4, 5, 6]
        >>> y = st.fwht(x, ind)
        >>> print(y)
        tensor([ 4, 14, -4, 18, -2, -8, -6, -8])

        Example 4: Comparison with the numpy transform

        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch
        >>> import spyrit.core.torch as st
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1, -2])
        >>> y_np = wh.fwht(x)
        >>> x_torch = torch.from_numpy(x).to(torch.device('cpu'))
        >>> y_torch = st.fwht(x_torch)
        >>> print(y_np)
        [14 -8 -8 18 -4 -2 -6  4]
        >>> print(y_torch)
        tensor([14, -8, -8, 18, -4, -2, -6,  4]...)

        Example 5: Computation times for a signal of length 2**12

        >>> import timeit
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch
        >>> import spyrit.core.torch as st
        >>> x = np.random.rand(2**12)
        >>> t = timeit.timeit(lambda: wh.fwht(x,False), number=2000)
        >>> print(f"Fast Hadamard transform numpy CPU (2000x): {t:.4f} seconds")
        Fast Hadamard transform numpy CPU (2000x): ... seconds
        >>> x_torch = torch.from_numpy(x)
        >>> t = timeit.timeit(lambda: st.fwht(x_torch,False), number=2000)
        >>> print(f"Fast Hadamard transform pytorch CPU (2000x): {t:.4f} seconds")
        Fast Hadamard transform pytorch CPU (2000x): ... seconds

        Example 6: CPU vs GPU: Computation times for 512 signals of length 2**12

        >>> import timeit
        >>> import torch
        >>> import spyrit.core.torch as st
        >>> x_cpu = torch.rand(512,2**12)
        >>> t = timeit.timeit(lambda: st.fwht(x_cpu,False), number=50)
        >>> print(f"Fast Hadamard transform pytorch CPU (50x): {t:.4f} seconds")
        Fast Hadamard transform pytorch CPU (50x): ... seconds

        Example 7: Repeating the Walsh-ordered transform using input indices is faster

        >>> import timeit
        >>> import torch
        >>> import spyrit.core.torch as st
        >>> x = torch.rand(256,2**12).to(torch.device('cpu'))
        >>> t = timeit.timeit(lambda: st.fwht(x), number=100)
        >>> print(f"No indices as inputs (100x): {t:.3f} seconds")
        No indices as inputs (100x): ... seconds
        >>> ind = st.sequency_perm(x).shape[-1]
        >>> t = timeit.timeit(lambda: st.fwht(x,ind), number=100)
        >>> print(f"With indices as inputs (100x): {t:.3f} seconds")
        With indices as inputs (100x): ... seconds
    """
    if dim != -1 and dim != x.ndim - 1:
        x = torch.moveaxis(x, dim, -1)
    original_shape = x.shape

    # create batch if x is 1D
    if len(original_shape) == 1:
        x = x.reshape(1, -1)  # shape (1, n)

    *batch, d = x.shape  # batch is tuple and d is int
    assert_power_of_2(d, raise_error=True)

    h = 2
    # put the "if" statement here to avoid repeating "if"s in the loop
    if order == True:
        while h <= d:
            x = x.reshape(*batch, d // h, h)
            half1, half2 = torch.split(x, h // 2, dim=-1)
            half2[..., 1::2] *= -1  # not from Amit Portnoy
            x = torch.stack((half1 + half2, half1 - half2), dim=-1)  # not from AP
            h *= 2

    else:
        while h <= d:
            x = x.reshape(*batch, d // h, h)
            half1, half2 = torch.split(x, h // 2, dim=-1)
            x = torch.cat((half1 + half2, half1 - half2), axis=-1)
            h *= 2

    x = x.reshape(original_shape)
    # ---------------------------------------
    # END OF ADAPTED CODE FROM AMIT PORTNOY

    # Arbitrary order
    if type(order) == list:
        x = sequency_perm(x, order)

    if dim != -1 and dim != x.ndim - 1:
        x = torch.moveaxis(x, -1, dim)

    return x


def ifwht(x, order=True, dim=-1):
    r"""Inverse fast Walsh-Hadamard transform of x

    Args:
        x (torch.tensor): *-by-n input signal, where n is a power of two.

        order (bool, optional): True for sequency (default), False for natural.
        If a list, it defines the permutation indices to use. Default is True.

        dim (int, optional): The dimension along which to apply the transform.
        Default is -1.

    Returns:
        torch.tensor: *-by-n transformed signal
    """
    if type(order) == list:
        raise NotImplementedError(
            "Inverse transform not implemented yet for arbitrary order"
        )
    return fwht(x, order, dim) / x.shape[dim]


def fwht_2d(x, order=True):
    r"""Returns the fast Walsh-Hadamard transform of a 2D tensor.

    This function uses the fast Walsh-Hadamard transform for 1D signals. It is
    optimized for the natural order (with `order = False`) and the sequency
    order (with `order = True`). The fast Walsh-Hadamard transform is applied
    along the last two dimensions of the input tensor.

    Args:
        x (torch.tensor): Batch of 2D tensors to transform. The last two
        dimensions must be a power of two. Has shape :math:`(*, h, w)` where
        :math:`h` and :math:`w` are the height and width of the image, and *
        represents any number of batch dimensions.

        order (bool or list, optional): If a bool, defines if the sequency
        order is used (`True`) or the natural order is used (`False`). If a
        list, it defines the permutation indices to use. Default is `True`.

    Raises:
        ValueError: If either of the last two dimensions of the input tensor is
        not a power of two.

    Returns:
        torch.tensor: 2D Walsh-Hadamard transformed tensor.
    """
    return fwht(fwht(x, order, dim=-1), order, dim=-2)


def ifwht_2d(x, order=True):
    r"""Returns the inverse fast Walsh-Hadamard transform of a 2D tensor.

    This function uses the inverse fast Walsh-Hadamard transform for 1D signals.
    It is optimized for the natural order (with `order = False`) and the sequency
    order (with `order = True`). In case a list is provided in :attr:`order`,
    it performs a permutation using the indices provided in the list. The inverse
    fast Walsh-Hadamard transform is applied along the last two dimensions of the
    input tensor.

    Args:
        x (torch.tensor): input tensor to transform. Must have shape
        :math:`(*, h, w)` where :math:`h` and :math:`w` are the height and width
        of the image and should be powers of two. :math:`*` represents zero or
        more batch dimensions.

        order (bool or list, optional): Whether to use the sequency/Walsh ordering
        (True) or the natural ordering (False). If a list, it defines the permutation
        indices to use. Default is True.

    Raises:
        ValueError: If either of the last two dimensions of the input tensor is
        not a power of two.

    Returns:
        torch.tensor: 2D inverse Walsh-Hadamard transformed tensor. Has the same
        shape as the input tensor.
    """
    return ifwht(ifwht(x, order, dim=-1), order, dim=-2)


def meas2img(meas: torch.tensor, Ord: torch.tensor) -> torch.tensor:
    r"""Returns measurement image from a single measurement tensor or from a
    batch of measurement tensors.

    This function is particulatly useful if the
    number of measurements is less than the number of pixels in the image, i.e.
    the image is undersampled.

    Args:
        meas : `torch.tensor` with shape :math:`(*, M)` where
        :math:`*` is any dimension (e.g. the batch size, channel, etc) and
        :math:`M` is the length of the measurement vector.

        Ord : `torch.tensor` with shape :math:`(N,N)`. Sampling order matrix, where
        high values indicate high significance. This matrix determines the order
        of the measurements. It must be the matrix used when generating the measurement vector.

    Returns:
        Img : `torch.tensor` with shape :math:`(*, N,N)`. batch of N-by-N
        measurement images.
    """
    out_shape = *meas.shape[:-1], Ord.numel()
    meas_padded = torch.zeros(out_shape, device=meas.device)
    meas_padded[..., : meas.shape[-1]] = meas
    Img = sort_by_significance(meas_padded, Ord, axis="cols", inverse_permutation=False)
    return Img.reshape(*meas.shape[:-1], *Ord.shape)


# =============================================================================
# Finite difference matrices
# =============================================================================


def spdiags(diagonals, offsets, shape):
    """
    Similar to torch.sparse.spdiags. Arguments are the same, excepted :
        - diagonals is a list of 1D tensors (does not need to be a tensor)
        - offsets is a list of integers (does not need to be a tensor)
        - shape is unchanged (a tuple)

    Most notably:
        - Using a positive offset, the first element of the matrix diagonal
        is the first element of the provided diagonal. torch.sparse.spdiags
        introduces an offset of k when using a positive offset k.
    """
    # if offset > 0, roll to keep first element in 'dia' displayed
    diags = torch.stack(
        [dia.roll(off) if off > 0 else dia for dia, off in zip(diagonals, offsets)]
    )
    offsets = torch.tensor(offsets)
    return torch.sparse.spdiags(diags, offsets, shape)


def finite_diff_mat(n, boundary="dirichlet"):
    r"""
    Creates a finite difference matrix of shape :math:`(n^2,n^2)` for a 2D
    image of shape :math:`(n,n)`.

    Args:
        :attr:`n` (int): The size of the image.

        :attr:`boundary` (str, optional): The boundary condition to use.
        Must be one of 'dirichlet', 'neumann', 'periodic', 'symmetric' or
        'antisymmetric'. Default is 'neumann'.

    Returns:
        :class:`torch.sparse.FloatTensor`: The finite difference matrix.
    """

    # nombre de blocs: height
    # taille de chaque bloc: width

    # max number of elements in the diagonal
    # height, width = shape
    N = n**2
    # here are all the possible matrices. Please add to this list if you
    # want to add a new boundary condition
    valid_boundaries = [
        "dirichlet",
        "neumann",
        "periodic",
        "symmetric",
        "antisymmetric",
    ]
    if boundary not in valid_boundaries:
        raise ValueError(
            "Invalid boundary condition. Must be one of {}.".format(valid_boundaries)
        )

    # create common diagonals
    ones = torch.ones(n, n).flatten()
    ones_0right = torch.ones(n, n)
    ones_0right[:, -1] = 0
    ones_0right = ones_0right.flatten()

    if boundary == "dirichlet":
        Dx = spdiags([ones, -ones_0right], [0, -1], (N, N))
        Dy = spdiags([ones, -ones], [0, -n], (N, N))

    elif boundary == "neumann":
        ones_0left = ones_0right.roll(1)
        ones_0top = ones_0left.reshape(n, n).T.flatten()
        Dx = spdiags([ones_0left, -ones_0right], [0, -1], (N, N))
        Dy = spdiags([ones_0top, -ones], [0, -n], (N, N))

    elif boundary == "periodic":
        zeros_1left = (1 - ones_0right).roll(1)
        Dx = spdiags([ones, -ones_0right, -zeros_1left], [0, -1, n - 1], (N, N))
        Dy = spdiags([ones, -ones, -ones], [0, -n, N - n], (N, N))

    elif boundary == "symmetric":
        zeros_1left = (1 - ones_0right).roll(1)
        zeros_1top = zeros_1left.reshape(n, n).T.flatten()
        Dx = spdiags([ones, -ones_0right, -zeros_1left], [0, -1, n - 1], (N, N))
        Dy = spdiags([ones, -ones, -zeros_1top], [0, -n, n], (N, N))

    elif boundary == "antisymmetric":
        zeros_1left = (1 - ones_0right).roll(1)
        zeros_1top = zeros_1left.reshape(n, n).T.flatten()
        Dx = spdiags([ones, -ones_0right, zeros_1left], [0, -1, 1], (N, N))
        Dy = spdiags([ones, -ones, zeros_1top], [0, -n, n], (N, N))

    return Dx, Dy


def neumann_boundary(img_shape):
    r"""
    Creates a finite difference matrix of shape :math:`(h*w,h*w)` for a 2D
    image of shape :math:`(h,w)`. The boundary condition used is Neumann.

    Args:
        :attr:`img_shape` (tuple): The size of the image :math:`(h,w)`.

    Returns:
        :class:`torch.tensor`: The finite difference matrix.

    .. note::
        This function returns the same matrix as :func:`finite_diff_mat` with
        the Neumann boundary condition. Internal implementation is different
        and allows to process rectangular images.
    """
    h, w = img_shape
    # create h blocks of wxw matrices
    max_ = max(h, w)

    # create diagonals
    ones = torch.ones(max_)
    ones[0] = 0
    m_ones = -torch.ones(max_)
    block_h = spdiags([ones[:h], m_ones[:h]], [0, -1], (h, h))
    block_w = spdiags([ones[:w], m_ones[:w]], [0, -1], (w, w))

    # create blocks using kronecker product
    Dx = torch.kron(torch.eye(h), block_w.to_dense())
    Dy = torch.kron(block_h.to_dense(), torch.eye(w))

    return Dx, Dy


# =============================================================================
# Permutations and Sorting
# =============================================================================


def Cov2Var(Cov: torch.tensor, out_shape=None):
    r"""
    Extracts Variance Matrix from Covariance Matrix.

    The Variance matrix is extracted from the diagonal of the Covariance matrix.

    Args:
        Cov (torch.tensor): Covariance matrix of shape :math:`(N_x, N_x)`.

        out_shape (tuple, optional): Shape of the output variance matrix. If
        `None`, :math:`N_x` must be a perfect square and the output is a square
        matrix whose shape is :math:`(\sqrt{N_x}, \sqrt{N_x})`. Default is `None`.

    Raises:
        ValueError: If the input matrix is not square.

        ValueError: If the output shape is not valid.

    Returns:
        torch.tensor: Variance matrix of shape :math:`(\sqrt{N_x}, \sqrt{N_x})` or
        :math:`out_shape` if provided.
    """
    row, col = Cov.shape
    # check Cov is square
    if row != col:
        raise ValueError("Covariance matrix must be a square matrix")

    if out_shape is None:
        out_shape = (int(math.sqrt(row)), int(math.sqrt(col)))

    if out_shape[0] * out_shape[1] != row:
        raise ValueError(
            f"Invalid output shape, got {out_shape} with "
            + f"{out_shape[0]}*{out_shape[1]} != {row}"
        )
    # copy is necessary (see np documentation about diagonal)
    return torch.diagonal(Cov).clone().reshape(out_shape)


def reindex(  # previously sort_by_indices
    values: torch.tensor,
    indices: torch.tensor,
    axis: str = "rows",
    inverse_permutation: bool = False,
) -> torch.tensor:
    """Sorts a tensor along a specified axis using the indices tensor.

    The indices tensor contains the new indices of the elements in the values tensor. :attr:`values[0]` will be placed at the index :attr:`indices[0]` :attr:`values[1]` at :attr:`indices[1]`, and so on.

    Using the inverse permutation allows to revert the permutation: in this
    case, it is the element at index `indices[0]` that will be placed at the
    index `0`, the element at index `indices[1]` that will be placed at the
    index `1`, and so on.

    Args:
        values (torch.tensor): The tensor to sort. Can be 1D, 2D, or any
        multi-dimensional batch of 2D tensors.

        indices (torch.tensor): Tensor containing the new indices of the
        elements contained in `values`.

        axis (str, optional): The axis to sort along. Must be either 'rows' or
        'cols'. If `values` is 1D, `axis` is not used. Default is 'rows'.

        inverse_permutation (bool, optional): Whether to apply the permutation
        inverse. Default is False.

    Raises:
        ValueError: If `axis` is not 'rows' or 'cols'.

    Returns:
        torch.tensor: The sorted tensor by the given indices along the
        specified axis.

    Example:
        >>> values = torch.tensor([[10, 20, 30], [100, 200, 300]])
        >>> indices = torch.tensor([2, 0, 1])
        >>> out = reindex(values, indices, axis="cols", inverse_permutation=False)
        >>> out
        tensor([[ 20,  30,  10],
                [200, 300, 100]])
        >>> reindex(out, indices, axis="cols", inverse_permutation=True)
        tensor([[ 10,  20,  30],
                [100, 200, 300]])
    """
    reindices = indices.argsort()

    # cols corresponds to last dimension
    if axis == "cols" or values.ndim == 1:
        if inverse_permutation:
            return values[..., reindices.argsort()]
        return values[..., reindices]

    # rows corresponds to second-to-last dimension
    # because it is equivalent to sorting along the last dimension of the
    # transposed tensor, we need to transpose (inverse) the permutation
    elif axis == "rows":
        inverse_permutation = not inverse_permutation
        if inverse_permutation:
            return values[..., reindices.argsort(), :]
        return values[..., reindices, :]
    else:
        raise ValueError("Invalid axis. Must be 'rows' or 'cols'.")


def sort_by_significance(
    values: torch.tensor,
    sig: torch.tensor,
    axis: str = "rows",
    inverse_permutation: bool = False,
    get_indices: bool = False,
) -> torch.tensor:
    """Returns a tensor sorted by decreasing significance of its elements as
    determined by the significance tensor.

    The element in the `values` tensor whose significance is the highest will
    be placed first, followed by the element with the second highest
    significance, and so on. The significance tensor `sig` must have the same
    shape as `values` along the specified axis.

    This function is equivalent to (but much faster than) the following code:

    .. code-block:: python

        from spyrit.core.torch import Permutation_Matrix

        h = 64
        values = torch.randn(2*h, h)
        sig_rows = torch.randn(2*h)
        sig_cols = torch.randn(h)

        # 1
        y1 = sort_by_significance(values, sig_rows, 'rows', False)
        y2 = Permutation_Matrix(sig_rows) @ values
        assert torch.allclose(y1, y2) # True

        # 2
        y1 = sort_by_significance(values, sig_rows, 'rows', True)
        y2 = Permutation_Matrix(sig_rows).T @ values
        assert torch.allclose(y1, y2) # True

        # 3
        y1 = sort_by_significance(values, sig_cols, 'cols', False)
        y2 = values @ Permutation_Matrix(sig_cols)
        assert torch.allclose(y1, y2) # True

        # 4
        y1 = sort_by_significance(values, sig_cols, 'cols', True)
        y2 = values @ Permutation_Matrix(sig_cols).T
        assert torch.allclose(y1, y2) # True

    Args:
        values (torch.tensor): Tensor to sort by significance. Can be 1D, 2D,
        or any multi-dimensional batch of 2D tensors.

        sig (torch.tensor): Significance tensor. Its length must be equal to
        the number of rows or columns in `values` depending on the specified
        axis.

        axis (str, optional): Axis along which to sort. Must be either 'rows'
        or 'cols'. Default is 'rows'.

        inverse_permutation (bool, optional): If True, the inverse permutation
        is applied. Default is False.

        get_indices (bool, optional): If True, the function will return the
        indices tensor used to sort the values tensor. Default is False.

    Returns:
        torch.tensor or 2-tuple of torch.tensors: Tensor ordered by decreasing
        significance along the specified axis. If `get_indices` is True, the
        function will return a tuple containing the ordered tensor and the
        indices tensor used to sort the values tensor.
    """
    indices = torch.argsort(-sig.flatten(), stable=True).to(torch.int32)
    if get_indices:
        return reindex(values, indices, axis, inverse_permutation), indices
    return reindex(values, indices, axis, inverse_permutation)


def Permutation_Matrix(sig: torch.tensor) -> torch.tensor:
    """Returns a permutation matrix based on the significance tensor.

    The permutation matrix is a square matrix whose rows or columns are
    permuted based on the significance tensor. The permutation matrix is
    used to sort a tensor by decreasing significance of its elements.

    Args:
        sig (torch.tensor): Significance tensor. Its length must be equal to
        the number of rows or columns in the tensor to be sorted. If it is not
        a 1D tensor, it is flattened.

    Returns:
        torch.tensor: Permutation matrix of shape `(n, n)` based on the
        significance tensor, where `n` is the length of the significance
        tensor.

    Example:
        >>> sig = torch.tensor([0.1, 0.4, 0.2, 0.3])
        >>> Permutation_Matrix(sig)
        tensor([[0., 1., 0., 0.],
                [0., 0., 0., 1.],
                [0., 0., 1., 0.],
                [1., 0., 0., 0.]])
    """
    indices = torch.argsort(-sig.reshape(-1), stable=True)
    return torch.eye(len(sig.reshape(-1)), device=sig.device)[indices]


# =============================================================================
# Image Processing
# =============================================================================


def center_crop(
    img: torch.tensor,
    out_shape: tuple,
    vectorized_in_shape: tuple = None,
) -> torch.tensor:
    """Crops the center of an image to the specified shape.

    This function uses the `torchvision.transforms.CenterCrop` class to crop
    the center of an image to the specified shape. This function can however
    crop images that are vectorized (flattened, 1D) by specifying the input
    shape.

    Args:
        img (torch.tensor): Image to crop. If the image is vectorized, the
        input shape must be specified.

        out_shape (tuple): Shape of the output image after cropping. Must be
        a tuple of two integers (height, width).

        vectorized_in_shape (tuple, optional): Shape of the input image, must be specified
        if and only if the input image is vectorized. Must be a tuple of two
        integers (height, width). If None, the input is supposed to be a 2D
        image. Defaults to None.

    Returns:
        torch.tensor: Cropped image. It has the same number of dimensions as
        the input image.
    """
    # if img has shape (..., h*w), reshape it to (..., h, w)
    img_shape = img.shape
    if vectorized_in_shape is not None:
        img = img.reshape(*img_shape[:-1], *vectorized_in_shape)
    img_cropped = torchvision.transforms.CenterCrop(out_shape)(img)
    if vectorized_in_shape is not None:
        img_cropped = img_cropped.reshape(*img_shape[:-1], -1)
    return img_cropped


def center_pad(
    img: torch.tensor,
    out_shape: tuple,
    vectorized_in_shape: tuple = None,
) -> torch.tensor:
    """Pads an image to the specified shape by centering it.

    Args:
        img (torch.tensor): Image to pad. If the image is vectorized, the
        input shape must be specified.

        out_shape (tuple): Shape of the output image after padding. Must be
        a tuple of two integers (height, width).

        vectorized_in_shape (tuple, optional): Shape of the input image, must be specified
        if and only if the input image is vectorized. Must be a tuple of two
        integers (height, width). If None, the input is supposed to be a 2D
        image. Defaults to None.

    Returns:
        torch.tensor: Padded image. It has the same number of dimensions as
        the input image.
    """
    img_shape = img.shape
    if vectorized_in_shape is None:
        vectorized_in_shape = img_shape[-2:]
        reshape = False
    else:
        img = img.reshape(*img_shape[:-1], *vectorized_in_shape)
        reshape = True

    pad_top = (out_shape[0] - vectorized_in_shape[0]) // 2
    pad_bottom = out_shape[0] - vectorized_in_shape[0] - pad_top
    pad_left = (out_shape[1] - vectorized_in_shape[1]) // 2
    pad_right = out_shape[1] - vectorized_in_shape[1] - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    img_padded = nn.ConstantPad2d(padding, 0)(img)

    if reshape:
        img_padded = img_padded.reshape(*img_shape[:-1], -1)
    return img_padded


# =============================================================================
# Linear Algebra
# =============================================================================


def regularized_pinv(A: torch.tensor, regularization: str, **kwargs) -> torch.tensor:
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
        A (torch.tensor): input 2D matrix to compute the pseudo-inverse. Must
        be 2D.

        regularization (str): Regularization method to use. Supported methods
        are "rcond", "L2", and "H1".

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
        pinv = torch.linalg.pinv(A, **kwargs)

    elif regularization == "L2":
        eta = kwargs.get("eta")
        pinv = (
            torch.linalg.inv(A.T @ A + eta * torch.eye(A.shape[1], device=A.device))
            @ A.T
        )

    elif regularization == "H1":
        eta = kwargs.get("eta")
        img_shape = kwargs.get("img_shape")
        Dx, Dy = neumann_boundary(img_shape)
        D2 = (Dx.T @ Dx + Dy.T @ Dy).to(A.device)
        pinv = torch.linalg.inv(A.T @ A + eta * D2) @ A.T

    else:
        raise NotImplementedError(
            f"Regularization method {regularization} not implemented. Currently supported methods are 'rcond', 'L2', and 'H1'."
        )

    return pinv


def regularized_lstsq(A: torch.tensor, y: torch.tensor, regularization: str, **kwargs):
    """Batched regularized least squares solution of a system of equations.

    It solves the linear system of equations :math:`Ax = y` using a regularized
    least squares method. The regularizations supported are:

        - "rcond": Uses the function :func:`torch.linalg.lstsq`. Additional
            arguments can be passed to this function through the `kwargs`
            parameters, such as `rcond` or `driver`. They are given to the
            function :func:`torch.linalg.lstsq`.

        - "L2": Uses the L2 regularization method. The regularization parameter
            `eta` must be passed as a keyword argument. It controls the amount
            of regularization applied to the least squares solution.

        - "H1": Uses the H1 regularization method. The regularization parameter
            `eta` must be passed as a keyword argument. It controls the amount
            of regularization applied to the least squares solution. This method
            is only implemented for 2D images.

    .. note:
        To speed up computation, you may provide the value of the finite
        difference matrices `D2` as a keyword argument. If not provided, the
        function will compute them using the keyword-provided image shape.

    Args:
        A (torch.tensor): Left-hand side tensor of shape :math:`(m, n)`, where
        * is any number of batch dimensions.

        y (torch.tensor): Right-hand side tensor of shape :math:`(*, m)`, where
        * is any number of batch dimensions.

        regularization (str): Regularization method to use. Supported methods
        are "rcond", "L2", and "H1".

        **kwargs: Additional keyword arguments to pass to the regularization
        method. Must include the regularization parameter `eta` when using
        the "L2" and "H1" regularization methods. Other keyword arguments
        include `rcond` and `driver` for the "rcond" method, as well as 'D2'
        (the finite difference matrices) for the "H1" and "L2" methods.

    Returns:
        torch.tensor: The regularized least squares solution of shape
        :math:`(*, n)`.
    """
    m, n = A.shape
    batches = y.shape[:-1]

    if regularization == "rcond":
        lhs = A.expand(*batches, m, n)
        rhs = y.unsqueeze(-1)
        x = torch.linalg.lstsq(lhs, rhs, **kwargs).solution
        x = x.squeeze(-1)

    elif regularization == "L2":
        eta = kwargs.get("eta")
        D2 = kwargs.get("D2", eta * torch.eye(A.shape[1], device=A.device))
        lhs = (A.T @ A + eta * D2).expand(*batches, m, n)
        rhs = torch.matmul(y, A)
        x = torch.linalg.solve(lhs, rhs)

    elif regularization == "H1":
        eta = kwargs.get("eta")
        img_shape = kwargs.get("img_shape")
        D2 = kwargs.get("D2", None)
        if D2 is None:
            Dx, Dy = neumann_boundary(img_shape)
            D2 = (Dx.T @ Dx + Dy.T @ Dy).to(A.device)
        lhs = (A.T @ A + eta * D2).expand(*batches, m, n)
        rhs = torch.matmul(y, A)
        x = torch.linalg.solve(lhs, rhs)

    else:
        raise NotImplementedError(
            f"Regularization method {regularization} not implemented. Currently supported methods are 'rcond', 'L2', and 'H1'."
        )

    return x


# =============================================================================
# Dynamic Handling
# =============================================================================


# def H_dyn_no_warping(
#     H: torch.tensor,
#     deformation_field: warp.DeformationField,
#     mode: str = "bilinear",
#     warping: bool = False,
# ) -> torch.tensor:
#     pass


# def H_dyn_warping(
#     H: torch.tensor,
#     deformation_field: warp.DeformationField,
#     mode: str = "bilinear",
# ) -> torch.tensor:
#     r""" """

#     det = deformation_field.det()

#     meas_pattern = meas_pattern.reshape(
#         meas_pattern.shape[0], 1, self.meas_shape[0], self.meas_shape[1]
#     )
#     meas_pattern_ext = torch.zeros(
#         (meas_pattern.shape[0], 1, self.img_shape[0], self.img_shape[1])
#     )
#     amp_max_h = (self.img_shape[0] - self.meas_shape[0]) // 2
#     amp_max_w = (self.img_shape[1] - self.meas_shape[1]) // 2
#     meas_pattern_ext[
#         :,
#         :,
#         amp_max_h : self.meas_shape[0] + amp_max_h,
#         amp_max_w : self.meas_shape[1] + amp_max_w,
#     ] = meas_pattern
#     meas_pattern_ext = meas_pattern_ext.to(dtype=motion.field.dtype)

#     H_dyn = nn.functional.grid_sample(
#         meas_pattern_ext,
#         motion.field,
#         mode=mode,
#         padding_mode="zeros",
#         align_corners=True,
#     )
#     H_dyn = det.reshape((meas_pattern.shape[0], -1)) * H_dyn.reshape(
#         (meas_pattern.shape[0], -1)
#     )

#     self._param_H_dyn = nn.Parameter(H_dyn, requires_grad=False).to(self.device)
