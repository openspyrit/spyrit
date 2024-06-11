"""
Contains pytorch-based functions used in spyrit.core modules.

The goal of this module is to provide a set of functions that use various
pytorch functionalities and optimizations to perform the necessary operations
in the spyrit.core modules. It mirrors the the spyrit.misc most used
functions, but using pytorch tensors instead of numpy arrays.
"""

# import warnings

import torch
import torch.nn as nn
import torchvision


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


def walsh_matrix(n):
    r"""Returns a 1D Walsh-ordered Hadamard transform matrix of size
    :math:`n \times n`.

    Args:
        n (int): Order of the Hadamard matrix. Must be a power of two.

    Raises:
        ValueError: If n is not a positive integer or if n is not a power of 2.

    Returns:
        torch.tensor: A n-by-n matrix representing the Walsh-ordered Hadamard
        matrix.
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


def walsh2_matrix(n):
    r"""Returns Walsh-ordered Hadamard matrix in 2D.

    Args:
        n (int): Order of the matrix, which must be a power of two.

    Raises:
        ValueError: If n is not a positive integer or if n is not a power of 2.

    Returns:
        torch.tensor: A n*n-by-n*n matrix representing the 2D Walsh-ordered
        Hadamard matrix.
    """
    H1d = walsh_matrix(n)
    return torch.kron(H1d, H1d)


def walsh2_torch(img, H=None):
    r"""Returns a 2D Walsh-ordered Hadamard transform of an image.

    Args:
        img (torch.tensor): Image to transform. Must have a shape of
        :math:`(*, h, w)` where :math:`h` and :math:`w` are the image height
        and width. The image must be square, i.e. :math:`h = w`. The image
        height and width must be a power of two.

        H (torch.tensor, optional): 1D Walsh-ordered Hadamard transformation
        matrix. Specify this if you have already calculated the transformation
        matrix, or leave this field to `None` to have the transformation matrix
        calculated. Defaults to None.

    Returns:
        torch.tensor: Hadamard-transformed image. Has same shape as input
        image.

    Example:
        >>> img = torch.rand(1, 3, 64, 64) # (batch, channels, height, width)
        >>> img_transformed = walsh2_torch(img)
        >>> img_transformed.shape
        torch.Size([1, 3, 64, 64])
    """
    if H is None:
        H = walsh_matrix(img.shape[-1])
    H = H.to(img.device)
    return H @ img @ H


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


def reindex(  # previously sort_by_indices
    values: torch.tensor,
    indices: torch.tensor,
    axis: str = "rows",
    inverse_permutation: bool = False,
) -> torch.tensor:
    """Sorts a tensor along a specified axis using the indices tensor.

    The indices tensor contains the new indices of the elements in the values
    tensor. `values[0]` will be placed at the index `indices[0]`, `values[1]`
    at `indices[1]`, and so on.

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
        >>> out = reindex(values, indices, axis="cols", False)
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

    This function is equivalent to (but much faster than) the following code::

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
    indices = torch.argsort(-sig.view(-1), stable=True)
    return torch.eye(len(sig.view(-1)), device=sig.device)[indices]


# =============================================================================
# Image Processing
# =============================================================================


def center_crop(
    img: torch.tensor,
    out_shape: tuple,
    in_shape: tuple = None,
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

        in_shape (tuple, optional): Shape of the input image, must be specified
        if and only if the input image is vectorized. Must be a tuple of two
        integers (height, width). If None, the input is supposed to be a 2D
        image. Defaults to None.

    Returns:
        torch.tensor: Cropped image. It has the same number of dimensions as
        the input image.
    """
    # if img has shape (..., h*w), reshape it to (..., h, w)
    img_shape = img.shape
    if in_shape is not None:
        img = img.reshape(*img_shape[:-1], *in_shape)
    img_cropped = torchvision.transforms.CenterCrop(out_shape)(img)
    if in_shape is not None:
        img_cropped = img_cropped.reshape(*img_shape[:-1], -1)
    return img_cropped


def center_pad(
    img: torch.tensor,
    out_shape: tuple,
    in_shape: tuple = None,
) -> torch.tensor:
    """Pads an image to the specified shape by centering it.

    Args:
        img (torch.tensor): Image to pad. If the image is vectorized, the
        input shape must be specified.

        out_shape (tuple): Shape of the output image after padding. Must be
        a tuple of two integers (height, width).

        in_shape (tuple, optional): Shape of the input image, must be specified
        if and only if the input image is vectorized. Must be a tuple of two
        integers (height, width). If None, the input is supposed to be a 2D
        image. Defaults to None.

    Returns:
        torch.tensor: Padded image. It has the same number of dimensions as
        the input image.
    """
    img_shape = img.shape
    if in_shape is None:
        in_shape = img_shape[-2:]
        reshape = False
    else:
        img = img.reshape(*img_shape[:-1], *in_shape)
        reshape = True

    pad_top = (out_shape[0] - in_shape[0]) // 2
    pad_bottom = out_shape[0] - in_shape[0] - pad_top
    pad_left = (out_shape[1] - in_shape[1]) // 2
    pad_right = out_shape[1] - in_shape[1] - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    img_padded = nn.ConstantPad2d(padding, 0)(img)

    if reshape:
        img_padded = img_padded.reshape(*img_shape[:-1], -1)
    return img_padded
