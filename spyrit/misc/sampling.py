from typing import Union
import warnings

import torch
import numpy as np
from scipy.stats import rankdata


# from /misc/statistics.py
def img2mask(Mat: np.ndarray, M: int):
    """Returns sampling mask from sampling matrix.

    Args:
        Mat (np.ndarray):
            N-by-N sampling matrix, where high values indicate high significance.
        M (int):
            Number of measurements to be kept.

    Returns:
        Mask (np.ndarray):
            N-by-N sampling mask, where 1 indicates the measurements to sample
            and 0 that to discard.
    """
    (nx, ny) = Mat.shape
    Mask = np.ones((nx, ny))
    ranked_data = np.reshape(rankdata(-Mat, method="ordinal"), (nx, ny))
    Mask[np.absolute(ranked_data) > M] = 0
    return Mask


# from /former/_model_Had_DCAN.py
def meas2img(meas: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    r"""Returns measurement image from a single measurement vector or from a
    batch of measurement vectors. This function is particulatly useful if the
    number of measurements is less than the number of pixels in the image, i.e.
    the image is undersampled.

    Args:
        meas : `np.ndarray` with shape :math:`(M)` or :math:`(B, M)` where
        :math:`B` is the batch size and :math:`M` is the length of the
        measurement vector.

        Mat : `np.ndarray` with shape :math:`(N,N)`. Sampling matrix, where
        high values indicate high significance. It must be the matrix used to
        generate the measurement vector.

    Returns:
        Img : `np.ndarray` with shape :math:`(N,N,)`. N-by-N measurement image
    """
    # y = np.pad(meas, (0, Mat.size - len(meas)))
    # # Perm = Permutation_Matrix(Mat)
    # # Img = np.dot(np.transpose(Perm), y)  #.reshape(Mat.shape)
    # return Img.reshape(Mat.shape)

    # y = np.pad(meas, ((0, 0), (0, Mat.size - meas.shape[0]))[2-meas.ndim:])
    ndim = meas.ndim
    if ndim == 1:
        meas = meas.reshape(1, -1)
    # meas is of shape (B, M), B is batch size
    y_padded = np.zeros((meas.shape[0], Mat.size))
    y_padded[:, : meas.shape[1]] = meas
    Img = sort_by_significance(y_padded, Mat, axis="cols", inverse_permutation=False)
    return Img.reshape((-1, *Mat.shape)[2 - ndim :])


def meas2img2(meas: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    r"""Return multiple measurement images from multiple measurement vectors.
    It is essentially the same as `meas2img`, but the `meas` argument is
    two-dimensional.

    .. warning::
        This function is deprecated. Use `spyrit.misc.sampling.meas2img` instead.
        In meas2img, the batch dimension comes first: (B, M) instead of (M, B).

    Args:
        meas : `np.ndarray` with shape :math:`(M,B)`. Set of :math:`B`
        measurement vectors of length :math:`M \le N^2`.

        Mat : `np.ndarray` with shape :math:`(N,N)`. Sampling matrix, where
        high values indicate high significance.

    Returns:
        Img : `np.ndarray` with shape :math:`(N,N,B)`
            Set of :math:`B` images of shape :math:`(N,N)`
    """
    warnings.warn(
        "This function is deprecated. Use `spyrit.misc.sampling.meas2img` "
        + "instead. Beware the batch dimension has moved.",
        DeprecationWarning,
    )
    return meas2img(np.moveaxis(meas, 0, -1), Mat)
    # M, B = meas.shape
    # Nx, Ny = Mat.shape

    # y = np.pad(meas, ((0, Mat.size - len(meas)), (0, 0)))
    # # Perm = Permutation_Matrix(Mat)
    # # Img = Perm.T @ y
    # Img = sort_by_significance(y, Mat, axis="rows", inverse_permutation=True)
    # Img = Img.reshape((Nx, Ny, B))
    # return Img


def img2meas(Img: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    """Return measurement vector from measurement image (not TESTED)

    Args:
        Img (np.ndarray):
            N-by-N measurement image.
        Mat (np.ndarray):
            N-by-N sampling matrix, where high values indicate high significance.

    Returns:
        meas (np.ndarray):
            Measurement vector of lenth M <= N**2.
    """
    # Perm = Permutation_Matrix(Mat)
    # meas = np.dot(Perm, np.ravel(Img))
    meas = sort_by_significance(
        np.ravel(Img), Mat, axis="rows", inverse_permutation=False
    )
    return meas


def Permutation_Matrix(Mat: np.ndarray) -> np.ndarray:
    """
        Returns permutation matrix from sampling matrix

    Args:
        Mat (np.ndarray):
            N-by-N sampling matrix, where high values indicate high significance.

    Returns:
        P (np.ndarray): N^2-by-N^2 permutation matrix (boolean)

    .. note::
        Consider using :func:`sort_by_significance` for increased
        computational performance if using :func:`Permutation_Matrix` to
        reorder a matrix as follows:
        ``y = Permutation_Matrix(Ord) @ Mat``
    """
    indices = np.argsort(-Mat.flatten(), kind="stable")
    return np.eye(len(Mat.flatten()))[indices]
    # (nx, ny) = Mat.shape
    # Reorder = rankdata(-Mat, method="ordinal")
    # Columns = np.array(range(nx * ny))
    # P = np.zeros((nx * ny, nx * ny))
    # P[Reorder - 1, Columns] = 1
    # return P


def sort_by_significance(
    arr: np.ndarray,
    sig: np.ndarray,
    axis: str = "rows",
    inverse_permutation: bool = False,
    get_indices: bool = False,
) -> np.ndarray:
    """
    Returns an array ordered by decreasing significance along the specified
    dimension.

    The significance values are given in the :math:`sig` array. The type of
    the output is the same as the input array :math:`arr`.

    This function is equivalent to (but faster) :func:`Permutation_Matrix` and
    multiplying the input array by the permutation matrix. More specifically,
    here are the four possible different calls and their equivalent::

        h = 64
        arr = np.random.randn(h, h)
        sig = np.random.randn(h)

        # 1
        y = sort_by_significance(arr, sig, axis='rows', inverse_permutation=False)
        y = Permutation_Matrix(sig) @ arr

        # 2
        y = sort_by_significance(arr, sig, axis='rows', inverse_permutation=True)
        y = Permutation_Matrix(sig).T @ arr

        # 3
        y = sort_by_significance(arr, sig, axis='cols', inverse_permutation=False)
        y = arr @ Permutation_Matrix(sig)

        # 4
        y = sort_by_significance(arr, sig, axis='cols', inverse_permutation=True)
        y = arr @ Permutation_Matrix(sig).T

    .. note::
        :math:`arr` must have the same number of rows or columns as there are
        elements in the flattened :math:`sig` array.

    Args:
        arr (np.ndarray or torch.tensor): Array to be ordered by rows or columns.
        The output's type is the same as this parameter's type.

        sig (np.ndarray or torch.tensor): Array containing the significance values.

        axis (str, optional): Axis along which to order the array. Must be either 'rows' or
        'cols'. Defaults to 'rows'.

        inverse_permutation (bool, optional): If True, the permutation matrix is
        transposed before being used. This is equivalent to using the inverse
        permutation matrix. Defaults to False.

        get_indices (bool, optional): If True, the function returns the indices of
        the significance values in decreasing order. Defaults to False.

    Shape:
        - arr: :math:`(*, r, c)` or :math:`(c)`, where :math:`(*)` is any number of
        dimensions, and :math:`r` and :math:`c` are the number of rows and columns respectively.

        - sig: :math:`(r)` if axis is 'rows' or :math:`(c)` if axis is 'cols' (or any shape
        that has the same number of elements). Not used if arr is 1D.

        - Output: :math:`(*, r, c)` or :math:`(c)`

    Returns:
        Tuple of np.ndarray:

        - **Array** :math:`arr` ordered by decreasing significance :math:`sig`
            along its rows or columns.

        - **Indices** :math:`indices` of the significance values in decreasing
            order. This is useful if you want to reorder other arrays in the
            same way.
    """
    # compute indices in a stable way (otherwise quicksort messes up the order)
    indices = np.argsort(-sig.flatten(), kind="stable")
    if get_indices:
        return (reindex(arr, indices, axis, inverse_permutation), indices)
    return reindex(arr, indices, axis, inverse_permutation)


def reindex(
    values: np.ndarray,
    indices: np.ndarray,
    axis: str = "rows",
    inverse_permutation: bool = False,
) -> np.ndarray:
    """Sorts a tensor along a specified axis using the indices tensor.

    The indices tensor contains the new indices of the elements in the values
    tensor. `values[0]` will be placed at the index `indices[0]`, `values[1]`
    at `indices[1]`, and so on.

    Args:
        values (np.ndarray): Array to sort. Can be 1D, 2D, or any
        multi-dimensional batch of 2D arrays.

        indices (np.ndarray): Array containing the new indices
        of the elements contained in `values`.

        axis (str, optional): The axis to sort along. Must be either 'rows',
        'cols' or None. If None, `values` is flattened before sorting,
        and then reshaped to its original shape. If `values` is 1D, `axis` is
        not used. Default is 'rows'.

        inverse_permutation (bool, optional): Whether to apply the permutation
        inverse. Default is False.

    Raises:
        ValueError: If `axis` is not 'rows' or 'cols'.

    Returns:
        np.ndarray: Array ordered by the given indices along
        the specified axis. The type is the same as the input array `values`.

    Example:
        >>> values = np.array([[10, 20, 30], [100, 200, 300]])
        >>> indices =  np.array([2, 0, 1])
        >>> reindex(values, indices, axis="cols")
        array([[ 20,  30,  10],
               [200, 300, 100]])
    """
    reindices = indices.argsort()

    if axis == "flatten" or values.ndim == 1:
        out_shape = values.shape
        values = values.flatten()
        if inverse_permutation:
            return values[reindices.argsort()].reshape(out_shape)
        return values[reindices].reshape(out_shape)

    # cols corresponds to last dimension
    if axis == "cols":
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
        raise ValueError("Invalid axis. Must be 'rows', 'cols' or 'flatten'.")


def reorder(meas: np.ndarray, Perm_acq: np.ndarray, Perm_rec: np.ndarray) -> np.ndarray:
    r"""Reorder measurement vectors

    Args:
        meas (np.ndarray):
            Measurements with dimensions (:math:`M_{acq} \times K_{rep}`), where
            :math:`M_{acq}` is the number of acquired patterns and
            :math:`K_{rep}` is the number of acquisition repetitions
            (e.g., wavelength or image batch).

        Perm_acq (np.ndarray):
            Permutation matrix used for acquisition
            (:math:`N_{acq}^2 \times N_{acq}^2` square matrix).

        Perm_rec (np.ndarray):
            Permutation matrix used for reconstruction
            (:math:`N_{rec} \times N_{rec}` square matrix).

    Returns:
        (np.ndarray):
            Measurements with dimensions (:math:`M_{rec} \times K_{rep}`),
            where :math:`M_{rec} = N_{rec}^2`.

    .. note::
            If :math:`M_{rec} < M_{acq}`, the input measurement vectors are
            subsampled.

            If :math:`M_{rec} > M_{acq}`, the input measurement vectors are
            filled with zeros.
    """
    # Dimensions (N.B: images are assumed to be square)
    N_acq = int(Perm_acq.shape[0] ** 0.5)
    N_rec = int(Perm_rec.shape[0] ** 0.5)
    K_rep = meas.shape[1]

    # Acquisition order -> natural order (fill with zeros if necessary)
    if N_rec > N_acq:
        # Square subsampling in the "natural" order
        Ord_sub = np.zeros((N_rec, N_rec))
        Ord_sub[:N_acq, :N_acq] = -np.arange(-(N_acq**2), 0).reshape(N_acq, N_acq)
        Perm_sub = Permutation_Matrix(Ord_sub)

        # Natural order measurements (N_acq resolution)
        Perm_raw = np.zeros((2 * N_acq**2, 2 * N_acq**2))
        Perm_raw[::2, ::2] = Perm_acq.T
        Perm_raw[1::2, 1::2] = Perm_acq.T
        meas = Perm_raw @ meas

        # Zero filling (needed only when reconstruction resolution is higher
        # than acquisition res)
        zero_filled = np.zeros((2 * N_rec**2, K_rep))
        zero_filled[: 2 * N_acq**2, :] = meas

        meas = zero_filled

        Perm_raw = np.zeros((2 * N_rec**2, 2 * N_rec**2))
        Perm_raw[::2, ::2] = Perm_sub.T
        Perm_raw[1::2, 1::2] = Perm_sub.T

        meas = Perm_raw @ meas

    elif N_rec == N_acq:
        Perm_sub = Perm_acq[: N_rec**2, :].T

    elif N_rec < N_acq:
        # Square subsampling in the "natural" order
        Ord_sub = np.zeros((N_acq, N_acq))
        Ord_sub[:N_rec, :N_rec] = -np.arange(-(N_rec**2), 0).reshape(N_rec, N_rec)
        Perm_sub = Permutation_Matrix(Ord_sub)
        Perm_sub = Perm_sub[: N_rec**2, :]
        Perm_sub = Perm_sub @ Perm_acq.T

    # Reorder measurements when the reconstruction order is not "natural"
    if N_rec <= N_acq:
        # Get both positive and negative coefficients permutated
        Perm = Perm_rec @ Perm_sub
        Perm_raw = np.zeros((2 * N_rec**2, 2 * N_acq**2))

    elif N_rec > N_acq:
        Perm = Perm_rec
        Perm_raw = np.zeros((2 * N_rec**2, 2 * N_rec**2))

    Perm_raw[::2, ::2] = Perm
    Perm_raw[1::2, 1::2] = Perm
    meas = Perm_raw @ meas

    return meas
