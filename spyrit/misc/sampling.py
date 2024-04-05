from typing import Union

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
    """Return measurement image from a single measurement vector

    Args:
        meas : `np.ndarray` with shape :math:`(M,)`
            Measurement vector of length :math:`M \le N^2`.
        Mat : `np.ndarray` with shape :math:`(N,N)`
            Sampling matrix, where high values indicate high significance.

    Returns:
        Img : `np.ndarray` with shape :math:`(N,N,)`
            N-by-N measurement image
    """
    y = np.pad(meas, (0, Mat.size - len(meas)))
    # Perm = Permutation_Matrix(Mat)
    # Img = np.dot(np.transpose(Perm), y)  #.reshape(Mat.shape)
    Img = sort_by_significance(y, Mat, axis="rows", inverse_permutation=True)
    return Img.reshape(Mat.shape)


def meas2img2(meas: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    """Return multiple measurement images from multiple measurement vectors.
    It is essentially the same as `meas2img`, but the `meas` argument is
    two-dimensional.

    Args:
        meas : `np.ndarray` with shape :math:`(M,B)`
            Set of :math:`B` measurement vectors of lenth :math:`M \le N^2`.
        Mat : `np.ndarray` with shape :math:`(N,N)`
            Sampling matrix, where high values indicate high significance.

    Returns:
        Img : `np.ndarray` with shape :math:`(N,N,B)`
            Set of :math:`B` images of shape :math:`(N,N)`
    """
    M, B = meas.shape
    Nx, Ny = Mat.shape

    y = np.pad(meas, ((0, Mat.size - len(meas)), (0, 0)))
    # Perm = Permutation_Matrix(Mat)
    # Img = Perm.T @ y
    Img = sort_by_significance(y, Mat, axis="rows", inverse_permutation=True)
    Img = Img.reshape((Nx, Ny, B))
    return Img


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
    (nx, ny) = Mat.shape
    Reorder = rankdata(-Mat, method="ordinal")
    Columns = np.array(range(nx * ny))
    P = np.zeros((nx * ny, nx * ny))
    P[Reorder - 1, Columns] = 1
    return P


def sort_by_significance(
    arr: Union[np.ndarray, torch.tensor],
    sig: Union[np.ndarray, torch.tensor],
    axis: str = "rows",
    inverse_permutation: bool = False,
    get_indices: bool = False,
) -> Union[np.ndarray, torch.tensor]:
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
        arr (np.ndarray or torch.tensor):
            Array to be ordered by rows or columns. The output's type is
            the same as this parameter's type.

        sig (np.ndarray or torch.tensor):
            Array containing the significance values.

        axis (str, optional):
            Axis along which to order the array. Must be either 'rows' or
            'cols'. Defaults to 'rows'.

        inverse_permutation (bool, optional):
            If True, the permutation matrix is transposed before being used.
            This is equivalent to using the inverse permutation matrix.
            Defaults to False.

        get_indices (bool, optional):
            If True, the function returns the indices of the significance
            values in decreasing order. Defaults to False.

    Shape:
        - arr: :math:`(*, r, c)` or :math:`(c)`, where :math:`(*)` is any
        number of dimensions, and :math:`r` and :math:`c` are the number of
        rows and columns respectively.

        - sig: :math:`(r)` if axis is 'rows' or :math:`(c)` if axis is 'cols'
        (or any shape that has the same number of elements). Not used if
        arr is 1D.

        - Output: :math:`(*, r, c)` or :math:`(c)`

    Returns:
        Tuple of np.ndarray or torch.tensor:

        - **Array** :math:`arr` ordered by decreasing significance :math:`sig`
            along its rows or columns.

        - **Indices** :math:`indices` of the significance values in decreasing
            order. This is useful if you want to reorder other arrays in the
            same way.
    """
    # compute indices in a stable way (otherwise quicksort messes up the order)
    if isinstance(sig, torch.Tensor):
        indices = torch.argsort(-sig.flatten(), stable=True)
        if get_indices and isinstance(arr, np.ndarray):
            indices = indices.numpy()  # both indices and arr are numpy arrays
    elif isinstance(sig, np.ndarray):
        indices = np.argsort(-sig.flatten(), kind="stable")
        if get_indices and isinstance(arr, torch.Tensor):
            indices = torch.from_numpy(indices)  # both are torch tensors
    else:
        raise ValueError("indices must be a numpy array or a torch tensor")

    if get_indices:
        return (sort_by_indices(arr, indices, axis, inverse_permutation), indices)
    return sort_by_indices(arr, indices, axis, inverse_permutation)


def sort_by_indices(
    arr: Union[np.ndarray, torch.tensor],
    indices: Union[np.ndarray, torch.tensor],
    axis: str = "rows",
    inverse_permutation: bool = False,
) -> Union[np.ndarray, torch.tensor]:
    """Returns an array ordered by the given indices along the specified axis.

    The indices are used to reorder the array along the specified axis. The
    indices give the order in which the elements should be placed. The type of
    the output is the same as the input array :math:`arr`.

    Args:
        arr (np.ndarray or torch.tensor):
            Array to be ordered by rows or columns. The output's type is
            the same as this parameter's type.

        indices (np.ndarray or torch.tensor):
            Array containing the indices of the elements in the order they
            should be placed.

        axis (str, optional):
            Axis along which to order the array. Must be either "rows" or
            "cols". Defaults to "rows".

        inverse_permutation (bool, optional):
            If True, the permutation matrix is transposed before being used.
            Defaults to False.

    Raises:
        ValueError:
            If axis is not "rows" or "cols".

        ValueError:
            If the number of rows or columns in x is not equal to the length
            of the indices.

    Returns:
        np.ndarray or torch.tensor:
            Array ordered by the given indices along the specified axis.
            The type is the same as the input array arr.

    Example:
        >>> arr = [[10, 20, 30], [100, 200, 300]]
        >>> indices = [2, 0, 1]
        >>> sort_by_indices(arr, indices, axis="cols")
        array([[ 30,  10,  20],
               [300, 100, 200]])
    """
    try:
        axis_index = ["rows", "cols"].index(axis)
    except ValueError:
        raise ValueError(f"axis must be either 'rows' or 'cols', not {axis}")

    # if it is a 1D array, just take the first dimension
    if len(indices.flatten()) != arr.shape[min(axis_index, arr.ndim - 1)]:
        raise ValueError(
            "The number of rows or columns in x must be equal to the "
            "length of the indices array."
        )
    reorder = indices.argsort()

    if ((axis == "rows") and (not inverse_permutation)) or (
        (axis == "cols") and inverse_permutation
    ):
        reorder = reorder.argsort()
        # now it corresponds to the permutation matrix

    if arr.ndim == 1:
        return arr[reorder]
    elif axis == "rows":
        return arr[..., reorder, :]
    return arr[..., :, reorder]


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
