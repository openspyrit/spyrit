from typing import Union
import warnings

import torch
import numpy as np
from scipy.stats import rankdata
from scipy.ndimage import label
from spyrit.core.torch import walsh_matrix_2d
import ptwt


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
    nx, ny = Mat.shape
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


def define_order(n: int, order: str, pdf: bool = False):
    """
    Creation of a Hadamard pattern order

    Parameters
    ----------
    n : int
        Dimension. (Patterns of size n by n)
    order : string
        Type of order.
    pdf : bool, optional
        If True the function returns a normalised PDF such that the sum of the values of the
        output tensor is equal to one.
        If False the output is the ranking associated to each pattern. The default is False.

    Returns
    -------
    torch.tensor
        tensor of size n by n containing the PDF or ranks.

    """

    if not isinstance(n, int) or n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be an integer power of 2, got {n}")

    H = walsh_matrix_2d(n)
    order_list = ["Sequency", "TV", "CC", "Variance"]

    if order not in order_list:
        raise ValueError(f"Order must be in {order_list}")

    N = n**2
    h = [H[i].reshape(n, n) for i in range(N)]

    if order == "Sequency":

        freq = torch.zeros(N)

        for i in range(N):
            freq_x = torch.sum(torch.diff(h[i], dim=1) != 0, dim=1)[
                0
            ]  # Number of sign changes per row
            freq_y = torch.sum(torch.diff(h[i], dim=0) != 0, dim=0)[
                0
            ]  # Number of sign changes per column
            freq[i] = freq_x + freq_y

        freq_score = 1 / (
            freq + 1e-8
        )  # Give highest score (weight) to lowest frequencies
        freq_score[0] = 1  # Avoid too high value due to the division by zero

        if pdf is False:
            freq_rank = torch.argsort(freq_score, descending=True)
            freq_order = torch.zeros(N)
            for i in range(N):
                freq_order[freq_rank[i]] = i
            return freq_order.reshape(n, n)
        else:
            return (freq_score / freq_score.sum()).reshape(n, n)

    elif order == "TV":
        TV = []
        for i in range(N):
            dx = h[i][:, 1:] - h[i][:, :-1]
            dy = h[i][1:, :] - h[i][:-1, :]

            TV.append(torch.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2).sum())
        TV = torch.tensor(TV)

        score_tv = 1 / (TV + 1e-8)
        score_tv[0] = 1

        if pdf is False:
            TV_rank = torch.argsort(score_tv, descending=True)
            TV_order = torch.zeros(N)
            for i in range(N):
                TV_order[TV_rank[i]] = i
            return TV_order.reshape(n, n)
        else:
            return (score_tv / score_tv.sum()).reshape(n, n)

    elif order == "CC":
        CC_values = torch.zeros(N)

        for i in range(N):

            patt = np.asarray(h[i])
            pos = (patt > 0).astype(int)
            neg = (patt < 0).astype(int)
            _ , num_pos = label(pos)
            _ , num_neg = label(neg)
            CC_values[i] = num_pos + num_neg

        score_CC = 1 / (CC_values + 1e-8)
        score_CC[0] = 1

        if pdf is False:
            CC_rank = torch.argsort(score_CC, descending=True)
            CC_order = torch.zeros(N)
            for i in range(N):
                CC_order[CC_rank[i]] = i
            return CC_order.reshape(n, n)
        else:
            return (score_CC / score_CC.sum()).reshape(n, n)

    elif order == "Variance":
        # The order matrix corresponding is obtained by computing the variance of the Hadamard coefficients of the images belonging to the ImageNet 2012 dataset.

        # First, we download the covariance matrix from our warehouse. The covariance was computed from the ImageNet 2012 dataset and has a size of (64*64, 64*64).
        from spyrit.misc.load_data import download_girder

        # url of the warehouse
        url = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
        dataId = "672207cbf03a54733161e95d"  # for reconstruction (imageNet, 64)
        data_folder = "./stat/"
        cov_name = "Cov_64x64.pt"

        # download the covariance matrix and get the file path
        file_abs_path = download_girder(url, dataId, data_folder, cov_name)

        try:
            # Load covariance matrix for "variance subsampling"
            Cov = torch.load(file_abs_path, weights_only=True)
            print(f"Cov matrix {cov_name} loaded")
        except (FileNotFoundError, OSError, RuntimeError):
            # Set to the identity if not found for "naive subsampling"
            Cov = torch.eye(64 * 64)
            print(f"Cov matrix {cov_name} not found! Set to the identity")

        from spyrit.core.torch import Cov2Var

        Ord_variance = Cov2Var(Cov)

        if pdf is False:
            Var_rank = torch.argsort(Ord_variance.flatten(), descending=True)
            Var_order = torch.zeros(N)
            for i in range(N):
                Var_order[Var_rank[i]] = i
            return Var_order.reshape(n, n)
        else:
            return Ord_variance / Ord_variance.sum()


def sampling_map_from_order(order: torch.tensor, M: int):
    """
    Generate a sampling map from a given order (ranking) and number of measurements

    Parameters
    ----------
    order : torch.tensor
        n by n matrix containing the rankings (order) corresponding to each Hadamard pattern.
    M : int
        Number of measurements.

    Returns
    -------
    s_map : torch.tensor
        n by n binary sampling map.

    """

    if (torch.sum(order) - 1) < 1e-6:
        raise ValueError("order must be a ranking of the patterns not a PDF.")

    if M > order.shape[0] ** 2:
        raise ValueError(
            "The number of measurements M must be lower or equal than the number of patterns"
        )

    s_map = torch.zeros_like(order)
    s_map[order < M] = 1

    return s_map


def sampling_map_VDS(pdf: torch.tensor, M: int, seed: int = 0):
    """
    Define a VDS sampling scheme that follows a PDF.

    Parameters
    ----------
    pdf : torch.tensor
        Probability distribution function.
    M : int
        Number of measurements.
    seed : int, optional
        Fixed seed for reproducibility. The default is 0.

    Returns
    -------
    sampling_map : torch.tensor
        Sampling map.

    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")

    torch.manual_seed(seed)

    n = pdf.shape[0]
    N = n**2

    samp = torch.multinomial(pdf.reshape(N)[1:], M - 1, replacement=False) + 1
    samp = torch.cat(
        (torch.tensor([0]), samp)
    )  # Force the selection of the first pattern

    sampling_map = torch.zeros(n, n)
    sampling_map.reshape(N)[samp] = 1

    return sampling_map


def sampling_map_multilevel_VDS(
    pdf: torch.tensor,
    M: int,
    levels: int,
    J: int = 3,
    wave: str = "sym8",
    mode: str = "periodic",
    seed: int = 0,
):
    """
    Generation of a sampling map following a Multilevel VDS sampling scheme

    Parameters
    ----------
    pdf : torch.tensor
        PDF (or order) used to discriminate the sampling levels.
    M : int
        Total number of measurements.
    levels : int
        Number of sampling levels.
    J : int, optional
        Number of wavelet decomposition levels. The default is 3.
    wave : string, optional
        Wavelet type. The default is 'sym8'.
    mode : string, optional
        Wavelet mode. The default is 'periodization'.
    seed: int, optional
        Fixed seed for reproducibility. Default is 0.

    Returns
    -------
    sampling_map : torch.tensor
        Multilevel sampling map.

    """
    torch.manual_seed(seed)

    n = pdf.shape[0]
    N = n**2
    H = walsh_matrix_2d(n)

    #dwt = DWTForward(J=J, wave=wave, mode=mode)

    lvl_sizes = torch.zeros(levels)  # number of elements in each level
    lvl_maps = torch.zeros(levels, n, n)
    selected = 0  # Number of elements already selected

    mu_kl = torch.zeros(
        levels, J + 1
    )  # Local coherences per sampling and wavelet levels

    sampling_map = torch.zeros(n, n)
    m_k = torch.zeros(levels)  # Number of measurements in each level

    for k in range(levels):
        lvl_sizes[k] = (n / (2 ** (levels - k - 1))) ** 2
        lvl_sizes[k] -= torch.sum(lvl_sizes[:k])

        mask_basis = torch.zeros(N)
        mask_basis[selected : selected + int(lvl_sizes[k])] = 1
        selected += int(lvl_sizes[k])

        lvl_maps[k] = sort_by_significance(mask_basis, pdf).reshape(n, n)

        H_k = H[
            lvl_maps[k].reshape(N).int() == 1
        ]  # Selection of the patterns in the desired level

        mu_loc = torch.zeros(
            int(lvl_sizes[k]), J + 1
        )  # Local coherences inside each level

        for i in range(int(lvl_sizes[k])):
            coeffs = ptwt.wavedec2(H_k[i].reshape(n, n).unsqueeze(0).unsqueeze(0),
                                   wavelet=wave,mode=mode,level=J)
            mu_loc[i, 0] = torch.max(abs(coeffs[0]))
            for j in range(J):
                mu_loc[i, j + 1] = torch.max(abs(coeffs[1][2 - j]))

        for l in range(J + 1):
            mu_kl[k, l] = torch.max(abs(mu_loc[:, l]))
            m_k[k] += mu_kl[k, l] * 2 ** (l + 1)

    m = m_k / m_k.sum() * M  # Normalise to have a total of M measurements
    m = torch.round(m)

    # Due to the rounding operation there might be slight mismatch that we must
    # fix between m and M
    if int(torch.sum(m)) < M:
        m[0] += M - int(torch.sum(m))
    if int(torch.sum(m)) > M:
        m[levels - 1] -= int(torch.sum(m)) - M

    selected_idx = torch.tensor([])
    for k in range(levels):
        if m[k] > lvl_sizes[k]:
            remaining = m[k] - lvl_sizes[k]
            m[k] = int(lvl_sizes[k])
            m[k + 1] += int(remaining)

        # Set of indices in the level
        level_idx = torch.nonzero(lvl_maps[k].reshape(N), as_tuple=False)
        # Draw uniformly the desired number of indices in this set
        Omega_k_idx = torch.multinomial(
            torch.ones(int(lvl_sizes[k])) / lvl_sizes[k], int(m[k]), replacement=False
        )
        # Apply the mask to select the indices
        Omega_k = level_idx[Omega_k_idx.long()]
        # Concatenate the list of selected indices in all levels
        selected_idx = torch.cat((selected_idx, Omega_k))

    sampling_map.reshape(N)[selected_idx.long()] = 1

    return sampling_map


def reorder_from_sampling_map(
    meas: np.ndarray, Ord_acq: np.ndarray, s_map: np.ndarray
) -> np.ndarray:
    """
    Reorder splitted measurements following a sampling map

    Parameters
    ----------
    meas : np.ndarray
        Measurement array of size (2*N,C) with N the number of patterns acquired and C the number of channels.
    Ord_acq : np.ndarray
        (N,) Array containing the indices of the patterns corresponding to each measurement.
    s_map : np.ndarray
        (n,n) array containing the sampling map.

    Returns
    -------
    meas_rec : np.ndarray
        Reordered measurement vector.

    """

    s_map = s_map.flatten()
    C = meas.shape[1]
    N_rec = int(
        s_map[s_map == 1].shape[0]
    )  # Number of patterns used for reconstruction

    # Pass from acquisition order to natural order
    # If some patterns have not been acquired, their slots are filled with zeros

    meas_nat = np.zeros((2 * len(s_map), C))

    for i, j in enumerate(Ord_acq):
        meas_nat[2 * j] = meas[2 * i]
        meas_nat[2 * j + 1] = meas[2 * i + 1]

    # Pass from natural order to reconstruction order
    meas_rec = np.zeros((2 * N_rec, C))
    j = 0
    for i, val in enumerate(s_map):
        if val == 1:
            meas_rec[2 * j] = meas_nat[2 * i]
            meas_rec[2 * j + 1] = meas_nat[2 * i + 1]
            j += 1

    return meas_rec
