from scipy.stats import rankdata
import numpy as np
import torch

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
    (nx, ny) = Mat.shape;
    Mask = np.ones((nx, ny));
    ranked_data = np.reshape(rankdata(-Mat, method = 'ordinal'),(nx, ny));
    Mask[np.absolute(ranked_data)>M]=0;
    return Mask

# from /former/_model_Had_DCAN.py
def meas2img(meas: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    """Return measurement image from measurement vector

    Args:
        meas (np.ndarray): 
            Measurement vector of lenth M <= N**2.
        Mat (np.ndarray): 
            N-by-N sampling matrix, where high values indicate high significance.

    Returns:
        Img (np.ndarray): N-by-N measurement image
    """
    y = np.pad(meas, (0, Mat.size-len(meas)))
    Perm = Permutation_Matrix(Mat)
    Img = np.dot(np.transpose(Perm),y).reshape(Mat.shape)
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
    Perm = Permutation_Matrix(Mat)
    meas = np.dot(Perm, np.ravel(Img))
    return meas

def meas2img_torch(meas, Mat):
    """Return image from measurement vector (NOT TESTED, requires too much memory?)

    Args:
        meas (torch.Tensor): Measurement vector.
        Mat (np.ndarray): Order matrix

    Returns:
        Img (torch.Tensor): Measurement image
    """
    y = torch.nn.functional.pad(meas, (0, Mat.size-meas.shape[2]))
    Perm = torch.from_numpy(Permutation_Matrix(Mat).astype('float32'))
    Perm = Perm.to(meas.device)
    Perm = torch.transpose(Perm,0,1)
    Img = torch.matmul(Perm,meas) # Requires too much memory
    
    return Img

def Permutation_Matrix(Mat: np.ndarray) -> np.ndarray:
    """
        Returns permutation matrix from sampling matrix
        
    Args:
        Mat (np.ndarray): 
            N-by-N sampling matrix, where high values indicate high significance.
        
    Returns:
        P (np.ndarray): N*N-by-N*N permutation matrix (boolean)
    """
    (nx, ny) = Mat.shape;
    Reorder = rankdata(-Mat, method = 'ordinal');
    Columns = np.array(range(nx*ny));
    P = np.zeros((nx*ny, nx*ny));
    P[Reorder-1, Columns] = 1;
    return P