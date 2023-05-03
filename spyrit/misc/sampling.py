from scipy.stats import rankdata
import numpy as np

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
    """Return measurement image from a single measurement vector

    Args:
        meas : `np.ndarray` with shape :math:`(M,)` 
            Set of :math:`B` measurement vectors of lenth :math:`M \le N^2`.
        Mat : `np.ndarray` with shape :math:`(N,N)` 
            Sampling matrix, where high values indicate high significance.

    Returns:
        Img : `np.ndarray` with shape :math:`(N,N,)`
            N-by-N measurement image
    """
    y = np.pad(meas, (0, Mat.size-len(meas)))
    Perm = Permutation_Matrix(Mat)
    Img = np.dot(np.transpose(Perm),y).reshape(Mat.shape)
    return Img

def meas2img2(meas: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    """Return measurement image from multiple measurement vectors

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
    
    y = np.pad(meas, ((0,Mat.size-len(meas)),(0,0)))
    Perm = Permutation_Matrix(Mat)
    Img = Perm.T @ y
    Img = Img.reshape((Nx,Ny,B))
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

def reorder(meas: np.ndarray, 
            Perm_acq: np.ndarray, 
            Perm_rec:  np.ndarray) -> np.ndarray:
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
    N_acq = int(Perm_acq.shape[0]**.5)
    N_rec = int(Perm_rec.shape[0]**.5)
    K_rep = meas.shape[1]
    
    # Acquisition order -> natural order (fill with zeros if necessary)
    if N_rec > N_acq:
        
        # Square subsampling in the "natural" order
        Ord_sub = np.zeros((N_rec,N_rec))
        Ord_sub[:N_acq,:N_acq]= -np.arange(-N_acq**2,0).reshape(N_acq,N_acq)
        Perm_sub = Permutation_Matrix(Ord_sub) 
        
        # Natural order measurements (N_acq resolution)
        Perm_raw = np.zeros((2*N_acq**2,2*N_acq**2))
        Perm_raw[::2,::2] = Perm_acq.T     
        Perm_raw[1::2,1::2] = Perm_acq.T
        meas = Perm_raw @ meas
        
        # Zero filling (needed only when reconstruction resolution is higher 
        # than acquisition res)
        zero_filled = np.zeros((2*N_rec**2, K_rep))
        zero_filled[:2*N_acq**2,:] = meas
        
        meas = zero_filled
        
        Perm_raw = np.zeros((2*N_rec**2,2*N_rec**2))
        Perm_raw[::2,::2] = Perm_sub.T     
        Perm_raw[1::2,1::2] = Perm_sub.T
        
        meas = Perm_raw @ meas
        
    elif N_rec == N_acq:
        Perm_sub = Perm_acq[:N_rec**2,:].T
      
    elif N_rec < N_acq:
        # Square subsampling in the "natural" order
        Ord_sub = np.zeros((N_acq,N_acq))
        Ord_sub[:N_rec,:N_rec]= -np.arange(-N_rec**2,0).reshape(N_rec,N_rec)
        Perm_sub = Permutation_Matrix(Ord_sub) 
        Perm_sub = Perm_sub[:N_rec**2,:]
        Perm_sub = Perm_sub @ Perm_acq.T    
        
    #Reorder measurements when the reconstruction order is not "natural"  
    if N_rec <= N_acq:   
        # Get both positive and negative coefficients permutated
        Perm = Perm_rec @ Perm_sub
        Perm_raw = np.zeros((2*N_rec**2,2*N_acq**2))
        
    elif N_rec > N_acq:
        Perm = Perm_rec
        Perm_raw = np.zeros((2*N_rec**2,2*N_rec**2))
    
    Perm_raw[::2,::2] = Perm     
    Perm_raw[1::2,1::2] = Perm
    meas = Perm_raw @ meas
    
    return meas