import torch
import torch.nn as nn
import numpy as np
from typing import Union
from spyrit.misc.walsh_hadamard import walsh2_torch, walsh2_matrix
from spyrit.misc.sampling import Permutation_Matrix
# ==================================================================================
class Linear(nn.Module):
# ==================================================================================
    r""" 
        Computes linear measurements from incoming images: :math:`y = Hx`, 
        where :math:`H` is a linear operator (matrix) and :math:`x` is a 
        vectorized image.
        
        The class is constructed from a :math:`M` by :math:`N` matrix :math:`H`, 
        where :math:`N` represents the number of pixels in the image and 
        :math:`M` the number of measurements.
        
        Args:
            :attr:`H`: measurement matrix (linear operator) with shape :math:`(M, N)`.
            
        Attributes:
             :attr:`H`: The learnable measurement matrix of shape 
             :math:`(M,N)` initialized as :math:`H`
             
             :attr:`H_adjoint`: The learnable adjoint measurement matrix 
             of shape :math:`(N,M)` initialized as :math:`H^\top`
             
             
        
        Example:
            >>> H = np.array(np.random.random([400, 1000]))
            >>> meas_op = Linear(H)
    """

    def __init__(self, H: np.ndarray):  
        super().__init__()
        # instancier nn.linear
        self.M = H.shape[0]
        self.N = H.shape[1]
        self.H = nn.Linear(self.N, self.M, False) 
        self.H.weight.data = torch.from_numpy(H).float()
        # Data must be of type float (or double) rather than the default float64 when creating torch tensor
        self.H.weight.requires_grad = False

        # adjoint (Remove?)
        self.H_adjoint = nn.Linear(self.M, self.N, False)
        self.H_adjoint.weight.data = torch.from_numpy(H.transpose()).float()
        self.H_adjoint.weight.requires_grad = False
               
    def forward(self, x: torch.tensor) -> torch.tensor: 
        r""" Applies linear transform to incoming images: :math:`y = Hx`.

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
        # x.shape[b*c,N]
        x = self.H(x)    
        return x
    
    def adjoint(self, x: torch.tensor) -> torch.tensor:
        r""" Applies adjoint transform to incoming measurements :math:`y = H^{T}x`

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
        #Pmat.transpose()*f
        x = self.H_adjoint(x)        
        return x

    def get_H(self) -> torch.tensor:          
        r""" Returns the measurement matrix :math:`H`.
        
        Shape:
            Output: :math:`(M, N)`
        
        Example:     
            >>> H = meas_op.get_H()
            >>> print('get_mat:', H.shape)
            get_mat: torch.Size([400, 1000])
            
        """
        return self.H.weight.data;
    
# ==================================================================================
class LinearSplit(Linear):
# ==================================================================================
    r""" 
    Computes linear measurements from incoming images: :math:`y = Px`, 
    where :math:`P` is a linear operator (matrix) and :math:`x` is a 
    vectorized image.
    
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

    def __init__(self, H: np.ndarray): 
        super().__init__(H)
        
        # [H^+, H^-]
                
        even_index = range(0,2*self.M,2);
        odd_index = range(1,2*self.M,2);

        H_pos = np.zeros(H.shape);
        H_neg = np.zeros(H.shape);
        H_pos[H>0] = H[H>0];
        H_neg[H<0] = -H[H<0];
        
        # pourquoi 2 *M ?
        P = np.zeros((2*self.M,self.N));
        P[even_index,:] = H_pos;
        P[odd_index,:] = H_neg;
        
        self.P = nn.Linear(self.N, 2*self.M, False) 
        self.P.weight.data=torch.from_numpy(P)
        self.P.weight.data=self.P.weight.data.float()
        self.P.weight.requires_grad=False
              
    def forward(self, x: torch.tensor) -> torch.tensor:
        r""" Applies linear transform to incoming images: :math:`y = Px`.
    
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
        x = self.P(x)    
        return x
    
    def forward_H(self, x: torch.tensor) -> torch.tensor:
        r""" Applies linear transform to incoming images: :math:`m = Hx`.
    
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
        x = self.H(x)    
        return x

# ==================================================================================
class HadamSplit(LinearSplit):
    r""" 
    Computes linear measurements from incoming images: :math:`y = Px`, 
    where :math:`P` is a linear operator (matrix) with positive entries and 
    :math:`x` is a vectorized image.
    
    The class is relies on a matrix :math:`H` with 
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
    
    .. note::
        :math:`H = H_{+} - H_{-}`
    
    Args:
        - :attr:`M`: Number of measurements
        - :attr:`h`: Image height :math:`h`. The image is assumed to be square.
        - :attr:`Ord`: Order matrix with shape :math:`(h,h)` used to compute the permutation matrix :math:`G^{T}` with shape :math:`(N, N)` (see the :mod:`~spyrit.misc.sampling` submodule)
    
    .. note::
        The matrix H has shape :math:`(M,N)` with :math:`N = h^2`.
        
    Example:
        >>> Ord = np.random.random([32,32])
        >>> meas_op = HadamSplit(400, 32, Ord)
    """

    def __init__(self, M: int, h: int, Ord: np.ndarray):
        
        F =  walsh2_matrix(h) # full matrix
        Perm = Permutation_Matrix(Ord)
        F = Perm@F # If Perm is not learnt, could be computed mush faster
        H = F[:M,:]
        w = h   # we assume a square image
        
        super().__init__(H)
        
        self.Perm = nn.Linear(self.N, self.N, False)
        self.Perm.weight.data=torch.from_numpy(Perm.T)
        self.Perm.weight.data=self.Perm.weight.data.float()
        self.Perm.weight.requires_grad=False
        self.h = h
        self.w = w
    
    def inverse(self, x: torch.tensor) -> torch.tensor:
        r""" Inverse transform of Hadamard-domain images 
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
        x = 1/self.N*walsh2_torch(x)       
        x = x.view(b, N)
        return x
    
    def pinv(self, x: torch.tensor) -> torch.tensor:
        r""" Pseudo inverse transform of incoming mesurement vectors :math:`x`
        
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
        x = self.adjoint(x)/self.N
        return x


# ==================================================================================
class Linear_shift(Linear):
# ==================================================================================
    r""" Linear with shifted pattern matrix of size :math:`(M+1,N)` and :math:`Perm` matrix of size :math:`(N,N)`.
    
        Args:
            - Hsub: subsampled Hadamard matrix
            - Perm: Permuation matrix
            
        Shape:
            - Input1: :math:`(M, N)`
            - Input2: :math:`(N, N)`
            
        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> Perm = np.array(np.random.random([32*32,32*32]))
            >>> FO_Shift = Linear_shift(Hsub, Perm)
    
    """
    def __init__(self, Hsub, Perm):           
        super().__init__(Hsub)
        
        # Todo: Use index rather than permutation (see misc.walsh_hadamard)
        self.Perm = nn.Linear(self.N, self.N, False)
        self.Perm.weight.data = torch.from_numpy(Perm.T)
        self.Perm.weight.data = self.Perm.weight.data.float()
        self.Perm.weight.requires_grad = False
        
        H_shift = torch.cat(
            (torch.ones((1,self.N)),(self.Hsub.weight.data+1)/2))
        
        self.H_shift = nn.Linear(self.N, self.M+1, False) 
        self.H_shift.weight.data = H_shift      # include the all-one pattern
        self.H_shift.weight.data = self.H_shift.weight.data.float() # keep ?
        self.H_shift.weight.requires_grad = False
         
    def forward(self, x: torch.tensor) -> torch.tensor:
        r""" Applies Linear transform such that :math:`y = \begin{bmatrix}{1}\\{H_{sub}}\end{bmatrix}x`.
        
            Args:
                :math:`x`: batch of images.
                
            Shape:
                - Input: :math:`(b*c, N)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`N` the number of pixels in the image.
                - Output: :math:`(b*c, M+1)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`M+1` the number of measurements + 1.
                
            Example:
                >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
                >>> y = FO_Shift(x)
                >>> print(y.shape)
                torch.Size([10, 401])
        """
        # input x is a set of images with shape (b*c, N)
        # output input is a set of measurement vector with shape (b*c, M+1)
        x = self.H_shift(x) 
        return x
              
        #x_shift = super().forward(x) - x_dark.expand(x.shape[0],self.M) # (H-1/2)x
        
# ==================================================================================
class Linear_pos(Linear):
# ==================================================================================
    r""" Linear with Permutation Matrix :math:`Perm` of size :math:`(N,N)`.
    
        Args:
            - Hsub: subsampled Hadamard matrix
            - Perm: Permuation matrix

        Shape:
            - Input1: :math:`(M, N)`
            - Input2: :math:`(N, N)`

        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> Perm = np.array(np.random.random([32*32,32*32]))
            >>> meas_op_pos = Linear_pos(Hsub, Perm)
    """
    def __init__(self, Hsub, Perm):           
        super().__init__(Hsub)
        
        # Todo: Use index rather than permutation (see misc.walsh_hadamard)
        self.Perm = nn.Linear(self.N, self.N, False)
        self.Perm.weight.data = torch.from_numpy(Perm.T)
        self.Perm.weight.data = self.Perm.weight.data.float()
        self.Perm.weight.requires_grad = False
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Computes :math:`y` according to :math:`y=0.5(H_{sub}x+\sum_{j=1}^{N}x_{j})` where :math:`j` is the pixel (column) index of :math:`x`.
        
        Args:
            :math:`x`: Batch of images.
            
        Shape:
            - Input: :math:`(b*c, N)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`N` the number of pixels in the image.
            - Output: :math:`(b*c, M)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`M` the number of measurements.
            
        Example:
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = meas_op_pos(x)
            >>> print(y.shape)
            torch.Size([100, 400]) 
        """
        # input x is a set of images with shape (b*c, N)
        # output is a set of measurement vectors with shape (b*c, M)
        
        # compute 1/2(H+1)x = 1/2 HX + 1/2 1x
        x = super().forward(x) + x.sum(dim=1,keepdim=True).expand(-1, self.M)
        x *= 0.5
        
        return x
    
# ==================================================================================
class Linear_shift_had(Linear_shift):
# ==================================================================================
    r""" Linear_shift operator with inverse method.
    
        Args:
            - Hsub: subsampled Hadamard matrix
            - Perm: Permuation matrix

        Shape:
            - Input1: :math:`(M, N)`
            - Input2: :math:`(N, N)`.   
    
        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> Perm = np.array(np.random.random([32*32,32*32]))
            >>> FO_Shift_Had = Linear_shift_had(Hsub, Perm)
    """
    def __init__(self, Hsub, Perm):           
        super().__init__(Hsub, Perm)
    
    def inverse(self, x: torch.tensor, n: Union[None, int] = None) -> torch.tensor:
        r""" Inverse transform such that :math:`x = \frac{1}{N}H_{sub}y`.
        
        Args:
            :math:`x`: Batch of completed measurements.
            
        Shape:
            - Input: :math:`(b*c, N)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`N` the number of measurements.
            - Output: :math:`(b*c, N)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`N` the number of reconstructed. pixels.
            
        Example:
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> x_reconstruct = FO_Shift_Had.inverse(y_pad) 
            >>> print(x_reconstruct.shape)
            torch.Size([10, 1024])          
        """
        # rearrange the terms + inverse transform
        # maybe needs to be initialised with a permutation matrix as well!
        # Permutation matrix may be sparsified when sparse tensors are no longer in
        # beta (as of pytorch 1.11, it is still in beta).
        
        # --> Use index rather than permutation (see misc.walsh_hadamard)
        
        # input x is a set of **measurements** with shape (b*c, N)
        # output is a set of **images** with shape (b*c, N)
        bc, N = x.shape
        x = self.Perm(x);
        
        if n is None:
            n = int(np.sqrt(N))
        
        # Inverse transform    
        x = x.view(bc, 1, n, n)
        x = 1/self.N*walsh2_torch(x) # todo: initialize with 1D transform to speed up
        x = x.view(bc, N)
        return x

# ==================================================================================
class LinearRowSplit(nn.Module):
# ================================================================================== 
    r""" Compute linear measurement of incoming images :math:`y = Px`, where 
        :math:`P` is a linear operator and :math:`x` is an image. Note that
        the same transform applies to each of the rows of the image :math:`x`.

        The class is constructed from the positive and negative components of 
        the measurement operator :math:`P = \begin{bmatrix}{H_{+}}\\{H_{-}}\end{bmatrix}`
        
        Args:
            - :attr:`H_pos`: Positive component of the measurement matrix :math:`H_{+}`
            - :attr:`H_neg`: Negative component of the measurement matrix :math:`H_{-}`
        
        Shape:
            :math:`H_{+}`: :math:`(M, N)`, where :math:`M` is the number of 
            patterns and :math:`N` is the length of the patterns. 
            
            :math:`H_{-}`: :math:`(M, N)`, where :math:`M` is the number of 
            patterns and :math:`N` is the length of the patterns.
            
        .. note::
            The class assumes the existence of the measurement operator 
            :math:`H = H_{+}-H_{-}` that contains negative values that cannot be
            implemented in practice (harware constraints).
        
        Example:
            >>> H_pos = np.random.rand(24,64)
            >>> H_neg = np.random.rand(24,64)
            >>> linop = LinearRowSplit(H_pos,H_neg)
        
        """    
    def __init__(self, H_pos: np.ndarray, H_neg: np.ndarray):
        
        
        super().__init__()
           
        self.M = H_pos.shape[0]
        self.N = H_pos.shape[1]
        
        # Split patterns ?
        # N.B.: Data must be of type float (or double) rather than the default 
        # float64 when creating torch tensor
        even_index = range(0,2*self.M,2)
        odd_index = range(1,2*self.M,2)
        P = np.zeros((2*self.M,self.N))
        P[even_index,:] = H_pos
        P[odd_index,:] = H_neg
        self.P = nn.Linear(self.N, 2*self.M, False);
        self.P.weight.data = torch.from_numpy(P).float()
        self.P.weight.requires_grad = False
        
        # "Unsplit" patterns
        H = H_pos - H_neg
        self.H = nn.Linear(self.N, self.M, False);
        self.H.weight.data = torch.from_numpy(H).float() 
        self.H.weight.requires_grad = False
              
    def forward(self, x: torch.tensor) -> torch.tensor:
        r""" Applies linear transform to incoming images: :math:`y = Px`
        
        Args:
            x: a batch of images
        
        Shape:
            x: :math:`(b*c, h, w)` with :math:`b` the batch size, :math:`c` the 
            number of channels, :math:`h` is the image height, and :math:`w` is the image 
            width.

            Output: :math:`(b*c, 2M, w)` with :math:`b` the batch size,
            :math:`c` the number of channels, :math:`2M` is twice the number of
            patterns (as it includes both positive and negative components), and 
            :math:`w` is the image width.
            
            .. warning::
                The image height :math:`h` should match the length of the patterns 
                :math:`N`

        Example:
            >>> H_pos = np.random.rand(24,64)
            >>> H_neg = np.random.rand(24,64)
            >>> linop = LinearRowSplit(H_pos,H_neg)
            >>> x = torch.rand(10,64,92)
            >>> y = linop(x)
            >>> print(y.shape)
            torch.Size([10,48,92])
         
        """
        x = torch.transpose(x,1,2) #swap last two dimensions
        x = self.P(x)
        x = torch.transpose(x,1,2) #swap last two dimensions
        return x
    
    def forward_H(self, x: torch.tensor) -> torch.tensor:
        r""" Applies linear transform to incoming images: :math:`m = Hx`
        
        Args:
            x: a batch of images
        
        Shape:
            x: :math:`(b*c, h, w)` with :math:`b` the batch size, :math:`c` the 
            number of channels, :math:`h` is the image height, and :math:`w` is the image 
            width.

            Output: :math:`(b*c, M, w)` with :math:`b` the batch size,
            :math:`c` the number of channels, :math:`M` is the number of
            patterns, and :math:`w` is the image width.
            
            .. warning::
                The image height :math:`h` should match the length of the patterns 
                :math:`N`

        Example:
            >>> H_pos = np.random.rand(24,64)
            >>> H_neg = np.random.rand(24,64)
            >>> meas_op = LinearRowSplit(H_pos,H_neg)
            >>> x = torch.rand(10,64,92)
            >>> y = meas_op.forward_H(x)
            >>> print(y.shape)
            torch.Size([10,24,92])
         
        """
        x = torch.transpose(x,1,2) #swap last two dimensions
        x = self.H(x)
        x = torch.transpose(x,1,2) #swap last two dimensions
        return x
    
    def get_H(self) -> torch.tensor:          
        r""" Returns the measurement matrix :math:`H`.
        
        Shape:
            Output: :math:`(M, N)`
        
        Example:     
            >>> H = meas_op.get_H()
            >>> print(H.shape)
            torch.Size([24, 64])
        """
        return self.H.weight.data;