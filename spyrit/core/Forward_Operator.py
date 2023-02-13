# ==================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import poisson
from collections import OrderedDict
from spyrit.misc.walsh_hadamard import walsh2_torch, walsh_matrix
from typing import Union

# ==================================================================================
# Forward operators
# ==================================================================================
# ==================================================================================
class Linear(nn.Module):
# ==================================================================================
    r""" 
        Computes linear measurements from incoming images: :math:`y = Hx`, 
        where :math:`H`is a linear operator (matrix) and :math:`x` is a 
        vectorized image.
        
        The class is constructed from a :math:`M` by :math:`N` matrix :math:`H`, 
        where :math:`N` represents the number of pixels in the image and 
        :math:`M` the number of measurements.
        
        
        Args:
            - :math:`H` (np.ndarray): measurement matrix (linear operator) with 
            shape :math:`(M, N)`.
        
        Example:
            >>> H = np.array(np.random.random([400,32*32]))
            >>> linop = Linear(H)
    """

    def __init__(self, Hsub: np.ndarray):  
        super().__init__()
        # instancier nn.linear        
        # Pmat --> (torch) --> Poids ()
        self.M = Hsub.shape[0];
        self.N = Hsub.shape[1];
        self.Hsub = nn.Linear(self.N, self.M, False); 
        self.Hsub.weight.data=torch.from_numpy(Hsub)
        # Data must be of type float (or double) rather than the default float64 when creating torch tensor
        self.Hsub.weight.data=self.Hsub.weight.data.float()
        self.Hsub.weight.requires_grad=False

        # adjoint (Not useful here ??)
        self.Hsub_adjoint = nn.Linear(self.M, self.N, False)
        self.Hsub_adjoint.weight.data=torch.from_numpy(Hsub.transpose())
        self.Hsub_adjoint.weight.data = self.Hsub_adjoint.weight.data.float()
        self.Hsub_adjoint.weight.requires_grad = False
               
    def forward(self, x: torch.tensor) -> torch.tensor: 
        r""" Applies linear transform to incoming images: :math:`y = Hx`.

        Args:
            :math:`x`: Batch of vectorized (flatten) images.
            
        Shape:
            - :math:`x`: :math:`(*, N)` where * denotes the batch size and `N` 
            the total number of pixels in the image.
            - Output: :math:`(*, M)` where * denotes the batch size and `M` 
            the number of measurements.
            
        Example:        
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = linop(x)
            >>> print('Output shape of forward:', y.shape)
            output shape: torch.Size([10, 400])
            
        """
        # x.shape[b*c,N]
        x = self.Hsub(x)    
        return x
    
    def adjoint(self, x: torch.tensor) -> torch.tensor:
        r""" Applies adjoint transform to incoming measurements: :math:`y = H^{T}x`

        Args:
            :math:`x`:  batch of measurements vector.
            
        Shape:
            - :math:`x`: :math:`(*, M)`
            - Output: :math:`(*, N)`
            
        Example:
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)        
            >>> y = linop.adjoint(x)
            >>> print('Output shape of adjoint:', y.shape)
            adjoint output shape: torch.Size([10, 1024])
            
        """
        # x.shape[b*c,M]
        #Pmat.transpose()*f
        x = self.Hsub_adjoint(x)        
        return x

    def get_mat(self) -> torch.tensor:          
        r""" Returns the measurement matrix :math:`H`.
        
        Shape:
            - Output: :math:`(M, N)`
        
        Example:     
            >>> H = linop.get_mat()
            >>> print('Shape of the measurement matrix:', H.shape)     
        """
        return self.Hsub.weight.data;


## Merge Linear_Split and Linear_Split_ft_had -> Linear_shift_had

# ==================================================================================
class Linear_Split(Linear):
# ==================================================================================
    r""" Linear with a :math:`H_{pos_neg}` matrix of size :math:`(2*M,N).
            
        Args:
            - :math:`H_{sub}`: subsampled Hadamard matrix
            
        Shape:
            - Input: :math:`(M, N)`
            
        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> Forward_Op_Split =  Linear_Split(Hsub)
     """

    def __init__(self, Hsub: np.ndarray): 
        super().__init__(Hsub)
        
        # [H^+, H^-]
                
        even_index = range(0,2*self.M,2);
        odd_index = range(1,2*self.M,2);

        H_pos = np.zeros(Hsub.shape);
        H_neg = np.zeros(Hsub.shape);
        H_pos[Hsub>0] = Hsub[Hsub>0];
        H_neg[Hsub<0] = -Hsub[Hsub<0];
        
        # pourquoi 2 *M ?
        Hposneg = np.zeros((2*self.M,self.N));
        Hposneg[even_index,:] = H_pos;
        Hposneg[odd_index,:] = H_neg;
        
        self.Hpos_neg = nn.Linear(self.N, 2*self.M, False) 
        self.Hpos_neg.weight.data=torch.from_numpy(Hposneg)
        self.Hpos_neg.weight.data=self.Hpos_neg.weight.data.float()
        self.Hpos_neg.weight.requires_grad=False
              
    def forward(self, x: torch.tensor) -> torch.tensor: # --> simule la mesure sous-chantillonnée
        r""" Linear transform of batch of images :math:`x` such that 
        :math:`y =H_{posneg}*x` where :math:`H_{posneg} = \begin{bmatrix}{H_{pos}}
        \\{H_{neg}}\end{bmatrix}`.
        
        Args:
            :math:`x`: Batch of images.
            
        Shape:
            - Input: :math:`(*,N)`
            - Output: :math:`(*, 2M)`
        
        Example:
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> x_output = Forward_Op_Split(x)
            >>> print(x_output.shape)
            torch.Size([10, 800])
                    
        """
        # x.shape[b*c,N]
        # output shape : [b*c, 2*M]
        x = self.Hpos_neg(x)    
        return x

# ==================================================================================
class Linear_Split_ft_had(Linear_Split): 
    r""" Linear_Split with implemented inverse transform and a permutation matrix: :math:`Perm` of size :math:`(N,N)`.

        Args:
            - :math:`H_{sub}`: subsampled Hadamard matrix
            - :math:`Perm`: Permutation Matrix
            - :math:`h`: Image height
            - :math:`w`: Image width
            
        Shape:
            - Input1: :math:`(M, N)`
            - Input2: :math:`(N, N)`
            - Input3: scalar
            - Input4: scalar
            
            
        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> Perm = np.array(np.random.random([32*32,32*32]))
            >>> FO_Had = Linear_Split_ft_had(Hsub, Perm, 32, 32)
    """
# ==================================================================================
# Forward operator with implemented inverse transform and a permutation matrix
    def __init__(self, Hsub: np.ndarray, Perm: np.ndarray, h: int, w: int) -> torch.tensor:
        
        super().__init__(Hsub);
        self.Perm = nn.Linear(self.N, self.N, False)
        self.Perm.weight.data=torch.from_numpy(Perm.T)
        self.Perm.weight.data=self.Perm.weight.data.float()
        self.Perm.weight.requires_grad=False
        
        self.h = h
        self.w = w
        
        # Build H - 1D, store and give it as argument
        #self.H_1_D = ; 
    
    def inverse(self, x: torch.tensor) -> torch.tensor:
        r""" Inverse transform of x with permutation matrix.
        
            Args:
                :math:`x` :  batch of images
                
            Shape:
                - Input: :math:`(b*c, N)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`N` the number of pixels in the image.
                - Output: math:`(b*c, N)`      
                
            Example:

                >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)  
                >>> x_inverse = FO_Had.inverse(x)
                >>> print(x_inverse.shape)
                torch.Size([10, 1024])
        """
        # rearrange the terms + inverse transform
        # maybe needs to be initialized with a permutation matrix as well!
        # Permutation matrix may be sparsified when sparse tensors are no longer in
        # beta (as of pytorch 1.11, it is still in beta).
        
        # input - x - shape [b*c, N]
        # output - x - shape [b*c, N]
        b, N = x.shape
        x = self.Perm(x)
        x = x.view(b, 1, self.h, self.w)
        x = 1/self.N*walsh2_torch(x)    #to apply the inverse transform
                                        # todo: initialize with 1D transform to speed up
        # Build H - 1D, store and give it as argument
        x = x.view(b, N);
        return x
    
    def pinv(self, x: torch.tensor) -> torch.tensor:
        r""" Inverse transform of x using Linear adjoint method.
        
            Args:
                :math:`x` :  batch of images
                
            Shape:
                - Input: :math:`(b*c, N)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`N` the number of pixels in the image.
                - Output: same as input.      
                
            Example:
                >>> x = torch.Tensor(np.random.random([10,400]))  
                >>> x_pinv = FO_Had.pinv(x)
                >>> print(x_pinv.shape)
                torch.Size([10, 1024])
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
            >>> Forward_OP_pos = Linear_pos(Hsub, Perm)
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
            >>> y = Forward_OP_pos(x)
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
    r""" Compute linear measurement of incoming images :math:`y = Hx`, where 
        :math:`H` is a linear operator and :math:`x` is an image. Note that
        the same transform applies to each of the rows of the image :math:`x`.

        The class is constructed from the positive and negative components of 
        the measurement patterns :math:`H_{raw} = \begin{bmatrix}{H_{pos}}\\{H_{neg}}\end{bmatrix}`
        
        Args:
            - :math:`H_{pos}` (np.ndarray): Positive component of the measurement patterns
            - :math:`H_{neg}`(np.ndarray): Negative component of the measurement patterns
        
        Shape:
            - :math:`H_{pos}`: :math:`(M, N)`, 
            - :math:`H_{neg}`: :math:`(M, N)`,
            where :math:`M` is the number of patterns and :math:`N` is the 
            length of the patterns.
            
        .. note::
            The class assumes the existence of the measurement operator 
            `H = H_{pos}-H_{neg}` that contains negative values that cannot be
            implemented in practice (harware constraints).
        
        Example:
            >>> H_pos = np.random.rand(64,128)
            >>> H_neg = np.random.rand(64,128)
            >>> linear_row = LinearRowSplit(H_pos,H_neg)
        
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
        Hposneg = np.zeros((2*self.M,self.N))
        Hposneg[even_index,:] = H_pos
        Hposneg[odd_index,:] = H_neg
        
        self.Hpos_neg = torch.from_numpy(Hposneg).float()
        self.Hpos_neg.requires_grad = False
        
        # "Unsplit" patterns
        H = H_pos - H_neg
        self.H = torch.from_numpy(H).float() 
        self.H.requires_grad = False
       
        # adjoint (Not useful here ??)
        # self.Hsub_adjoint = nn.Linear(self.M, self.N, False)
        # self.Hsub_adjoint.weight.data=torch.from_numpy(Hsub.transpose())
        # self.Hsub_adjoint.weight.data = self.Hsub_adjoint.weight.data.float()
        # self.Hsub_adjoint.weight.requires_grad = False
              
    def forward(self, x: torch.tensor) -> torch.tensor:
        r""" Applies linear transform to incoming images: :math:`y = Hx`
        
        Args:
            - :math:`x`: a batch of images
        
        Shape:
            - Input: :math:`(b*c, h, w)` with :math:`b` the batch size, :math:`c` the 
            number of channels, :math:`h` is the image height, and :math:`w` is the image 
            width.

            - Output: :math:`(b*c, 2M, w)` with :math:`b` the batch size,
            :math:`c` the number of channels, :math:`2M` is twice the number of
            patterns (as it includes both positive and negative components), and 
            :math:`w` is the image width.
            
            .. warning::
                The image height :math:`h` should match the length of the patterns 
                :math:`N`

        Example:
            >>> H_pos = np.random.rand(24,64)
            >>> H_neg = np.random.rand(24,64)
            >>> A = LinearRowSplit(H_pos,H_neg)
            >>> x = np.random.rand(10,64,92)
            >>> y = A(x)
            >>> print(y.shape)
            torch.Size([10,48,92])
         
        """
        x = self.Hpos_neg @ x  
        return x