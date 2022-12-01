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
class Forward_operator(nn.Module):
# ==================================================================================
    r""" Computes Linear transform of image batch x such that :math:`y = H_{sub}x` where :math:`Hsub` (standing for "sub-sampled Hadamard") is a an :math:`M` by  :math:`N` matrix. :math:`N` is the number of pixels per image, and :math:`M` is the number of measurements.
            
        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> Forward_OP = Forward_operator(Hsub)             
    """
# Faire le produit H*f sans bruit, linear (pytorch) 
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
        r""" Applies Linear transform such that :math:`y = H_{sub}x`

        Args:
            :math:`x` : Batch of images of size :math:`N` where :math:`N=img_x*img_y`
            
        Shape:
            - Input: :math:`(*, N)` where * denotes the batch size and `N` the image size
            - Output: :math:`(*, M)` where * denotes the batch size and `M` the number of simulated measurements
            
        Example:        
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = Forward_OP(x)
            >>> print('output shape:', y.shape)
            output shape: torch.Size([10, 400])
            
        """
        # x.shape[b*c,N]
        x = self.Hsub(x)    
        return x

    def Forward_op(self,x: torch.tensor) -> torch.tensor:     # todo: Rename to "direct"
        r""" same as forward.
        
        """
        # x.shape[b*c,N]
        x = self.Hsub(x)    
        return x
    
    def adjoint(self, x: torch.tensor) -> torch.tensor:
        r""" Applies Linear transform such that :math:`y = H_{sub}^{T}x`

        Args:
            :math:`x`:  batch of sub-sampled and convolved images.
            
        Shape:
            - Input: :math:`(*, M)`
            - Output: :math:`(*, N)`
            
        Example:
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)        
            >>> x_back = Forward_OP.adjoint(x)
            >>> print('adjoint output shape:', x_back.shape)
            adjoint output shape: torch.Size([10, 1024])
            
        """
        # x.shape[b*c,M]
        #Pmat.transpose()*f
        x = self.Hsub_adjoint(x)        
        return x

    def Mat(self) -> torch.tensor:          # todo: Remove capital letter
        r""" Provides :math:`H_{sub}` matrix weigths.
        """
        return self.Hsub.weight.data;


## Merge Forward_operator_Split and Forward_operator_Split_ft_had -> Forward_operator_shift_had

# ==================================================================================
class Forward_operator_Split(Forward_operator):
# ==================================================================================
    r""" Child class of Forward_operator. It splits :math:`H_{sub}` into :math:`H_{pos}` and :math:`H_{neg}` before it applies a linear transform to a batch of images denoted :math:`x`.
    See Antonio Lorente Mur et. al. Handling negative patterns for fast single-pixel lifetime imaging. 2019 - Molecular-Guided Surgery: Molecules, Devices, and Applications V, Feb 2019, San Francisco, United States. pp.1-10, `10.1117/12.2511123 <https://hal.archives-ouvertes.fr/hal-02017598/document/>`_.
            
        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> Forward_Op_Split =  Forward_operator_Split(Hsub)
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
        r""" Linear transform of batch of images :math:`x` such that :math:`y =H_{posneg}*x` where :math:`H_{posneg} = \begin{bmatrix}{H_{pos}}\\{H_{neg}}\end{bmatrix}`.
        
        Args:
            :math:`H_{sub}`: Global pattern matrix with both positive and negative values.
            
        Shape:
            - Input: :math:`(*,N)`
            - Output: :math:`(*, 2M)`
        
        Example:
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> x_output = Forward_Op_Split(x)
            >>> print('output shape:', x_output.shape)
            output shape: torch.Size([10, 800])
                    
        """
        # x.shape[b*c,N]
        # output shape : [b*c, 2*M]
        x = self.Hpos_neg(x)    
        return x

# ==================================================================================
class Forward_operator_Split_ft_had(Forward_operator_Split): 
    r""" Child class of Forward_operator_Split with implemented inverse transform and a permutation matrix. It is instantiated by :math:`H_{sub}` matrix of size :math:`(M,N)`, :math:`Perm` matrix of size :math:`(N,N)`, :math:`h`, and :math:`w` that verify :math:`N = h*w`.

        
        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> Perm = np.array(np.random.random([32*32,32*32]))
            >>> FO_Had = Forward_operator_Split_ft_had(Hsub, Perm, 32, 32)
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
                - Output: same as input.      
                
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
        r""" Inverse transform of x using Forward_Operator adjoint method.
        
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
class Forward_operator_shift(Forward_operator):
# ==================================================================================
    r""" Creates forward operator with shifted pattern matrix according to: :math:`H_{sub}(i,j) = \frac{H_{sub}(i,j)+1}{2}`.
        
        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> Perm = np.array(np.random.random([32*32,32*32]))
            >>> FO_Shift = Forward_operator_shift(Hsub, Perm)
    
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
        """
        # input x is a set of images with shape (b*c, N)
        # output input is a set of measurement vector with shape (b*c, M+1)
        x = self.H_shift(x) 
        return x
              
        #x_shift = super().forward(x) - x_dark.expand(x.shape[0],self.M) # (H-1/2)x
        
# ==================================================================================
class Forward_operator_pos(Forward_operator):
# ==================================================================================
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
            >>> h, w = 32, 32
            >>> img_size = h*w
            >>> nb_measurements = 400
            >>> batch_size = 100
            >>> Hcomplete = walsh_matrix(img_size)
            >>> Perm = np.array(np.random.random([img_size,img_size]))
            >>> Permuted_H = np.dot(Perm,Hcomplete)
            >>> Hsub = Permuted_H[:nb_measurements,:]
            >>> Forward_OP_pos = Forward_operator_pos(Hsub, Perm)
            >>> x = torch.tensor(np.random.random([batch_size,img_size]), dtype=torch.float)
            >>> y = Forward_OP_pos(x)
            >>> print(x.shape)
            >>> print(y.shape)
            torch.Size([100, 1024])
            torch.Size([100, 400]) 
        """
        # input x is a set of images with shape (b*c, N)
        # output is a set of measurement vectors with shape (b*c, M)
        
        # compute 1/2(H+1)x = 1/2 HX + 1/2 1x
        x = super().forward(x) + x.sum(dim=1,keepdim=True).expand(-1, self.M)
        x *= 0.5
        
        return x
    
# ==================================================================================
class Forward_operator_shift_had(Forward_operator_shift):
# ==================================================================================
    def __init__(self, Hsub, Perm):           
        super().__init__(Hsub, Perm)
    
    def inverse(self, x: torch.tensor, n: Union[None, int] = None) -> torch.tensor:
        r""" Inverse transform such that :math:`x = \frac{1}{N}H_{sub}y`.
        
        Args:
            :math:`x`: Batch of measurements.
            
        Shape:
            - Input: :math:`(b*c, N)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`N` the number of measurements.
            - Output: :math:`(b*c, N)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`N` the number of reconstructed. pixels.
            
        Example:
            >>> img_size = 64*64
            >>> M = 1024
            >>> batch_size = 10
            >>> Hcomplete = walsh_matrix(img_size)
            >>> Hsub = Hcomplete[:M,:]
            >>> Perm = np.array(np.random.random([img_size,img_size]))
            >>> FO_Shift = Forward_operator_shift(Hsub, Perm)
            >>> x = torch.tensor(np.random.random([batch_size,img_size]), dtype=torch.float)
            >>> y = FO_Shift(x) # y: belongs to Hadamard domain, which is square matrix
            >>> # as Hsub is not square, a completion by padding is proposed to implement the inverse recosntruction
            >>> y_pad = F.pad(y, (img_size-(M+1),0), "constant", 0)
            >>> FO_Shift_Had = Forward_operator_shift_had(Hsub, Perm)
            >>> x_reconstruct = FO_Shift_Had.inverse(y_pad) 
            >>> print(x.shape)
            >>> print(x_reconstruct.shape)
            torch.Size([10, 4096])
            torch.Size([10, 4096])          
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
