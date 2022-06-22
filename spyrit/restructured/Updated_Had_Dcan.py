# ==================================================================================
#from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch import poisson
from scipy.sparse.linalg import aslinearoperator
#from pylops_gpu import Diagonal, LinearOperator
#from pylops_gpu.optimization.cg import cg --- currently not working
#from pylops_gpu.optimization.leastsquares import NormalEquationsInversion

from ..misc.walsh_hadamard import walsh2_torch


# ==================================================================================
# Forward operators
# ==================================================================================
# ==================================================================================
class Forward_operator(nn.Module):
# ==================================================================================
# Faire le produit H*f sans bruit, linear (pytorch) 
    def __init__(self, Hsub):           
        # récupérer la méthode __init__ de la classe parent nn.Module et créer un objet de cette classe
        super().__init__()

        # instancier nn.linear        
        # Pmat --> (torch) --> Poids ()
        self.M = Hsub.shape[0];
        self.N = Hsub.shape[1] ;
        self.Hsub = nn.Linear(self.N, self.M, False); 
        self.Hsub.weight.data=torch.from_numpy(Hsub)
        self.Hsub.weight.data=self.Hsub.weight.data.float()
        self.Hsub.weight.requires_grad=False

        # adjoint
        self.Hsub_adjoint = nn.Linear(self.M, self.N, False)
        self.Hsub_adjoint.weight.data=torch.from_numpy(Hsub.transpose())
        self.Hsub_adjoint.weight.data = self.Hsub_adjoint.weight.data.float()
        self.Hsub_adjoint.weight.requires_grad = False
               
    def forward(self, x): # --> simule la mesure sous-chantillonnée
        """
        """
        # x.shape[b*c,N]
        x = self.Hsub(x)    
        return x

    def Forward_op(self,x):
        # x.shape[b*c,N]
        x = self.Hsub(x)    
        return x
    
    def adjoint(self,x):
        # x.shape[b*c,M]
        #Pmat.transpose()*f
        x = self.Hsub_adjoint(x)            
        return x

    def Mat(self):
        return self.Hsub.weight.data;

# ==================================================================================
class Split_Forward_operator(Forward_operator):
# ==================================================================================
# Faire le produit H*f sans bruit, linear (pytorch) 
    def __init__(self, Hsub):           
        # [H^+, H^-]
        super().__init__(Hsub)
        
        even_index = range(0,2*self.M,2);
        odd_index = range(1,2*self.M,2);

        H_pos = np.zeros(Hsub.shape);
        H_neg = np.zeros(Hsub.shape);
        H_pos[Hsub>0] = Hsub[Hsub>0];
        H_neg[Hsub<0] = -Hsub[Hsub<0];
        Hposneg = np.zeros((2*self.M,self.N));
        Hposneg[even_index,:] = H_pos;
        Hposneg[odd_index,:] = H_neg;
        
        self.Hpos_neg = nn.Linear(self.N, 2*self.M, False) 
        self.Hpos_neg.weight.data=torch.from_numpy(Hposneg)
        self.Hpos_neg.weight.data=self.Hpos_neg.weight.data.float()
        self.Hpos_neg.weight.requires_grad=False
              
    def forward(self, x): # --> simule la mesure sous-chantillonnée
        # x.shape[b*c,N]
        # output shape : [b*c, 2*M]
        x = self.Hpos_neg(x)    
        return x
#
## ==================================================================================
#class Split_Forward_operator_pylops(Split_Forward_operator):
## ==================================================================================
## Pylops compatible split forward operator 
#    def __init__(self, Hsub, device = "cpu"):           
#        # [H^+, H^-]
#        super().__init__(Hsub)
#        self.Op = LinearOperator(aslinearoperator(Hsub), device = device, dtype = torch.float32)
#

# ==================================================================================
class Split_Forward_operator_ft_had(Split_Forward_operator): # forward tranform hadamard
# ==================================================================================
# Forward operator with implemented inverse transform and a permutation matrix
    def __init__(self, Hsub, Perm):
        super().__init__(Hsub);
        self.Perm = nn.Linear(self.N, self.N, False)
        self.Perm.weight.data=torch.from_numpy(Perm.T)
        self.Perm.weight.data=self.Perm.weight.data.float()
        self.Perm.weight.requires_grad=False
 
    def inverse(self, x, n = None):
        # rearrange the terms + inverse transform
        # maybe needs to be initialised with a permutation matrix as well!
        # Permutation matrix may be sparsified when sparse tensors are no longer in
        # beta (as of pytorch 1.11, it is still in beta).
        
        # input - x - shape [b*c, N]
        # output - x - shape [b*c, N]
        b, N = x.shape
        x = self.Perm(x);
        if n is None:
            n = int(np.sqrt(N));
        x = x.view(b, 1, n, n);
        x = 1/self.N*walsh2_torch(x);#to apply the inverse transform
        x = x.view(b, N);
        return x;


# ==================================================================================
# Acquisition
# ==================================================================================
# ==================================================================================        
class Acquisition(nn.Module):
# ==================================================================================
#                      Forward multiply with Hadamard matrix H * f
# ==================================================================================
# exemple:
# on calcule Pmat --> on écrit H = Forward_operator(Pmat)
# acq = Acquisition(H)
# avec une image x, m = acq.forward(x) (ce qui est équivalent à m = acq(x)) Syntax Pytorch

    def __init__(self, FO):
        super().__init__()
        # FO = forward operator
        self.FO = FO
    
    def forward(self, x):
        # input x.shape - [b*c,h*w] - [b*c,N] 
        # output x.shape - [b*c,M] 
        #--Scale input image
        x = (x+1)/2; 
        x = self.FO.Forward_op(x); 
        # x is the product of Hsub-sampled*f ?
        return x

# ==================================================================================
class Bruit_Poisson_approx_Gauss(Acquisition):
# ==================================================================================    
    def __init__(self, alpha, FO):
        super().__init__(FO)
        self.alpha = alpha
        
    def forward(self, x):
        # Input shape [b*c, N]  
        # Output shape [b*c, 2*M]

        #--Scale input image      
        x = self.alpha*(x+1)/2;
        
        #--Acquisition
        x = self.FO(x);
        x = F.relu(x);# to remove small negative values that stem from numerical issue
        
        #--Measurement noise (Gaussian approximation of Poisson)
        x = x + torch.sqrt(x)*torch.randn_like(x);  
        return x  
    
# ==================================================================================
class Bruit_Poisson_Pytorch(Acquisition):
# ==================================================================================           
    def __init__(self, alpha, H):
        super().__init__(H)
        self.alpha = alpha

    def forward(self, x):
        # Input shape [b*c, N]  
        # Output shape [b*c, 2*M]

        #--Scale input image      
        x = self.alpha*(x+1)/2;
        
        #--Acquisition
        x = self.FO(x);
        x = F.relu(x);  
        
        #--Measurement noise imported from Pytorch
        x = poisson(x) 
        return x           
       
 
# ==================================================================================
# Preprocessing
# ==================================================================================
# ==================================================================================        
class Split_diag_poisson_preprocess(nn.Module):  
# ==================================================================================
    """
    computes m = (m_+-m_-)/N_0
    and also allows to compute var = 2*Diag(m_+ + m_-)/N0**2
    """
    def __init__(self, N0, M, N):
        super().__init__()
        self.N0 = N0;
                
        self.N = N;
        self.M = M;
        
        self.even_index = range(0,2*M,2);
        self.odd_index = range(1,2*M,2);


    def forward(self, x, FO):
        # Input shape [b*c,2*M]
        # Output shape [b*c,M]
        x = x[:,self.even_index] - x[:,self.odd_index]
        x = 2*x/self.N0 - FO.Forward_op(torch.ones(x.shape[0], self.N).to(x.device))
        return x
    
    def sigma(self, x):
        # Input shape (b*c, 2*M)
        # output shape (b*c, M)
        x = x[:,self.even_index] + x[:,self.odd_index]
        x = x/(4*self.N0**2)
        return x


# ==================================================================================
# Data consistency
# ==================================================================================
# ==================================================================================
class Pinv_orthogonal(nn.Module):
# ==================================================================================
    def __init__(self, FO):
        super().__init__()
        # FO = Forward Operator
        #-- Pseudo-inverse to determine levels of noise.
        self.FO = FO;
        
    def forward(self, x):
        # input (b*c, M)
        # output (b*c, N)
        x = (1/self.FO.N)*self.FO.adjoint(x);
        return x

# ==================================================================================
class learned_measurement_to_image(nn.Module):
# ==================================================================================
    def __init__(self, FO):
        super().__init__()
        # FO = Forward Operator
        self.FC = nn.Linear(FO.M, FO.N, True) # FC - fully connected
        
    def forward(self, x):
        # input (b*c, M)
        # output (b*c, N)
        x = self.FC(x);
        return x
 
# ==================================================================================
class gradient_step(nn.Module):
# ==================================================================================
    def __init__(self, FO, mu = 0.1):
        super().__init__()
        # FO = Forward Operator
        #-- Pseudo-inverse to determine levels of noise.
        self.FO = FO;
        self.mu = nn.Parameter(torch.tensor([mu], requires_grad=True)) #need device maybe?
        # if user wishes to keep mu constant, then he can change requires gard to false 
        
    def forward(self, x, x_0):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        # z = x_0 - mu*A^T(Ax_0-x)
        x = self.FO.Forward_op(x_0)-x;
        x = x_0 - self.mu*self.FO.adjoint(x);
        return x
 
# ==================================================================================
class Tikhonov_cg(nn.Module):
# ==================================================================================
    def __init__(self, FO, n_iter = 5, mu = 0.1, eps = 1e-6):
        super().__init__()
        # FO = Forward Operator - Works for ANY forward operator
        self.FO = FO;
        self.n_iter = n_iter
        self.mu = nn.Parameter(torch.tensor([float(mu)], requires_grad=True)) #need device maybe?
        # if user wishes to keep mu constant, then he can change requires gard to false 
        self.eps = eps;


    def A(self,x):
        return self.FO.Forward_op(self.FO.adjoint(x)) + self.mu*x;

    def CG(self, y, shape, device):
        x = torch.zeros(shape).to(device); 
        r = y - self.A(x);
        c = r.clone()
        kold = torch.sum(r * r)
        a = torch.ones((1));
        for i in range(self.n_iter): 
            if a>self.eps : # Necessary to avoid numerical issues with a = 0 -> a = NaN
                Ac = self.A(c)
                cAc =  torch.sum(c * Ac)
                a =  kold / cAc
                x += a * c
                r -= a * Ac
                k = torch.sum(r * r)
                b = k / kold
                c = r + b * c
                kold = k
        return x
        
    def forward(self, x, x_0):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        # n_step steps of Conjugate gradient to solve \|Ax-b\|^2 + mu \|x - x_0\|^2
        y = x-self.FO.Forward_op(x_0);
        x = self.CG(y, x.shape, x.device);
        x = x_0 + self.FO.adjoint(x)
        return x


# ==================================================================================
class Tikhonov_solve(nn.Module):
# ==================================================================================
    def __init__(self, FO, mu = 0.1):
        super().__init__()
        # FO = Forward Operator - Needs to be matrix-storing
        #-- Pseudo-inverse to determine levels of noise.
        self.FO = FO;
        self.mu = nn.Parameter(torch.tensor([float(mu)], requires_grad=True)) #need device maybe?
    
    def solve(self, x):
        A = self.FO.Mat()@torch.transpose(self.FO.Mat(), 0,1)+self.mu*torch.eye(self.FO.M); # Can precompute H@H.T to save time!
        A = A.view(1, self.FO.M, self.FO.M);
        A = A.repeat(x.shape[0],1, 1); # Not optimal in terms of memory, but otherwise the torch.linalg.solve doesn't work
        x = torch.linalg.solve(A, x);
        return x;

    def forward(self, x, x_0):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        
        # uses torch.linalg.solve [As of Pytorch 1.9 autograd supports solve!!]
        x = x - self.FO.Forward_op(x_0);
        x = self.solve(x);
        x = x_0 + self.FO.adjoint(x)
        return x

# ==================================================================================
class Orthogonal_Tikhonov(nn.Module):
# ==================================================================================
    def __init__(self, FO, mu = 0.1):
        super().__init__()
        # FO = Forward Operator
        #-- Pseudo-inverse to determine levels of noise.
        self.FO = FO;
        self.mu = nn.Parameter(torch.tensor([float(mu)], requires_grad=True)) #need device maybe?
        
    def forward(self, x, x_0):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        
        x = x - self.FO.Forward_op(x_0);
        #x = torch.div(x,(self.mu+1));
        x = x*(1/(self.FO.N+self.mu));# for hadamard, otherwise, line above
        x = self.FO.adjoint(x) + x_0;
        return x;


# ==================================================================================
class Generalised_Tikhonov_cg(nn.Module):# not inheriting from Tikhonov_cg because 
#                           the of the way var is called in CG
# ==================================================================================
    def __init__(self, FO, Sigma_prior, n_iter = 6, eps = 1e-6):
        super().__init__()
        # FO = Forward Operator - Works for ANY forward operator
        # if user wishes to keep mu constant, then he can change requires gard to false 
        self.FO = FO;
        self.n_iter = n_iter;

        self.Sigma_prior = nn.Linear(self.FO.N, self.FO.N, False); 
        self.Sigma_prior.weight.data=torch.from_numpy(Sigma_prior)
        self.Sigma_prior.weight.data=self.Sigma_prior.weight.data.float()
        self.Sigma_prior.weight.requires_grad=False
        self.eps = eps;


    def A(self,x, var):
        return self.FO.Forward_op(self.Sigma_prior(self.FO.adjoint(x))) + torch.mul(x,var); # the first part can be precomputed for optimisation

    def CG(self, y, var, shape, device):
        x = torch.zeros(shape).to(device); 
        r = y - self.A(x, var);
        c = r.clone()
        kold = torch.sum(r * r)
        a = torch.ones((1));
        for i in range(self.n_iter):
            if a > self.eps :
                Ac = self.A(c, var)
                cAc =  torch.sum(c * Ac)
                a =  kold / cAc
                x += a * c
                r -= a * Ac
                k = torch.sum(r * r)
                b = k / kold
                c = r + b * c
                kold = k
        return x
        
    def forward(self, x, x_0, var_noise):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # var_noise - input (b*c, M) - estimated variance of noise
        # z - output (b*c, N)
        # n_step steps of Conjugate gradient to solve 
        # \|Ax-b\|^2_{sigma_prior^-1} + \|x - x_0\|^2_{var_noise^-1}
        y = x-self.FO.Forward_op(x_0);
        x = self.CG(y, var_noise, x.shape, x.device);
        x = x_0 + self.Sigma_prior(self.FO.adjoint(x))
        return x


# ==================================================================================
class Generalised_Tikhonov_solve(nn.Module):
# ==================================================================================
    def __init__(self, FO, Sigma_prior):
        super().__init__()
        # FO = Forward Operator - Needs to be matrix-storing
        # -- Pseudo-inverse to determine levels of noise.
        self.FO = FO;
        self.Sigma_prior = nn.Parameter(\
                torch.from_numpy(Sigma_prior.astype("float32")), requires_grad=True)

    def solve(self, x, var):
        A = self.FO.Mat()@self.Sigma_prior@torch.transpose(self.FO.Mat(), 0,1)
        A = A.view(1, self.FO.M, self.FO.M);
        A = A.repeat(x.shape[0],1,1);# this could be precomputed maybe
        A += torch.diag_embed(var);
        x = torch.linalg.solve(A, x);
        return x;

    def forward(self, x, x_0, var_noise):
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        
        # uses torch.linalg.solve [As of Pytorch 1.9 autograd supports solve!!]
        # torch linal solve uses (I believe the LU decomposition of matrix A to
        # solve the linear system.

        x = x - self.FO.Forward_op(x_0);
        x = self.solve(x, var_noise);
        x = x_0 + torch.matmul(self.Sigma_prior,self.FO.adjoint(x).T).T
        return x


# ===========================================================================================
class Generalized_Orthogonal_Tikhonov(nn.Module): 
# ===========================================================================================   
    def __init__(self, FO, sigma_prior):
        super().__init__()
        # FO = Forward Operator - needs foward operator with defined inverse transform
        #-- Pseudo-inverse to determine levels of noise.
        self.FO = FO;
        self.comp = nn.Linear(self.FO.M, self.FO.N-self.FO.M, False)
        self.denoise_layer = Denoise_layer(self.FO.M);
        
        diag_index = np.diag_indices(self.FO.N);
        var_prior = sigma_prior[diag_index];
        var_prior = var_prior[:self.FO.M]

        self.denoise_layer.weight.data = torch.from_numpy(var_prior)
        self.denoise_layer.weight.data = self.denoise_layer.weight.data.float();
        self.denoise_layer.weight.requires_grad = False

        Sigma1 = sigma_prior[:self.FO.M,:self.FO.M];
        Sigma21 = sigma_prior[self.FO.M:,:self.FO.M];
        W = Sigma21@np.linalg.inv(Sigma1);
        
        self.comp.weight.data=torch.from_numpy(W)
        self.comp.weight.data=self.comp.weight.data.float()
        self.comp.weight.requires_grad=False
        
    def forward(self, x, x_0, var):
        # x - input (b*c, M) - measurement vector
        # var - input (b*c, M) - measurement variance
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        #

        x = x - self.FO.Forward_op(x_0);
        y1 = torch.mul(self.denoise_layer(var),x);
        y2 = self.comp(y1);

        y = torch.cat((y1,y2),-1);
        x = x_0+self.FO.inverse(y) 
        return x;

# ===========================================================================================
class Denoise_layer(nn.Module):
# ===========================================================================================
    r"""Applies a transformation to the incoming data: :math:`y = A^2/(A^2+x) `
    Args:
        in_features: size of each input sample
    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{in})`.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, 1)`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
    Examples::
        >>> m = Denoise_layer(30)
        >>> input = torch.randn(128, 30)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, in_features):
        super(Denoise_layer, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, 0, 2/math.sqrt(self.in_features))

    def forward(self, inputs):
        return tikho(inputs, self.weight)

    def extra_repr(self):
        return 'in_features={}'.format(self.in_features)

def tikho(inputs, weight):
    # type: (Tensor, Tensor) -> Tensor
    r"""
    Applies a transformation to the incoming data: :math:`y = A^2/(A^2+x)`.
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(in\_features)`
        - Output: :math:`(N, in\_features)`
    """
    sigma = weight**2;
    den = sigma + inputs;
    ret = sigma/den;
    return ret


# ==================================================================================
# Image Domain denoising layers
# ==================================================================================
# ===========================================================================================
class Unet(nn.Module):
# ===========================================================================================
    def __init__(self, in_channel=1, out_channel=1):
        super(Unet, self).__init__()
        #Descending branch
        self.conv_encode1 = self.contract(in_channels=in_channel, out_channels=16)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contract(16, 32)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contract(32, 64)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        #Bottleneck
        self.bottleneck = self.bottle_neck(64)
        #Decode branch
        self.conv_decode4 = self.expans(64,64,64)
        self.conv_decode3 = self.expans(128, 64, 32)
        self.conv_decode2 = self.expans(64, 32, 16)
        self.final_layer = self.final_block(32, 16, out_channel)
        
    
    def contract(self, in_channels, out_channels, kernel_size=3, padding=1):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    )
        return block
    
    def expans(self, in_channels, mid_channel, out_channels, kernel_size=3,padding=1):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=padding),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=kernel_size, stride=2,padding=padding, output_padding=1)
                    )

            return block
    

    def concat(self, upsampled, bypass):
        out = torch.cat((upsampled,bypass),1)
        return out
    
    def bottle_neck(self,in_channels, kernel_size=3, padding=1):
        bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=2*in_channels, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=2*in_channels, out_channels=in_channels, padding=padding),
            torch.nn.ReLU(),
            )
        return bottleneck
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel,padding=1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel,padding=1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    )
            return  block
    
    def forward(self,x):
        
        #Encode
        encode_block1 = self.conv_encode1(x)
        x = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(x)
        x = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(x)
        x = self.conv_maxpool3(encode_block3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decode
        x = self.conv_decode4(x)
        x = self.concat(x, encode_block3)
        x = self.conv_decode3(x)
        x = self.concat(x, encode_block2)
        x = self.conv_decode2(x)
        x = self.concat(x, encode_block1)
        x = self.final_layer(x)      
        return x
    
# ===========================================================================================
class ConvNet(nn.Module):
# ===========================================================================================
    def __init__(self):
        super(ConvNet,self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(64,32,kernel_size=1, stride=1, padding=0)),
                ('relu2', nn.ReLU()),
                ('conv3', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
                ]));
                
    def forward(self,x):
        x = self.convnet(x)
        return x
    
# ===========================================================================================
class DConvNet(nn.Module):  
# ===========================================================================================
    def __init__(self):
        super(DConvNet,self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1,64,kernel_size=9, stride=1, padding=4)),
                ('relu1', nn.ReLU()),
                ('BN1', nn.BatchNorm2d(64)),
                ('conv2', nn.Conv2d(64,64,kernel_size=1, stride=1, padding=0)),
                ('relu2', nn.ReLU()),
                ('BN2', nn.BatchNorm2d(64)),
                ('conv3', nn.Conv2d(64,32,kernel_size=3, stride=1, padding=1)),
                ('relu3', nn.ReLU()),
                ('BN3', nn.BatchNorm2d(32)),
                ('conv4', nn.Conv2d(32,1,kernel_size=5, stride=1, padding=2))
                ]));
    def forward(self,x):
        x = self.convnet(x)
        return x





# ==================================================================================
# Complete Reconstruction method
# ==================================================================================
# ===========================================================================================
class DC_Net(nn.Module):
# ===========================================================================================
    def __init__(self, in_channel=1, out_channel=1):




# DOES NOT WORK YET BECAUSE MatrixLinearOperator are not subscriptable
#
## ==================================================================================
#class Tikhonov_cg_pylops(nn.Module):
## ==================================================================================
#    def __init__(self, FO, n_iter = 5, mu = 0.1):
#        super().__init__()
#        # FO = Forward Operator - Needs to be pylops compatible!!
#        #-- Pseudo-inverse to determine levels of noise.
#        # Not sure about the support of learnable mu!!! (to be tested)
#        self.FO = FO;
#        self.mu = mu;
#        self.n_iter = n_iter
#
#    def A(self):
#        print(type(self.FO.Op))
#        # self.FO.Op.H NOT WORKING FOR NOW - I believe it's a bug, but here it isa
#        #
#        #File ~/.conda/envs/spyrit-env/lib/python3.8/site-packages/pylops_gpu/LinearOperator.py:336, in LinearOperator._adjoint(self)
#        #    334 def _adjoint(self):
#        #    335     """Default implementation of _adjoint; defers to rmatvec."""
#        #--> 336     shape = (self.shape[1], self.shape[0])
#        #    337     return _CustomLinearOperator(shape, matvec=self.rmatvec,
#        #    338                                  rmatvec=self.matvec,
#        #    339                                  dtype=self.dtype, explicit=self.explicit,
#        #    340                                  device=self.device, tocpu=self.tocpu,
#        #    341                                  togpu=self.togpu)
#        #
#        #TypeError: 'MatrixLinearOperator' object is not subscriptable
#        # Potentially needs to be improved
#        return self.FO.Op*self.FO.Op.T + self.mu*Diagonal(torch.ones(self.FO.M).to(self.FO.OP.device))
#        
#    def forward(self, x, x_0):
#        # x - input (b*c, M) - measurement vector
#        # x_0 - input (b*c, N) - previous estimate
#        # z - output (b*c, N)
#        
#        # Conjugate gradient to solve \|Ax-b\|^2 + mu \|x - x_0\|^2
#        #y = self.FO.Forward_op(x_0)-x;
#        #x,_ = cg(self.A(), y, niter = self.n_iter) #see pylops gpu conjugate gradient
#        #x = x_0 + self.FO.adjoint(x)
#        x = NormalEquationsInversion(Op = self.FO.Op, Regs = None, data = x, \
#                epsI = self.mu, x0 = x_0, device = self.FO.Op.device, \
#                **dict(niter = self.n_iter))
#        return x
#
# DOES NOT WORK YET BECAUSE MatrixLinearOperator are not subscriptable
#
## ==================================================================================
#class Generalised_Tikhonov_cg_pylops(nn.Module):
## ==================================================================================
#    def __init__(self, FO, Sigma_prior, n_steps):
#        super().__init__()
#        # FO = Forward Operator - pylops compatible! Does not allow to
#        # optimise the matrices Sigma_prior yet
#        self.FO = FO;
#        self.Sigma_prior = LinearOperator(aslinearoperator(Sigma_prior), self.FO.OP.device, dtype = self.FO.OP.dtype)
#
#    def A(self, var):
#        return self.FO.OP*self.Sigma_prior*self.FO.OP.H + Diagonal(var.to(self.FO.OP.device));
#        
#    def forward(self, x, x_0, var_noise):
#        # x - input (b*c, M) - measurement vector
#        # x_0 - input (b*c, N) - previous estimate
#        # z - output (b*c, N)
#        
#        # Conjugate gradient to solve \|Ax-b\|^2_Var_noise + \|x - x_0\|^2_Sigma_prior
#        y = self.FO.Forward_op(x_0)-x;
#        x,_ = cg(self.A(var_noise), y, niter = self.n_iter)
#        x = x_0 + self.Sigma_prior(self.FO.adjoint(x)) # to check that cast works well!!!
#        return x
#
