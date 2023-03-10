# -*- coding: utf-8 -*-


#%% PseudoInverse
import numpy as np
import torch
from spyrit.core.meas import HadamSplit
from spyrit.core.noise import NoNoise
from spyrit.core.prep import SplitPoisson 
from spyrit.core.recon import PseudoInverse

# EXAMPLE 1
# constructor
recon_op = PseudoInverse()

# forward
Ord = np.random.random([32,32])
meas_op = HadamSplit(400, 32, Ord)
y = torch.rand([85,400], dtype=torch.float)  

x = recon_op(y, meas_op)
print(x.shape)

# EXAMPLE 2
# constructor
M = 64
H = 64
B = 1

Ord = np.random.random([H,H])
meas_op = HadamSplit(M, H, Ord)
noise_op = NoNoise(meas_op)
split_op = SplitPoisson(1.0, M, H**2)
recon_op = PseudoInverse()

x = torch.FloatTensor(B,H**2).uniform_(-1, 1)
y = noise_op(x)
m = split_op(y, meas_op)
z = recon_op(m, meas_op)
print(z.shape)
print(torch.linalg.norm(x - z)/torch.linalg.norm(x))

#%% PseudoInverseStore
import numpy as np
import torch
from spyrit.core.meas import LinearRowSplit
from spyrit.core.noise import NoNoise
from spyrit.core.prep import SplitRowPoisson
from spyrit.core.recon import PseudoInverseStore
from spyrit.misc.walsh_hadamard import walsh_matrix


# EXAMPLE 1
# constructor
H_pos = np.random.rand(24,64)
H_neg = np.random.rand(24,64)
meas_op = LinearRowSplit(H_pos,H_neg)
recon_op = PseudoInverseStore(meas_op)

# forward
x = torch.rand([10,24,92], dtype=torch.float)
y = recon_op(x)
print(y.shape)

# EXAMPLE 2
# constructor
M = 63
N = 64
B = 1

H = walsh_matrix(N)
H_pos = np.where(H>0,H,0)[:M,:]
H_neg = np.where(H<0,-H,0)[:M,:]
meas_op = LinearRowSplit(H_pos,H_neg)
noise_op = NoNoise(meas_op)
split_op = SplitRowPoisson(1.0, M, 92)
recon_op = PseudoInverseStore(meas_op)

# forward
x = torch.FloatTensor(B,N,92).uniform_(-1, 1)
y = noise_op(x)
m = split_op(y, meas_op)
z = recon_op(m)
print(z.shape)
print(torch.linalg.norm(x - z)/torch.linalg.norm(x))

#%% PinvNet
import numpy as np
import torch
from spyrit.core.meas import HadamSplit
from spyrit.core.noise import NoNoise
from spyrit.core.prep import SplitPoisson 
from spyrit.core.recon import PseudoInverse, PinvNet

# EXAMPLE
# constructor
B, C, H, M = 10, 1, 64, 64**2
Ord = np.ones((H,H))
meas = HadamSplit(M, H, Ord)
noise = NoNoise(meas)
prep = SplitPoisson(1.0, M, H**2)
recnet = PinvNet(noise, prep) 

# forward
x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
z = recnet(x)
print(z.shape)
print(torch.linalg.norm(x - z)/torch.linalg.norm(x))

# meas2img
x = torch.rand(B*C,2*M)
z = recnet.meas2img(x)
print(z.shape)

# acquire
x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
z = recnet.acquire(x)
print(z.shape)

# reconstruct
x = torch.rand((B*C,2*M), dtype=torch.float)
z = recnet.reconstruct(x)
print(z.shape)

#%% TikhonovMeasurementPriorDiag
import numpy as np
import torch
from spyrit.core.recon import TikhonovMeasurementPriorDiag
from spyrit.core.meas import HadamSplit

B, H, M = 85, 32, 512

# constructor
sigma = np.random.random([H**2, H**2])
recon = TikhonovMeasurementPriorDiag(sigma, M)

# forward
Ord = np.ones((H,H))
meas = HadamSplit(M, H, Ord)
y = torch.rand([B,M], dtype=torch.float)  
x_0 = torch.zeros((B, H**2), dtype=torch.float)
var = torch.zeros((B, M), dtype=torch.float)

x = recon(y, x_0, var, meas)
print(x.shape)

#%% DCNet
import numpy as np
import torch
from spyrit.core.recon import DCNet
from spyrit.core.meas import HadamSplit
from spyrit.core.noise import NoNoise
from spyrit.core.prep import SplitPoisson 
from spyrit.core.meas import HadamSplit

B, C, H, M = 10, 1, 64, 64**2//2

# constructor
Ord = np.ones((H,H))
meas = HadamSplit(M, H, Ord)
noise = NoNoise(meas)
prep = SplitPoisson(1.0, M, H**2)
sigma = np.random.random([H**2, H**2])
recnet = DCNet(noise,prep,sigma)

# forward
x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
z = recnet(x)
print(z.shape)

# reconstruct
x = torch.rand((B*C,2*M), dtype=torch.float)
z = recnet.reconstruct(x)
print(z.shape)