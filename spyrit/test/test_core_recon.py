# -*- coding: utf-8 -*-


#%%
import numpy as np
import torch
from spyrit.core.meas import HadamSplit
from spyrit.core.recon import PseudoInverse

# constructor
recon_op = PseudoInverse()

# forward
Ord = np.random.random([32,32])
meas_op = HadamSplit(400, 32, Ord)
y = torch.rand([85,400], dtype=torch.float)  

x = recon_op(y, meas_op)
print(x.shape)

#%%
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

#%%
from spyrit.core.recon import TikhonovMeasurementPriorDiag

# constructor
sigma = np.random.random([32*32, 32*32])
recon_op = TikhonovMeasurementPriorDiag(sigma, 400)

# forward
H = np.random.random([400,32*32])
Perm = np.random.random([32*32,32*32])
meas_op =  HadamSplit(H, Perm, 32, 32)
y = torch.rand([85,400], dtype=torch.float)  

x_0 = torch.zeros((85, 32*32), dtype=torch.float)
var = torch.zeros((85, 400), dtype=torch.float)

x = recon_op(y, x_0, var, meas_op)
print(x.shape)