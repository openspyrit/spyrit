# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:19:24 2023

@author: ducros
"""
#%% Test SplitPoisson
import torch
import numpy as np
from spyrit.core.forwop import LinearSplit, HadamSplit
from spyrit.core.preproc import SplitPoisson

# constructor
split_op = SplitPoisson(10, 400, 32*32)

# forward with LinearSplit
x = torch.rand([10,2*400], dtype=torch.float)
H = np.random.random([400,32*32])

# forward
forward_op =  LinearSplit(H)
m = split_op(x, forward_op)
print(m.shape)

# forward with HadamSplit
Perm = np.random.random([32*32,32*32])
forward_op = HadamSplit(H, Perm, 32, 32)
m = split_op(x, forward_op)
print(m.shape)

# forward_expe
m, alpha = split_op.forward_expe(x, forward_op)
print(m.shape)
print(alpha.shape)

# sigma
x = torch.rand([10,2*400], dtype=torch.float)
v = split_op.sigma(x)
print(v.shape)

# set_expe
split_op.set_expe(gain=1.6)
print(split_op.gain)

# sigma_expe
v = split_op.sigma_expe(x)
print(v.shape)

# sigma_from_image
x = torch.rand([10,32*32], dtype=torch.float)
v = split_op.sigma_from_image(x, forward_op)
print(v.shape)

# denormalize_expe
x = torch.rand([10, 1, 32,32], dtype=torch.float)
beta = 9*torch.rand([10,1])
y = split_op.denormalize_expe(x, beta, 32, 32)
print(y.shape)

#%% Test SplitRowPoisson
from spyrit.core.forwop import LinearRowSplit
from spyrit.core.preproc import SplitRowPoisson

# constructor
split_op = SplitRowPoisson(2.0, 24, 64)

# forward with LinearRowSplit
x = torch.rand([10,48,64], dtype=torch.float)
H_pos = np.random.random([24,64])
H_neg = np.random.random([24,64])
forward_op = LinearRowSplit(H_pos, H_neg)

# forward
m = split_op(x, forward_op)
print(m.shape)