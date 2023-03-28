# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:19:24 2023

@author: ducros
"""
#%% Test DirectPoisson
from spyrit.core.meas import Linear
from spyrit.core.prep import DirectPoisson
import numpy as np
import torch

# constructor and forward
x = torch.rand([10,400], dtype=torch.float)
H = np.random.random([400,32*32])
meas_op =  Linear(H)
prep_op = DirectPoisson(1.0, meas_op)
m = prep_op(x)
print(m.shape)

# variance
x = torch.rand([10,400], dtype=torch.float)
v = prep_op.sigma(x)
print(v.shape)

# denormalize_expe
x = torch.rand([10, 1, 32,32], dtype=torch.float)
beta = 9*torch.rand([10])
y = prep_op.denormalize_expe(x, beta, 32, 32)
print(y.shape)

#%% Test SplitPoisson
import torch
import numpy as np
from spyrit.core.meas import LinearSplit, HadamSplit
from spyrit.core.prep import SplitPoisson

# constructor
split_op = SplitPoisson(10, 400, 32*32)

# forward with LinearSplit
x = torch.rand([10,2*400], dtype=torch.float)
H = np.random.random([400,32*32])

# forward
meas_op =  LinearSplit(H)
m = split_op(x, meas_op)
print(m.shape)

# forward with HadamSplit
Perm = np.random.random([32*32,32*32])
meas_op = HadamSplit(H, Perm, 32, 32)
m = split_op(x, meas_op)
print(m.shape)

# forward_expe
m, alpha = split_op.forward_expe(x, meas_op)
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
v = split_op.sigma_from_image(x, meas_op)
print(v.shape)

# denormalize_expe
x = torch.rand([10, 1, 32,32], dtype=torch.float)
beta = 9*torch.rand([10,1])
y = split_op.denormalize_expe(x, beta, 32, 32)
print(y.shape)

#%% Test SplitRowPoisson
from spyrit.core.meas import LinearRowSplit
from spyrit.core.prep import SplitRowPoisson

# constructor
split_op = SplitRowPoisson(2.0, 24, 64)

# forward with LinearRowSplit
x = torch.rand([10,48,64], dtype=torch.float)
H_pos = np.random.random([24,64])
H_neg = np.random.random([24,64])
meas_op = LinearRowSplit(H_pos, H_neg)

# forward
m = split_op(x, meas_op)
print(m.shape)