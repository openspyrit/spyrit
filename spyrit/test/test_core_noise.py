# -*- coding: utf-8 -*-
import numpy as np
import torch
from spyrit.core.forwop import Linear, LinearSplit, LinearRowSplit, HadamSplit
from spyrit.core.noise import Acquisition

# constructor
H = np.random.random([400,32*32])
linear_op = Linear(H)
linear_acq = Acquisition(linear_op)

# forward
x = torch.rand([10,32*32], dtype=torch.float)
y = linear_acq(x)
print(y.shape)

# forward with HadamSplit
Perm = np.random.random([32*32,32*32])
split_op = HadamSplit(H, Perm, 32, 32)
split_acq = Acquisition(split_op)

y = split_acq(x)
print(y.shape)