# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:19:24 2023

@author: ducros
"""
#%% Test SplitPoisson
import torch
import numpy as np
from spyrit.core.Forward_Operator import LinearSplit
from spyrit.core.Preprocess import SplitPoisson

# constructor
split_op = SplitPoisson(10, 400, 32*32)

# forward with LinearSplit
x = torch.rand([10,2*400], dtype=torch.float)
H = np.random.random([400,32*32])
forward_op =  LinearSplit(H)

# forward
m = split_op(x, forward_op)
print(m.shape)

#%% Test SplitRowPoisson
from spyrit.core.Forward_Operator import LinearRowSplit
from spyrit.core.Preprocess import SplitRowPoisson

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