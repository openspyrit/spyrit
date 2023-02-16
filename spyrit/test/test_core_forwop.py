# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:04:11 2023

@author: ducros
"""


#%% 

import torch
import numpy as np
#from spyrit.misc.walsh_hadamard import walsh_matrix





#%% Test Linear
from spyrit.core.Forward_Operator import Linear

# constructor
H = np.array(np.random.random([400,1000]))
linop = Linear(H)

# forward
x = torch.rand([10,32*32], dtype=torch.float)
y = linop(x)
print('Output shape of forward:', y.shape)

# adjoint
x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
y = linop.adjoint(x)
print('Output shape of adjoint:', y.shape)

# get_mat
H = linop.pinv()
print('Shape of the measurement matrix:', H.shape)

# pinv


#%% Test LinearSplit
from spyrit.core.Forward_Operator import LinearSplit

# constructor
H = np.array(np.random.random([400,32*32]))
linop = LinearSplit(H)

# forward
x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
y = linop(x)
print('Output shape of forward:', y.shape)

# adjoint
x = torch.rand([10,400], dtype=torch.float)
y = linop.adjoint(x)
print('Output shape of adjoint:', y.shape)

# get_mat
H = linop.get_mat()
print('Shape of the measurement matrix:', H.shape)

#%% Test HadamSplit
from spyrit.core.Forward_Operator import HadamSplit

# constructor
H = np.array(np.random.random([400,32*32]))
linop = HadamSplit(H)

#%% Test LinearRowSplit
from spyrit.core.Forward_Operator import LinearRowSplit

# constructor
H_pos = np.random.rand(24,64)
H_neg = np.random.rand(24,64)
linop = LinearRowSplit(H_pos,H_neg)

# forward
x = torch.rand([10,64,92], dtype=torch.float)
y = linop(x)
print(y.shape)

# forward_H
x = torch.rand([10,64,92], dtype=torch.float)
y = linop(x)
print(y.shape)

