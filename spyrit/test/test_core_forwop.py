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
forward_op = Linear(H)

# forward
x = torch.rand([10,1000], dtype=torch.float)
y = forward_op(x)
print('forward:', y.shape)

# adjoint
x = torch.rand([10,400], dtype=torch.float)
y = forward_op.adjoint(x)
print('adjoint:', y.shape)

# get_mat
H = forward_op.get_mat()
print('get_mat:', H.shape)

# pinv
y = torch.rand([85,400], dtype=torch.float)
x = forward_op.pinv(y)
print('pinv:', x.shape)

#%% Test LinearSplit
from spyrit.core.Forward_Operator import LinearSplit

# constructor
H = np.array(np.random.random([400,1000]))
forward_op = LinearSplit(H)

# forward
x = torch.rand([10,1000], dtype=torch.float)
y = forward_op(x)
print('Forward:', y.shape)

# forward_H
x = torch.rand([10,1000], dtype=torch.float)
y = forward_op.forward_H(x)
print('Forward_H:', y.shape)

# adjoint
x = torch.rand([10,400], dtype=torch.float)
y = forward_op.adjoint(x)
print('Adjoint:', y.shape)

# get_mat
H = forward_op.get_mat()
print('Measurement matrix:', H.shape)

# pinv
y = torch.rand([85,400], dtype=torch.float)
x = forward_op.pinv(y)
print('Pinv:', x.shape)

#%% Test HadamSplit
from spyrit.core.Forward_Operator import HadamSplit

# constructor
H = np.array(np.random.random([400,32*32]))
Perm = np.random.random([32*32,32*32])
forward_op = HadamSplit(H, Perm, 32, 32)

# forward
x = torch.rand([10,32*32], dtype=torch.float)
y = forward_op(x)
print('Forward:', y.shape)

# forward_H
x = torch.rand([10,32*32], dtype=torch.float)
y = forward_op.forward_H(x)
print('Forward_H:', y.shape)

# adjoint
x = torch.rand([10,400], dtype=torch.float)
y = forward_op.adjoint(x)
print('Adjoint:', y.shape)

# get_mat
H = forward_op.get_mat()
print('Measurement matrix:', H.shape)

# pinv
y = torch.rand([85,400], dtype=torch.float)
x = forward_op.pinv(y)
print('Pinv:', x.shape)

# inverse
y = torch.rand([85,32*32], dtype=torch.float)
x = forward_op.inverse(y)
print('Inverse:', x.shape)

#%% Test LinearRowSplit
from spyrit.core.Forward_Operator import LinearRowSplit

# constructor
H_pos = np.random.rand(24,64)
H_neg = np.random.rand(24,64)
forward_op = LinearRowSplit(H_pos,H_neg)

# forward
x = torch.rand([10,64,92], dtype=torch.float)
y = forward_op(x)
print(y.shape)

# forward_H
x = torch.rand([10,64,92], dtype=torch.float)
y = forward_op(x)
print(y.shape)