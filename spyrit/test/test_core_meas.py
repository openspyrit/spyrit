# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:04:11 2023

@author: ducros
"""


#%% 
import torch
import numpy as np

#%% Test Linear
from spyrit.core.meas import Linear

# constructor
H = np.array(np.random.random([400,1000]))
meas_op = Linear(H)

# forward
x = torch.rand([10,1000], dtype=torch.float)
y = meas_op(x)
print('forward:', y.shape)

# adjoint
x = torch.rand([10,400], dtype=torch.float)
y = meas_op.adjoint(x)
print('adjoint:', y.shape)

# get_mat
H = meas_op.get_mat()
print('get_mat:', H.shape)

#%% Test LinearSplit
from spyrit.core.meas import LinearSplit

# constructor
H = np.array(np.random.random([400,1000]))
meas_op = LinearSplit(H)

# forward
x = torch.rand([10,1000], dtype=torch.float)
y = meas_op(x)
print('Forward:', y.shape)

# forward_H
x = torch.rand([10,1000], dtype=torch.float)
y = meas_op.forward_H(x)
print('Forward_H:', y.shape)

# adjoint
x = torch.rand([10,400], dtype=torch.float)
y = meas_op.adjoint(x)
print('Adjoint:', y.shape)

# get_mat
H = meas_op.get_mat()
print('Measurement matrix:', H.shape)

#%% Test HadamSplit
from spyrit.core.meas import HadamSplit

# constructor
Ord = np.random.random([32,32])
meas_op = HadamSplit(400, 32, Ord)

# forward
x = torch.rand([10,32*32], dtype=torch.float)
y = meas_op(x)
print('Forward:', y.shape)

# forward_H
x = torch.rand([10,32*32], dtype=torch.float)
y = meas_op.forward_H(x)
print('Forward_H:', y.shape)

# adjoint
x = torch.rand([10,400], dtype=torch.float)
y = meas_op.adjoint(x)
print('Adjoint:', y.shape)

# get_mat
H = meas_op.get_mat()
print('Measurement matrix:', H.shape)

# pinv
y = torch.rand([85,400], dtype=torch.float)
x = meas_op.pinv(y)
print('Pinv:', x.shape)

# inverse
y = torch.rand([85,32*32], dtype=torch.float)
x = meas_op.inverse(y)
print('Inverse:', x.shape)

#%% Test LinearRowSplit
from spyrit.core.meas import LinearRowSplit

# constructor
H_pos = np.random.rand(24,64)
H_neg = np.random.rand(24,64)
meas_op = LinearRowSplit(H_pos,H_neg)

# forward
x = torch.rand([10,64,92], dtype=torch.float)
y = meas_op(x)
print(y.shape)

# forward_H
x = torch.rand([10,64,92], dtype=torch.float)
y = meas_op(x)
print(y.shape)