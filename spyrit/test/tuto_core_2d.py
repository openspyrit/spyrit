# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:16:27 2023

@author: ducros
"""
from spyrit.core.Forward_Operator import LinearSplit
from spyrit.core.Preprocess import SplitPoisson
from spyrit.misc.walsh_hadamard import walsh2_matrix
from spyrit.misc.statistics import data_loaders_stl10
from spyrit.misc.disp import imagesc

M=500
N=64

# A batch of images
dataloaders = data_loaders_stl10('../../../data', img_size=N, batch_size=10)  
x, _ = next(iter(dataloaders['train']))

# Associated measurements
H = walsh2_matrix(N)
H = H[:M]

# Raw measurement
x = (x.view(-1,N*N) + 1)/2 
linop = LinearSplit(H)
y = linop(x)

# Split measurement
split = SplitPoisson(1.0, M, N**2)
m = split(x, linop)

# reshape
x = x.view(-1,N,N) 
#y = y.view(-1,N,N)
#m = m.view(-1,N,N)

# plot
imagesc(x[0,:,:])
imagesc(y)
imagesc(m)