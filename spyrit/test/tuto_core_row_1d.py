from spyrit.core.meas import LinearRowSplit
from spyrit.core.prep import SplitRowPoisson
from spyrit.misc.walsh_hadamard import walsh_matrix
from spyrit.misc.statistics import data_loaders_stl10
from spyrit.misc.disp import imagesc

import numpy as np

M=24
N=64

# A batch of images
dataloaders = data_loaders_stl10('../../../data', img_size=N, batch_size=10)  
x, _ = next(iter(dataloaders['train']))

# Associated measurements
H = walsh_matrix(N)
H = H[:M]
H_pos = np.where(H>0, H, 0)
H_neg = np.where(H<0,-H, 0)

# Raw measurement
x = (x.view(-1,N,N) + 1)/2 
linop = LinearRowSplit(H_pos,H_neg)
y = linop(x)

# Split measurement
split = SplitRowPoisson(1.0, M, N)
m = split(x, linop)

# plot
imagesc(x[0,:,:])
imagesc(y[0,:,:])
imagesc(m[0,:,:])