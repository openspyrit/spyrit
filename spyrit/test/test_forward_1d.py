from spyrit.core.Forward_Operator import LinearRowSplit
from spyrit.misc.walsh_hadamard import walsh_matrix
from spyrit.misc.statistics import data_loaders_stl10
from spyrit.misc.disp import imagesc

import numpy as np

# A batch of images
dataloaders = data_loaders_stl10('../../../data', img_size=64, batch_size=10)  
x, _ = next(iter(dataloaders['train']))

# Associated measurements
H = walsh_matrix(64)
H = H[:24]
H_pos = np.where(H>0, H, 0)
H_neg = np.where(H<0,-H, 0)

A = LinearRowSplit(H_pos,H_neg)
y = A(x)

# plot
imagesc(x[0,0,:,:])
imagesc(y[0,0,:,:])