r"""
01. Tutorial Row 1D
======================
This tutorial focuses on Bayesian inversion, a special type of inverse problem
that aims at incorporating prior information in terms of model and data
probabilities in the inversion process.
"""

from spyrit.core.meas import LinearRowSplit
from spyrit.core.prep import SplitRowPoisson
from spyrit.misc.walsh_hadamard import walsh_matrix
from spyrit.misc.statistics import Cov2Var, data_loaders_stl10, transform_gray_norm
from spyrit.misc.disp import imagesc

import numpy as np
import os
import torch
import torchvision

###############################################################################
# Let's start by creating our images

M=24
N=64
B = 10                          # Batch size

# A batch of images
spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, '../images')
print(imgs_path)

transform = transform_gray_norm(img_size=N)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = min(B, len(dataset)))


#dataloaders = data_loaders_stl10('../../../data', img_size=N, batch_size=10)  
x, _ = next(iter(dataloader))

###############################################################################
# And now measure

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

###############################################################################
# Note that here we have been able to compute a sample posterior covariance
# from its estimated samples. By displaying it we can see  how both the overall
# variances and the correlation between different parameters have become
# narrower compared to their prior counterparts.

