
r"""
1.1. Pseudoinverse solution from linear measurements
======================
This tutorial shows how to simulate data and perform image reconstruction. 
The measurement operator is a Hadamard matrix with positive coefficients. 
Note that this matrix can be replaced with the desired matrix. Undersampled 
measurements are simulated by selecting the undersampling factor. 

"""

# import matplotlib.pyplot as plt
# 
# from spyrit.core.prep import DirectPoisson
# from spyrit.core.recon import PinvNet
# from spyrit.core.meas import Linear, HadamSplit
# from spyrit.core.noise import NoNoise, Poisson


# %%
# Load a batch of images
#-----------------------

###############################################################################
# Images :math:`x` for training expect values in [-1,1]. The images are normalized
# using the :func:`transform_gray_norm` function.

import os
from spyrit.misc.statistics import transform_gray_norm
import torchvision
import torch
from spyrit.misc.disp import imagesc

h = 64            # image size hxh 
i = 1             # Image index (modify to change the image) 
spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, '../images')


# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=h)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 7)

x, _ = next(iter(dataloader))
print(f'Shape of input images: {x.shape}')

# Select image
x = x[i:i+1,:,:,:]
x = x.detach().clone()
b,c,h,w = x.shape

# plot
x_plot = x.view(-1,h,h).cpu().numpy() 
imagesc(x_plot[0,:,:], r'$x$ in [-1, 1]')

# %% 
# Define a measurement operator
#------------------------------

###############################################################################
# We consider the case where the measurement matrix is the positive
# component of a Hadamard matrix, which if often used in single-pixel imaging.
# First, we compute a full Hadamard matrix that computes the 2D transforme of an
# image of size :attr:`h` and take its positive part.

from spyrit.misc.walsh_hadamard import walsh2_matrix
import numpy as np

F = walsh2_matrix(h)
F = np.where(F>0, F, 0)

###############################################################################
# Next we subsample the rows of the measurement matrix to simulate an 
# accelerated acquisition. For this, we use the 
# :func:`spyrit.misc.sampling.Permutation_Matrix` function 
# that returns a :attr:`h*h`-by-:attr:`h*h` permutation matrix from a 
# :attr:`h`-by-:attr:`h` sampling maps that indicates the location of the most 
# relevant coefficients in the transformed domain.
#
# To keep the low-frequency Hadamard coefficients, we choose a sampling map 
# with ones in the top left corner and zeros elsewhere.

import math 

und = 4                # undersampling factor
M = h**2 // und        # number of measurements (undersampling factor = 4)

Sampling_map = np.ones((h,h))
M_xy = math.ceil(M**0.5)
Sampling_map[:,M_xy:] = 0
Sampling_map[M_xy:,:] = 0

imagesc(Sampling_map, 'low-frequency sampling map')

###############################################################################
# After permutation of the full Hadamrd matrix, we keep only its first 
# :attr:`M` rows

from spyrit.misc.sampling import Permutation_Matrix

Perm = Permutation_Matrix(Sampling_map)
F = Perm@F 
H = F[:M,:]

print(f"Shape of the measurement matrix: {H.shape}")

###############################################################################
# Then, we instantiate a :class:`spyrit.core.meas.Linear` measurement operator

from spyrit.core.meas import Linear
meas_op = Linear(H, pinv=True)      

# %% 
# Noiseless case
#------------------------------

###############################################################################
# In the noiseless case, we consider a :class:`spyrit.core.noise.NoNoise` noise
# operator, together with a :class:`spyrit.core.prep.DirectPoisson` 
# preprocessing operator with :math:`\alpha` = 1, which correct only for the 
# image normalisation in [0,1] (see `tuto_acquisition_operators`).

from spyrit.core.prep import DirectPoisson
from spyrit.core.noise import NoNoise


noise = NoNoise(meas_op)        
prep = DirectPoisson(1.0, meas_op) # To undo the "NoNoise" operator

# Simulate measurements 
y = noise(x.view(b*c,h*w))
print(f'Shape of raw measurements: {y.shape}')

###############################################################################
# To display the subsampled measurement vector as an image in the transformed 
# domain, we use the :func:`spyrit.misc.sampling.meas2img2` function

# plot
from spyrit.misc.disp import add_colorbar, noaxis
from spyrit.misc.sampling import meas2img

y_plot = y.detach().numpy().squeeze()
y_plot = meas2img(y_plot, Sampling_map)
print(f'Shape of the raw measurement image: {y_plot.shape}')

imagesc(y_plot, 'Raw measurements')


###############################################################################
# Preprocessed measurements corresponding to an image in [-1,1]
m = prep(y)
print(f'Shape of the preprocessed measurements: {m.shape}')

# plot
m_plot = m.detach().numpy().squeeze()
m_plot = meas2img(m_plot, Sampling_map)
print(f'Shape of the preprocessed measurement image: {m_plot.shape}')

imagesc(m_plot, 'Preprocessed measurements')

# %% 
# PinvNet Network 
# ---------------------
#
# [UPDATE !!]  PinvNet allows to perform image reconstruction using the pseudoinverse. 
# *spyrit.core.recon.PinvNet* includes the measurement operator, 
# the noise model and reconstruction. 
# Measurements can be obtained as 
#   y = pinv_net.acquire(x)
# Alternatively, the measurements can be obtained as
#   y = noise(x)
#
# The reconstruction can be obtained as
#   z = pinv_net.reconstruct(y)
# or as 
#   z = pinv_net(x)       


from spyrit.core.recon import PinvNet

pinv_net = PinvNet(noise, prep)

# measurements and images
with torch.no_grad():
    y = pinv_net.acquire(x)
    z = pinv_net.reconstruct(y)
#z = pinv_net(x)

# reshape
x_plot = x.view(-1,h,h).cpu().numpy() 
z_plot = z.view(-1,h,h).cpu().numpy()
z_plot[0,0,0] = 0.0

# plot
imagesc(z_plot[0,:,:], 'Reconstructed image with PinvNet')


# %% 
# Poisson-corrupted measurement
#------------------------------

###############################################################################
# Here again, we consider a :class:`spyrit.core.noise.NoNoise` noise
# operator, together with a :class:`spyrit.core.prep.DirectPoisson` 
# preprocessing operator (see `tuto_acquisition_operators`).

alpha = 100  # maximum number of photons in the image

from spyrit.core.noise import Poisson
noise = Poisson(meas_op, alpha)        
prep = DirectPoisson(alpha, meas_op) # To undo the "NoNoise" operator

# Simulate measurements 
y = noise(x.view(b*c,h*w))
print(f'Shape of raw measurements: {y.shape}')

###############################################################################
# To display the subsampled measurement vector as an image in the transformed 
# domain, we use the :func:`spyrit.misc.sampling.meas2img2` function

# plot
from spyrit.misc.sampling import meas2img

y_plot = y.detach().numpy().squeeze()
y_plot = meas2img(y_plot, Sampling_map)
print(f'Shape of the raw measurement image: {y_plot.shape}')

imagesc(y_plot, 'Raw measurements')


###############################################################################
# Preprocessed measurements corresponding to an image in [-1,1] [!! The range of values look weird !!]
m = prep(y)
print(f'Shape of the preprocessed measurements: {m.shape}')

# plot
m_plot = m.detach().numpy().squeeze()
m_plot = meas2img(m_plot, Sampling_map)
print(f'Shape of the preprocessed measurement image: {m_plot.shape}')

imagesc(m_plot, 'Preprocessed measurements')


###############################################################################
# Postprocessing can be added as a last layer of PinvNet, as shown in the 
# next tutorial.

# %% 
# PinvNet Network 
# ---------------------
#
# [UPDATE !!] PinvNet allows to perform image reconstruction using the pseudoinverse. 
# *spyrit.core.recon.PinvNet* includes the measurement operator, 
# the noise model and reconstruction. 
# Measurements can be obtained as 
#   y = pinv_net.acquire(x)
# Alternatively, the measurements can be obtained as
#   y = noise(x)
#
# The reconstruction can be obtained as
#   z = pinv_net.reconstruct(y)
# or as 
#   z = pinv_net(x)       

pinv_net = PinvNet(noise, prep)

# measurements and images
with torch.no_grad():
    y = pinv_net.acquire(x)
    z = pinv_net.reconstruct(y)
#z = pinv_net(x)

# reshape
x_plot = x.view(-1,h,h).cpu().numpy() 
z_plot = z.view(-1,h,h).cpu().numpy()
z_plot[0,0,0] = 0.0

# plot
imagesc(z_plot[0,:,:], 'Reconstructed image with PinvNet')