
r"""
05. Acquisition operators 2 - Split measurements
==========================

.. _tuto_acquisition_split_measurements:
This tutorial is a continuation of the :ref:`Acquisition operators tutorial <tuto_acquisition_operators>` 
for single-pixel imaging, which showed how to simulate linear measurements using the 
:class:`spyrit.core` submodule (based on three classes :class:`spyrit.core.meas`, 
:class:`spyrit.core.noise`, and :class:`spyrit.core.prep`). 
This tutorial extends the previous case to Hadamard patterns and introduces 
split measurements in order to handle negative measurements. 
"""

import numpy as np
import os
from spyrit.misc.disp import imagesc
import matplotlib.pyplot as plt


# %%
# Load a batch of images
# -----------------------------------------------------------------------------

###############################################################################
# Images :math:`x` for training neural networks expect values in [-1,1]. The images are normalized
# using the :func:`transform_gray_norm` function.

from spyrit.misc.statistics import transform_gray_norm
import torchvision
import torch

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
# The measurement and noise operators
# -----------------------------------------------------------------------------

############################################################################### 
# Noise operators are defined in the :mod:`~spyrit.core.noise` module. A noise
# operator computes the following three steps sequentially: 
#   1. Normalization of the image :math:`x` with values in [-1,1] to get an 
#      image :math:`\tilde{x}=\frac{x+1}{2}` in [0,1], as it is required for measurement simulation
#   2. Application of the measurement model, i.e., computation of :math:`P\tilde{x}`
#   3. Application of the noise model
# 
# .. math::
#       y \sim \texttt{Noise}(P\tilde{x}) = \texttt{Noise}\left(\frac{P(x+1)}{2}\right).
#
# The normalization is usefull when considering distributions such
# as the Poisson distribution that are defined on positive values. 

# %% 
# Split measurement operator and no noise
# -----------------------------------------------------------------------------
# .. _split_measurements:

###############################################################################
# .. math::
#       y = P\tilde{x}= \begin{bmatrix} H_{+} \\ H_{-} \end{bmatrix} \tilde{x}.

###############################################################################
# Hadamard split measurement operator is defined in the :class:`spyrit.core.meas.HadamSplit` class. 
# It computes linear measurements from incoming images, where :math:`P` is a 
# linear operator (matrix) with positive entries and :math:`\tilde{x}` is a vectorized image. 
# The class relies on a matrix :math:`H` with 
# shape :math:`(M,N)` where :math:`N` represents the number of pixels in the 
# image and :math:`M \le N` the number of measurements. The matrix :math:`P` 
# is obtained by splitting the matrix :math:`H` where 
# :math:`H_{+} = \max(0,H)`, :math:`H_{-} = \max(0,-H)`, and 
# :math:`H = H_{+}-H_{-}`.

###############################################################################
# Then, we simulate an accelerated acquisition by subsampling the measurement matrix 
# by retaining only the first :math:`M` rows of 
# a permuted Hadamard matrix :math:`\textrm{Perm} H`, where :math:`\textrm{Perm}` is a 
# permutation matrix with shape with shape :math:`(M,N)` and :math:`H` is a 
# "full" Hadamard matrix with shape :math:`(N,N)` 
# (see :ref:`tutorial on pseudoinverse solution <tuto_pseudoinverse_linear>`).
# The permutation matrix :math:`\textrm{Perm}` is obtained from the ordering matrix 
# :math:`\textrm{Ord}` with shape :math:`(h,h)`. This is all handled internally 
# by the :class:`spyrit.core.meas.HadamSplit` class.

###############################################################################
# .. note::
#   Note that the positive component of a Hadamard matrix has been previously introduced  
#   :ref:`here <hadamard_positive>` to simulate linear measurements. 
#   In this case, we could proceed as with other commonly used linear operators. 

###############################################################################
# We compute the measurement, noise and preprocessing operators and then 
# simulate a noiseless measurement vector :math:`y`.

# We consider the noiseless case handled 
# by the :class:`spyrit.core.noise.NoNoise` class.
M = 64*64 // 4       # number of measurements (here, 1/4 of the pixels)

from spyrit.core.meas import HadamSplit   
from spyrit.core.noise import NoNoise
from spyrit.misc.sampling import meas2img2

Ord = np.ones((h,h))                
meas_op = HadamSplit(M, h, Ord)
nonoise_op = NoNoise(meas_op) # noiseless

x = x.view(b*c,h*w)  # vectorized image
print(f'Shape of vectorized image: {x.shape}')
y_noiseless = nonoise_op(x)  # noiseless measurement vector
print(f'Shape of simulated measurements y: {y_noiseless.shape}')

# %% 
# Same example with Poisson noise
# -----------------------------------------------------------------------------

###############################################################################
# We now consider Poisson noise, i.e., a noisy measurement vector given by
#
# .. math::
#       y \sim \mathcal{P}(\alpha P \tilde{x}),
#
# where :math:`\alpha` is a scalar value that represents the maximum image intensity
# (in photons). The larger :math:`\alpha`, the higher the signal-to-noise ratio.
#

###############################################################################
# We consider the :class:`spyrit.core.noise.Poisson` class, set :math:`\alpha`
# to 100 photons, and simulate a noisy measurement vector. 

from spyrit.core.noise import Poisson
from spyrit.misc.disp import add_colorbar, noaxis
from spyrit.misc.disp import imagecomp

alpha = 100.0 # number of photons
noise_op = Poisson(meas_op, alpha) 
y_noisy = noise_op(x) # a noisy measurement vector


# %% 
# Full-covariance matrix
# -----------------------------------------------------------------------------

###############################################################################
# We have previously considered a unit Covariance matrix, i.e.,
# image pixels are assumed to be independent. Results can be improved by
# considering a full-covariance matrix, i.e., image pixels are assumed to be
# correlated. We consider a full-covariance matrix that has been obtained 
# from a set of natural images.

###############################################################################
# Frist, we download the covariance matrix and load it.

import girder_client

# api Rest url of the warehouse
url='https://pilot-warehouse.creatis.insa-lyon.fr/api/v1'

# Generate the warehouse client
gc = girder_client.GirderClient(apiUrl=url)

# Download the covariance matrix and mean image
data_folder = './stat/'
dataId_list = [
        '63935b624d15dd536f0484a5', # for reconstruction (imageNet, 64)
        '63935a224d15dd536f048496', # for reconstruction (imageNet, 64)
        ]
cov_name = './stat/Cov_64x64.npy'

for dataId in dataId_list:
    myfile = gc.getFile(dataId)
    gc.downloadFile(dataId, data_folder + myfile['name'])

print(f'Created {data_folder}') 

try:
    Cov  = np.load(cov_name)
    print(f"Cov matrix {cov_name} loaded")
except:
    Cov = np.eye(h*h)
    print(f"Cov matrix {cov_name} not found! Set to the identity")
    
###############################################################################
# We define the order matrix :math:`Ord` from the full covariance, and then the  
# measurement and noise operators.
from spyrit.misc.statistics import Cov2Var

Ord_cov = Cov2Var(Cov)
meas_cov_op = HadamSplit(M, h, Ord_cov)
noise_cov_op = Poisson(meas_cov_op, alpha) 

# Finally we simulate a noisy measurement vector.
y_noisy_cov = noise_cov_op(x) # a noisy measurement vector


# %% 
# The preprocessing operator measurements for split measurements
# -----------------------------------------------------------------------------

###############################################################################
# We compute the preprocessing operators for the three cases considered above,  
# using the :mod:`spyrit.core.prep` module. As previously introduced, 
# a preprocessing operator applies to the noisy measurements in order to 
# to compensate for the scaling factors that appear in the measurement or noise operators:  
# 
# .. math::
#       m = \texttt{Prep}(y),

###############################################################################
# We consider the :class:`spyrit.core.prep.SplitPoisson` class that intends 
# to "undo" the :class:`spyrit.core.noise.Poisson` class, for split measurements, by compensating for
#   * the scaling that appears when computing Poisson-corrupted measurements
#   * the affine transformation to get images in [0,1] from images in [-1,1]
#
# For this, it computes
#
# .. math::
#       m = \frac{2(y_+-y_-)}{\alpha} - P\mathbb{1},
#     
# where :math:`y_+=H_+\tilde{x}` and :math:`y_-=H_-\tilde{x}`. 
# This in handled internally by the :class:`spyrit.core.prep.SplitPoisson` class.

###############################################################################
# We consider first preprocessing the measurements corrupted by Poisson noise
from spyrit.core.prep import SplitPoisson

alpha = 100.0 # number of photons
prep_noisy_op = SplitPoisson(alpha, meas_op) 
m_noisy = prep_noisy_op(y_noisy) 

###############################################################################
# Similarly, we can preprocess the noiseless measurement by setting :math:`\alpha` to 1.
prep_noiseless_op = SplitPoisson(1.0, meas_op) 
m_noiseless = prep_noiseless_op(y_noiseless)

###############################################################################
# Finally, we can preprocess the noisy measurement for full covariance
prep_noisy_cov_op = SplitPoisson(alpha, meas_cov_op) 
m_noisy_cov = prep_noisy_cov_op(y_noisy_cov)

###############################################################################
# We can now plot the three measurement vectors

# Plot the three measurement vectors
m_plot = m_noisy.numpy()   
m_plot = meas2img2(m_plot.T, Ord)
m_plot = np.moveaxis(m_plot,-1, 0)
m_plot_max = np.max(m_plot[0,:,:])
m_plot_min = np.min(m_plot[0,:,:])

m_plot2 = m_noiseless.numpy()   
m_plot2 = meas2img2(m_plot2.T, Ord)
m_plot2 = np.moveaxis(m_plot2,-1, 0)

m_plot3 = m_noisy_cov.numpy()   
m_plot3 = meas2img2(m_plot3.T, Ord_cov)
m_plot3 = np.moveaxis(m_plot3,-1, 0)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
im1=ax1.imshow(m_plot[0,:,:], cmap='gray')
ax1.set_title(r'Noiseless $m$', fontsize=20)
noaxis(ax1)
add_colorbar(im1, 'bottom', size="20%")

im2=ax2.imshow(m_plot2[0,:,:], cmap='gray', vmin=m_plot_min, vmax=m_plot_max)
ax2.set_title(r'Noisy $m$', fontsize=20)
noaxis(ax2)
add_colorbar(im2, 'bottom', size="20%")

im3=ax3.imshow(m_plot3[0,:,:], cmap='gray', vmin=m_plot_min, vmax=m_plot_max)
ax3.set_title(r'Noisy $m$ (full Cov)', fontsize=20)
noaxis(ax3)
add_colorbar(im3, 'bottom', size="20%")

# %%
# PinvNet network 
# -----------------------------------------------------------------------------

###############################################################################
# We recontruct with the :class:`spyrit.core.recon.PinvNet` class the three cases 
# and plot results side by side.

from spyrit.core.recon import PseudoInverse
from spyrit.misc.disp import add_colorbar, noaxis
recon_op = PseudoInverse()

z_noiseless = recon_op(m_noiseless, meas_op)
z_noisy = recon_op(m_noisy, meas_op)
z_noisy_cov = recon_op(m_noisy_cov, meas_cov_op)

# Plot
x_plot = x.view(-1,h,h).numpy() 
z_plot_noiseless = z_noiseless.view(-1,h,h).numpy() 
z_plot_noisy = z_noisy.view(-1,h,h).numpy() 
z_plot_noisy_cov = z_noisy_cov.view(-1,h,h).numpy() 

# sphinx_gallery_thumbnail_number = 3
f, axs = plt.subplots(2, 2, figsize=(10,10))
im1=axs[0,0].imshow(x_plot[0,:,:], cmap='gray')
axs[0,0].set_title('Ground-truth image')
noaxis(axs[0,0])
add_colorbar(im1, 'bottom')

im2=axs[0,1].imshow(z_plot_noiseless[0,:,:], cmap='gray')
axs[0,1].set_title('Reconstruction noiseless')
noaxis(axs[0,1])
add_colorbar(im2, 'bottom')

im3=axs[1,0].imshow(z_plot_noisy[0,:,:], cmap='gray')
axs[1,0].set_title('Reconstruction noisy')
noaxis(axs[1,0])
add_colorbar(im3, 'bottom')

im4=axs[1,1].imshow(z_plot_noisy_cov[0,:,:], cmap='gray')
axs[1,1].set_title('Reconstruction noisy (full Cov)')
noaxis(axs[1,1])
add_colorbar(im4, 'bottom')

plt.show()

###############################################################################
# .. note::
#    
#       Note that reconstructed images are pixelized as pixels when using a unit covariance matrix  
#       while they are smooth when using a full covariance matrix. 
#       Another way to further improve results is to include a nonlinear post-processing step, 
#       which we will consider in a future tutorial. 