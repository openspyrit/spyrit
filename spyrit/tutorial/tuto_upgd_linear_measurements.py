r"""
10. Unrolled proximal gradient descent - linear measurements
==========================
.. _tuto_upgd_linear_measurements:
This tutorial shows how to perform image reconstruction with an unrolled proximal gradient 
descent (UPGD) network for linear measurements. An unrolled network is an end-to-end 
network that replicates a fix number of iterations of an splitting iterative method where the
proximal operator or denoising step is replaced by a denoising network. These methods are more robust 
than post-processing methods as they impose the data consistency constraint at each 
iteration. In addition, they are faster than the equivalent iterative method and 
generally lead to improved results as they learn an optimal proximal operator (image prior) 
from the data. The UPGD network is one of the simplest versions where data update consists 
on a simple gradient descent step. 

As in previous tutorials, we consider Hadamard matrix with positive coefficients, 
but this matrix can be replaced by any desired matrix 
(see :ref:`Pseudoinverse solution tutorial <tuto_pseudoinverse_linear>`).  

"""

# %%
# Load a batch of images
# -----------------------------------------------------------------------------

###############################################################################
# First, we load an image :math:`x` and normalized it to [-1,1], as in previous examples. 

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
# -----------------------------------------------------------------------------

###############################################################################
# We consider the case where the measurement matrix is the positive
# component of a Hadamard matrix and the sampling operator preserves only 
# the first :attr:`M` low-frequency coefficients 
# (see :ref:`Positive Hadamard matrix <hadamard_positive>` for full explantion).

import numpy as np
import math 
from spyrit.misc.sampling import Permutation_Matrix
from spyrit.misc.walsh_hadamard import walsh2_matrix

F = walsh2_matrix(h)
F = np.where(F>0, F, 0)
und = 4                # undersampling factor
M = h**2 // und        # number of measurements (undersampling factor = 4)

Sampling_map = np.ones((h,h))
M_xy = math.ceil(M**0.5)
Sampling_map[:,M_xy:] = 0
Sampling_map[M_xy:,:] = 0

Perm = Permutation_Matrix(Sampling_map)
F = Perm@F 
H = F[:M,:]
print(f"Shape of the measurement matrix: {H.shape}")

imagesc(Sampling_map, 'low-frequency sampling map')

###############################################################################
# Then, we instantiate a :class:`spyrit.core.meas.Linear` measurement operator

from spyrit.core.meas import Linear
meas_op = Linear(H, pinv=True)      

# %% 
# Noise and preprocessing operators
# -----------------------------------------------------------------------------

###############################################################################
# We consider Poisson noise, so we define measurement, noise and preprocessing operators 
# as in previous tutorial :ref:`tuto_acquisition_operators`.

from spyrit.core.noise import Poisson
from spyrit.core.prep import DirectPoisson

alpha = 100.0         # Noise level (noiseless)
noise_op = Poisson(meas_op, alpha=alpha)        
prep_op = DirectPoisson(alpha, meas_op) 

# Simulate measurements 
y = noise_op(x.view(b*c,h*w))
m = prep_op(y)

# %%
# UPGD network 
# -----------------------------------------------------------------------------

############################################################################### 
# The UPGD is updated iteratively as follows:
#
# .. math::
#       x_{k+1} = \mathcal{P}(x_{k}-\eta H^T(Hx_k-m)),  
#
# where :math:`\mathcal{P}` can be replaced by a denoising network. 
#
# UPGD is defined by the :class:`~spyrit.core.recon.UPGD` class, which inheritages 
# from :class:`~spyrit.core.recon.PinvNet`. It requires to set the number of unrolled iterations 
# :attr:`num_iter`, which is set to 6 by default, and the denoising network :attr:`denoi`. 
# We create two UPGD instances with different number of iterations.
#
# We define the denoising network as a small CNN, using the class :class:`spyrit.core.nnet.ConvNet`, 
# and then instantiate the UPGD network. Then, we download the pretrained weights  
# and load them into the network.

from spyrit.core.nnet import ConvNet
from spyrit.core.train import load_net, save_net
from spyrit.core.recon import UPGD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Denoising network
denoi = ConvNet()

# UPGD for 5 iterations
upgd_cnn = UPGD(noise_op, prep_op, denoi, num_iter = 5, split=False)
upgd_cnn = upgd_cnn.to(device)

# Load previously trained models
try:
    import gdown
    model_path = "./model"    
    if os.path.exists(model_path) is False:
        os.mkdir(model_path)
        print(f'Created {model_path}')

    url_upgd = ''
    name_net = 'upgd_cnn_stl10_N0_100_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_uit_2_la_1e-05_linmeas'

    model_upgd_path = os.path.join(model_path, name_net)

    # Download weights
    gdown.download(url_upgd, f'{model_upgd_path}.pth', quiet=False,fuzzy=True)

    # Load pretrained model
    load_net(model_upgd_path, upgd_cnn, device=device, strict=False)
except:
    print(f'Model not found!')

# Reconstruction
with torch.no_grad():
    z_upgd_cnn = upgd_cnn.reconstruct(y.to(device))  # reconstruct from raw measurements

# %%
# PinvNet network 
# -----------------------------------------------------------------------------

###############################################################################
# We reconstruct with the pseudo inverse using :class:`spyrit.core.recon.PinvNet` class 
# as in the previous tutorial. For this, we define the neural network and then perform the reconstruction.
from spyrit.core.recon import PinvNet

# Reconstruction with for Core module (linear net)
pinvnet = PinvNet(noise_op, prep_op)
pinvnet = pinvnet.to(device)

# With CNN denoising
pinvnet_cnn = PinvNet(noise_op, prep_op, denoi=denoi)
pinvnet_cnn = pinvnet_cnn.to(device)

# Reconstruction
with torch.no_grad():
    z_invnet = pinvnet.reconstruct(y.to(device))  # reconstruct from raw measurements
    z_invnet_cnn = pinvnet_cnn.reconstruct(y.to(device))  # reconstruct from raw measurements

###############################################################################
# We plot all results side by side.

from spyrit.misc.disp import add_colorbar, noaxis
import matplotlib.pyplot as plt

x_plot = x.view(-1,h,h).cpu().numpy()    
x_plot2 = z_invnet.view(-1,h,h).cpu().numpy() 
x_plot3 = z_invnet_cnn.view(-1,h,h).cpu().numpy() 
x_plot4 = z_upgd_cnn.view(-1,h,h).cpu().numpy() 

f, axs = plt.subplots(2, 2, figsize=(15,12))
im1=axs[0,0].imshow(x_plot[0,:,:], cmap='gray')
axs[0,0].set_title('Ground-truth image', fontsize=16)
noaxis(axs[0,0])
add_colorbar(im1, 'bottom')

im2=axs[0,1].imshow(x_plot2[0,:,:], cmap='gray')
axs[0,1].set_title(f'PinvNet + I', fontsize=16)
noaxis(axs[0,1])
add_colorbar(im2, 'bottom')

im5=axs[0,2].imshow(x_plot3[0,:,:], cmap='gray')
axs[0,2].set_title(f'PinvNet + CNN', fontsize=16)
noaxis(axs[0,2])
add_colorbar(im5, 'bottom')

im3=axs[1,0].imshow(x_plot4[0,:,:], cmap='gray')
axs[1,0].set_title(f'UPGD (CNN)', fontsize=16)
noaxis(axs[1,0])
add_colorbar(im3, 'bottom')

im4=axs[1,1].imshow(x_plot4[0,:,:], cmap='gray')
axs[1,1].set_title(f'UPGD (CNN) \n num_iter=6', fontsize=16)
noaxis(axs[1,1])
add_colorbar(im4, 'bottom')

plt.show()
