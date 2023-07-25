
r"""
1.0. Acquisition operators
======================
This tutorial shows how to simulate measurements using the :class:`spyrit.core`
submodule, which is based on three classes:

* **Measurement operators** compute linear measurements :math:`y = Hx` from 
  images :math:`x`, where :math:`H` is a linear operator (matrix) and :math:`x`
  is a vectorized image.

* **Noise operator** corrupts measurements :math:`y` with noise

* **Preprocessing operators** are typically used to process the noisy 
  measurements prior to reconstruction

"""

import numpy as np
import os
from spyrit.misc.disp import imagesc
import matplotlib.pyplot as plt

und = 2                 # undersampling factor
M = 32*32 // und        # number of measurements (undersampling factor = 2)
B = 10                  # batch size
alpha = 100             # number of mean photon counts
mode_noise = False      # noiseless or 'poisson'

spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, '../images')

# %%
# Load a batch of images
#-----------------------

###############################################################################
# Images :math:`x` for training expect values in [-1,1]. The images are normalized
# using the :func:`transform_gray_norm` function.

from spyrit.misc.statistics import transform_gray_norm
import torchvision
import torch

# A batch of images
#dataloaders = data_loaders_stl10('../../data', img_size=h, batch_size=10)  
#x, _ = next(iter(dataloaders['train']))
h = 64                  # image size hxh 
ind_img = 1             # Image index (modify to change the image) 

# N.B.: no view here compared to previous example
# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=h)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = min(B, len(dataset)))

# Select image
x0, _ = next(iter(dataloader))
x0 = x0[ind_img:6,:,:,:]
x = x0.detach().clone()
b,c,h,w = x.shape
#x = x.view(b*c,h*w)
print(f'Shape of incoming image (b*c,h*w): {x.view(b*c,h*w).shape}')


# plot
x_plot = x.view(-1,h,h).cpu().numpy() 
imagesc(x_plot[0,:,:], 'Ground-truth image normalized to [-1,1]')

# %% 
# A simple example: identity measurement matrix and no noise
#-----------------------------------------------------------

###############################################################################
# We  define a linear operator, :class:`~spyrit.core.meas.Linear` that 
# applies the measurement matrix to the image. We start with a basis example
# where  :math:`y = x`, i.e., where the measurement matrix :math:`H` is the identity
from spyrit.core.meas import Linear
meas_op_eye = Linear(np.eye(h*h), pinv=True) 

############################################################################### 
# Then, we define a noise operator. We start by considering the noiseless case 
# handled by :class:`~spyrit.core.NoNoise`. This operator normalizes 
# the image :math:`x` from [-1,1] to an image in [0,1], i.e.,
# 
# .. math::
#       \tilde{x}=\frac{x+1}{2}
#
# This will be usefull later on, when considering Poisson noise. The class is 
# constructed from a measurement operator (see the :mod:`~spyrit.core.meas` 
# submodule), such that the "noisy" measurements are then obtained as
#
# .. math::
#       y = H\tilde{x} = \frac{H(x+1)}{2}.

from spyrit.core.noise import NoNoise
noise_op_eye = NoNoise(meas_op_eye) 

###############################################################################
# We finally simulate the measurements that we visualise as an image
x = x.view(b*c,h*w)      # vectorized image
y_eye = noise_op_eye(x)  # noisy measurement vector

# plot
x_plot = y_eye.view(-1,h,h).cpu().numpy() 
imagesc(x_plot[0,:,:], r'Image $\tilde{x}$ in [0, 1]')

###############################################################################
# Note that the image is normalized between [0,1].


# %% 
# Same example with Poisson noise
#--------------------------------

###############################################################################
# We now consider Poisson noise, i.e., noisy measurement vector given by
#
# .. math::
#       \hat{x}_\alpha \sim \mathcal{P}(\alpha H \tilde{x}),
#
# where :math:`\alpha` is a scalar value that represents the image intensity
# (in photons). The larger :math:`\alpha`, the higher the signal-to-noise ratio.
# 
from spyrit.core.noise import Poisson

alpha = 10 # number of photons
noise_op_eye = Poisson(meas_op_eye, alpha) 

# noisy measurement vector
y_1 = noise_op_eye(x)  

# another noisy measurement vector
y_2 = noise_op_eye(x)  

# another noisy measurement vector  
noise_op_eye.alpha = 100
y_2 = noise_op_eye(x)  # noisy measurement vector

# plot
x_plot = y_eye.view(-1,h,h).cpu().numpy() 
imagesc(x_plot[0,:,:], r'Image $\tilde{x}$ in [0, 1]')

# plot
# f, axs = plt.subplots(3, 1)
# axs[0].set_title('Target measurement patterns')
# im = axs[0].imshow(y_1, cmap='gray') 
# add_colorbar(im, 'bottom')
# axs[0].get_xaxis().set_visible(False)

# axs[1].set_title('Experimental measurement patterns')
# im = axs[1].imshow(y_2, cmap='gray') 
# add_colorbar(im, 'bottom')
# axs[1].get_xaxis().set_visible(False)

# %% 
# The preprocessing operator
#---------------------------

###############################################################################
# We now discuss the preprocessing operator that allows to convert the 
# measurements *y* to 
#     measurements *m* for the original image *x*. For instance, using the 
#     operator *spyrit.core.prep.DirectPoisson(nn.Module)*, the measurements $m$ for $x$ are 
#     then obtained as
#
# .. math::
#       m=2y-H*I.
#     
#        
# Similarly, for the Poisson case, :math:`y=\alpha \mathcal{P}(H\tilde{x})` and 
# :math:`m=\frac{2y}{\alpha}-H\mathbf{I}`.
#
# Prior to reconstruction, images are normalized so :math:`\tilde{x}` in [0,1] 
# using *NoNoise(nn.Module)*. By defening a linear operator equal to the identity, 
# the measurements are then the normalized images.
#
# The measurement operator is a Hadamard matrix with positive coefficients. 
# Note that this matrix can be replaced with the desired matrix. Undersampled 
# measurements are simulated by selecting the undersampling factor. 