

r"""
.. _tuto_pseudoinverse_cnn_linear:
03. Pseudoinverse solution + CNN denoising from linear measurements
======================
This tutorial shows how to simulate measurements and perform image reconstruction 
using PinvNet (Linear net pseudoinverse) with CNN denoising. 
The measurement operator is chosen as a Hadamard matrix with positive coefficients. 
Note that this matrix can be replaced any the desired matrix. 
"""


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
# After permutation of the full Hadamard matrix, we keep only its first 
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
# In the noiseless case, we consider the :class:`spyrit.core.noise.NoNoise` noise
# operator

from spyrit.core.noise import NoNoise

noise = NoNoise(meas_op)        

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

imagesc(y_plot, 'Raw measurements (no noise)')

###############################################################################
# We now compute and plot the preprocessed measurements corresponding to an 
# image in [-1,1]
# 
# .. note::
#    
#       Using :class:`spyrit.core.prep.DirectPoisson` with :math:`\alpha` = 1 
#       allows to compensate for the image normalisation achieved by 
#       :class:`spyrit.core.noise.NoNoise`.

from spyrit.core.prep import DirectPoisson
prep = DirectPoisson(1.0, meas_op) # "Undo" the NoNoise operator

m = prep(y)
print(f'Shape of the preprocessed measurements: {m.shape}')

# plot
m_plot = m.detach().numpy().squeeze()
m_plot = meas2img(m_plot, Sampling_map)
print(f'Shape of the preprocessed measurement image: {m_plot.shape}')

imagesc(m_plot, 'Preprocessed measurements (no noise)')

# %% 
# PinvNet Network 
# ---------------

###############################################################################
# We consider the :class:`spyrit.core.recon.PinvNet` class that reconstructs an
# image by computing the pseudoinverse solution, which is fed to a neural 
# network denoiser. To compute the pseudoinverse solution only, the denoiser  
# can be set to the identity operator 

from spyrit.core.recon import PinvNet
pinv_net = PinvNet(noise, prep, denoi=torch.nn.Identity())

###############################################################################
# or equivalently
pinv_net = PinvNet(noise, prep)

###############################################################################
# Then, we reconstruct the image from the measurement vector :attr:`y` using the 
# :func:`~spyrit.core.recon.PinvNet.reconstruct` method

x_rec = pinv_net.reconstruct(y)

# plot
x_plot = x_rec.squeeze().cpu().numpy() 
imagesc(x_plot, 'Pseudoinverse reconstruction (no noise)')

# %%
# Removing artefacts with a CNN
#
# ----------------

###############################################################################
# Artefacts can be removed by selecting a neural network denoiser 
# (last layer of PinvNet). We select a simple CNN using the 
# :class:`spyrit.core.nnet.ConvNet` class. 

from spyrit.misc.disp import imagesc
from spyrit.core.nnet import ConvNet, Unet
from spyrit.core.train import load_net

# Define PInvNet with ConvNet denoising layer
denoi = ConvNet()
pinv_net_cnn = PinvNet(noise, prep, denoi)

# Send to GPU if available 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pinv_net_cnn = pinv_net_cnn.to(device)

###############################################################################
# As an example, we use a simple ConvNet that has been pretrained using STL-10 dataset.   

# Load pretrained model
try:
    # Download weights
    import gdown
    url_cnn = 'https://drive.google.com/file/d/1iGjxOk06nlB5hSm3caIfx0vy2byQd-ZC/view?usp=drive_link'
    model_cnn_path = "./model"

    if os.path.exists(model_cnn_path) is False:
        os.mkdir(model_cnn_path)
        print(f'Created {model_cnn_path}')

    model_cnn_path = os.path.join(model_cnn_path, 'dc-net_unet_imagenet_var_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07_light')
    gdown.download(url_cnn, f'{model_cnn_path}.pth', quiet=False,fuzzy=True)

    model_path = "./model/pinv-net_cnn_stl10_N0_1_N_64_M_1024_epo_1_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07"
    load_net(model_cnn_path, pinv_net_cnn, device, False)
    print(f'Model {model_path} loaded.')
except:
    print(f'Model {model_path} not found!')


# We now reconstruct the image using PinvNet with pretrained CNN denoising
with torch.no_grad():
    x_rec_cnn = pinv_net_cnn.reconstruct(y.to(device))
    x_rec_cnn = pinv_net_cnn(x.to(device))

# reshape
x_plot = x_rec_cnn.squeeze().cpu().numpy() 
imagesc(x_plot, f'PinvNet ConvNet image after training for 1 epoch')

###############################################################################
# In the next tutorial, we will show how to train the CNN denoiser from scratch.

