r"""
07. PnP DC-DRUNet for split measurements
==========================
.. _tuto_dcdrunet_split_measurements:
This tutorial shows how to perform image reconstruction with plug-and-play (PnP) DC-DRUNet 
(data completion with pretrained DRUNet denoising network) for single-pixel imaging. 
DC-DRUNet builds from the PnP network DRUNet, which is a denoising network 
that has been pretrained for a wide range of noise levels and admits the noise level 
as an input. Thus, it requires no training while providing state-of-the-art postprocessing 
performance! 

As in previous tutorials, we consider split Hadamard operator and poisson noise 
(see :ref:`Acquisition - split measurements <tuto_acquisition_split_measurements>`).  

"""


###############################################################################
# .. note::
#    
#       DRUNet has been taken from https://github.com/cszn/DPIR
#       Deep Plug-and-Play Image Restoration (DPIR) toolbox
#       June 2023
#
#       Citation:
#
#       @article{zhang2021plug,
#         title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
#         author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
#         journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
#         volume={44},
#         number={10},
#         pages={6360-6376},
#         year={2021}}
#        @inproceedings{zhang2017learning,
#          title={Learning Deep CNN Denoiser Prior for Image Restoration},
#          author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
#          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
#          pages={3929--3938},
#          year={2017}}

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
# Forward operators for split measurements
# -----------------------------------------------------------------------------

############################################################################### 
# We consider noisy split measurements for a Hadamard operator and a 
# “variance subsampling” strategy that preserves the coefficients with the 
# largest variance, obtained from a previously estimated covariance matrix 
# (for more details, refer to :ref:`Acquisition - split measurements <tuto_acquisition_split_measurements>`).  

###############################################################################
# First, we download the covariance matrix and load it.

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

try:
    for dataId in dataId_list:
        myfile = gc.getFile(dataId)
        gc.downloadFile(dataId, data_folder + myfile['name'])

    print(f'Created {data_folder}') 

    Cov  = np.load(cov_name)
    print(f"Cov matrix {cov_name} loaded")
except:
    Cov = np.eye(h*h)
    print(f"Cov matrix {cov_name} not found! Set to the identity")

###############################################################################
# We define the measurement, noise and preprocessing operators and then 
# simulate a noiseless measurement vector :math:`y`. As in the previous tutorial,
# we simulate an accelerated acquisition by subsampling the measurement matrix 
# by retaining only the first :math:`M` rows of a Hadamard matrix :math:`\textrm{Perm} H`. 

from spyrit.core.meas import HadamSplit   
from spyrit.core.noise import Poisson
from spyrit.misc.sampling import meas2img2
from spyrit.misc.statistics import Cov2Var
from spyrit.core.prep import SplitPoisson

# Measurement parameters
M = 64*64 // 4      # Number of measurements (here, 1/4 of the pixels)
alpha = 100.0       # number of photons

# Measurement and noise operators
Ord = Cov2Var(Cov)
meas_op = HadamSplit(M, h, Ord)
noise_op = Poisson(meas_op, alpha) 
prep_op = SplitPoisson(alpha, meas_op) 

# Vectorize image
x = x.view(b*c,h*w)  
print(f'Shape of vectorized image: {x.shape}')

# Measurements
y = noise_op(x)     # a noisy measurement vector
m = prep_op(y)      # preprocessed measurement vector

m_plot = m.detach().numpy()
m_plot = meas2img2(m_plot.T, Ord)
imagesc(m_plot, r'Measurements $m$')

# %%
# DC-DRUNet network 
# -----------------------------------------------------------------------------

###############################################################################
# We use DCDRUNet from the :class:`spyrit.core.recon.DCDRUNet` class. 
# It is a denoised completion reconstruction network based on DRUNet wich allows 
# to denoise an image with any noise level by concatenating a noise level map to the input. 
# The class :class:`spyrit.core.recon.DCDRUNet` builds from the class :class:`spyrit.core.recon.DCNet` 
# introduced in the previous :ref:`DCNet tutorial <tuto_dcnet_split_measurements>`. 
# The definition of the DRUNet network is given in the :class:`spyrit.external.drunet.UNetRes` class, 
# provided in the submodule :mod:`spyrit.external.drunet`. 
# In order to use the DRUNet network, we define it and then we download and load its pretrained weights. 

###############################################################################
# We download the pretrained weights of the DRUNet denoiser, call the DRUNet network and 
# load the pretrained weights.

from spyrit.external.drunet import UNetRes as drunet
import gdown

# Download weights
model_drunet_path = './model'
url_drunet = 'https://drive.google.com/file/d/1oSsLjPPn6lqtzraFZLZGmwP_5KbPfTES/view?usp=drive_link'

if os.path.exists(model_drunet_path) is False:
    os.mkdir(model_drunet_path)
    print(f'Created {model_drunet_path}')

model_drunet_path = os.path.join(model_drunet_path, 'drunet_gray.pth')
gdown.download(url_drunet, model_drunet_path, quiet=False,fuzzy=True)

# Define denoising network
n_channels = 1                   # 1 for grayscale image    
denoi_drunet = drunet(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',                     
            downsample_mode="strideconv", upsample_mode="convtranspose")  

# Load pretrained model
try:       
    denoi_drunet.load_state_dict(torch.load(model_drunet_path), strict=True)       
    print(f'Model {model_drunet_path} loaded.')
except:
    print(f'Model {model_drunet_path} not found!')
    load_drunet = False

denoi_drunet.eval()         
for k, v in denoi_drunet.named_parameters():             
    v.requires_grad = False  
print(sum(map(lambda x: x.numel(), denoi_drunet.parameters())) )  

###############################################################################
# We define the DCDRUNet network by providing the measurement, noise and preprocessing operators, 
# the covariance matrix, the denoising network and the noise level :attr:`noise_level`, 
# which is expected to be in [0, 255]. The larger the noise level, the higher the denoising.
# The noise level is not required as it can be set later on. 

from spyrit.core.recon import DCDRUNet

noise_level_1 = 7
dcdrunet = DCDRUNet(noise_op, prep_op, Cov, denoi_drunet, noise_level=noise_level_1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dcdrunet = dcdrunet.to(device)

###############################################################################
# We reconstruct the image from the measurement vector :math:`m`.

with torch.no_grad():
    # reconstruct from raw measurements
    z_dcdrunet = dcdrunet.reconstruct(y.to(device))  

###############################################################################
# We can set another noise level and reconstruct another image.

from spyrit.misc.disp import add_colorbar, noaxis

# Set noise level
noise_level_2 = 3
dcdrunet.set_noise_level(noise_level_2)

with torch.no_grad():
    # reconstruct from raw measurements
    z_dcdrunet_2 = dcdrunet.reconstruct(y.to(device))  

# Plot the two reconstructions
x_plot = z_dcdrunet.view(-1,h,h).cpu().numpy() 
x_plot2 = z_dcdrunet_2.view(-1,h,h).cpu().numpy() 

# sphinx_gallery_thumbnail_number = 3
f, axs = plt.subplots(1, 2, figsize=(10, 5))
im1=axs[0].imshow(x_plot[0,:,:], cmap='gray')
axs[0].set_title(f'DCDRUNet (n map={noise_level_1})', fontsize=16)
noaxis(axs[0])
add_colorbar(im1, 'bottom')

im2=axs[1].imshow(x_plot2[0,:,:], cmap='gray')
axs[1].set_title(f'DCDRUNet (n map={noise_level_2})', fontsize=16)
noaxis(axs[1])
add_colorbar(im2, 'bottom')


# %%
# DRUNet denoising
# -----------------------------------------------------------------------------

###############################################################################
# Alternatively, we can reconstruct with DCNet and apply DRUNet in a second step, 
# which is equivalent to DCDRUNet but it is more involved. 

from spyrit.core.recon import DCNet
from spyrit.external.drunet import uint2single, single2tensor4

# Define DCNet
dcnet = DCNet(noise_op, prep_op, Cov) 
dcnet = dcnet.to(device)

# Set the device for DRUNet
denoi_drunet = denoi_drunet.to(device)

# 1st step - Reconstruction
with torch.no_grad():
    z_dcnet = dcnet.reconstruct(y.to(device))  # reconstruct from raw measurements

# 2nd step - Denoising
# Convert to [0,1]
x_sample = 0.5*(z_dcnet[0,0,:,:] + 1).cpu().numpy()

# Create noise-level map and concatenate to the image
noise_level_3 = 7
x_sample = uint2single(255*x_sample)
x_sample = single2tensor4(x_sample[:,:,np.newaxis])
x_sample = torch.cat((x_sample, torch.FloatTensor([noise_level_3/255.]).repeat(1, 1, x_sample.shape[2], x_sample.shape[3])), dim=1)        
x_sample = x_sample.to(device)

with torch.no_grad():
    z_dcnet_den = denoi_drunet(x_sample)

###############################################################################
# We plot all results side by side.

from spyrit.misc.disp import add_colorbar, noaxis

x_plot = x.view(-1,h,h).cpu().numpy()    
x_plot2 = z_dcnet.view(-1,h,h).cpu().numpy() 
x_plot3 = z_dcnet_den.view(-1,h,h).cpu().numpy() 
x_plot4 = z_dcdrunet.view(-1,h,h).cpu().numpy() 

f, axs = plt.subplots(2, 2, figsize=(10,10))
im1=axs[0,0].imshow(x_plot[0,:,:], cmap='gray')
axs[0,0].set_title('Ground-truth image', fontsize=16)
noaxis(axs[0,0])
add_colorbar(im1, 'bottom')

im2=axs[0,1].imshow(x_plot2[0,:,:], cmap='gray')
axs[0,1].set_title('DCNet (without denoising)', fontsize=16)
noaxis(axs[0,1])
add_colorbar(im2, 'bottom')

im3=axs[1,0].imshow(x_plot3[0,:,:], cmap='gray')
axs[1,0].set_title(f'1) DCNet. 2) DRUNet (n map={noise_level_3})', fontsize=16)
noaxis(axs[1,0])
add_colorbar(im3, 'bottom')

im4=axs[1,1].imshow(x_plot4[0,:,:], cmap='gray')
axs[1,1].set_title(f'DCDRUNet (n map={noise_level_1})', fontsize=16)
noaxis(axs[1,1])
add_colorbar(im4, 'bottom')

plt.show()

###############################################################################
# We see the results by DCNet (without denoising), DCNet + DRUNet (with 
# PnP denoising network for noise level map equal to 7) and DCDRUNet (with noise level map equal to 7). 
# Note that the last two results are equivalent, as expected.

###############################################################################
# .. note::
#    
#       We refer to `spyrit-examples tutorials <http://github.com/openspyrit/spyrit-examples/tree/master/tutorial>`_
#       for a comparison of different solutions for split measurements (pinvNet, DCNet and DRUNet) that can be run in colab.


