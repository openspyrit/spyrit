r"""
08. Unrolled proximal gradient descent - split measurements
==========================
.. _tuto_upgd_split_measurements:
This tutorial shows how to perform image reconstruction with an unrolled proximal gradient 
descent (UPGD) network for single-pixel imaging. An unrolled network is an end-to-end 
network that replicates a fix number of iterations of an splitting iterative method where the
proximal operator or denoising step is replaced by a denoising network. These methods are more robust 
than post-processing methods as they impose the data consistency constraint at each 
iteration. In addition, they are faster than the equivalent iterative method and 
generally lead to improved results as they learn an optimal proximal operator (image prior) 
from the data. The UPGD network is one of the simplest versions where data update consists 
on a simple gradient descent step. 

As in previous tutorials, we consider split Hadamard operator and poisson noise 
(see :ref:`Acquisition - split measurements <tuto_acquisition_split_measurements>`).  

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
# Forward operators for split measurements
# -----------------------------------------------------------------------------

############################################################################### 
# We consider noisy split measurements for a Hadamard operator and a full 
# covariance matrix to take into account the correlation between measurements 
# (for more details, we refer to :ref:`Acquisition - split measurements <tuto_acquisition_split_measurements>`).  

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
# We define the measurement, noise and preprocessing operators and then 
# simulate a measurement vector :math:`m` corrupted by Poisson noise. As in the previous tutorial,
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

# UPGD for 2 iterations
upgd_cnn_2it = UPGD(noise_op, prep_op, denoi, num_iter = 2, split=True)
upgd_cnn_2it = upgd_cnn_2it.to(device)

# UPGD for 6 iterations
upgd_cnn_6it = UPGD(noise_op, prep_op, denoi, num_iter = 6, split=True)
upgd_cnn_6it = upgd_cnn_6it.to(device)

# Load previously trained models
try:
    import gdown
    model_path = "./model"    
    if os.path.exists(model_path) is False:
        os.mkdir(model_path)
        print(f'Created {model_path}')

    url_upgd_2it = 'https://drive.google.com/file/d/1SKvolg1ICXDeQJmPS7ejcGKfPWny5RpS/view?usp=drive_link'
    name_net_2it = 'upgd_cnn_stl10_N0_100_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_uit_2_la_1e-05'

    url_upgd_6it = 'https://drive.google.com/file/d/1Oyn6UZPlpzWyYzQYJ8ZhSooXC4zdREid/view?usp=drive_link'
    name_net_6it = 'upgd_cnn_stl10_N0_100_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_uit_6_la_1e-05'

    model_upgd_2it_path = os.path.join(model_path, name_net_2it)
    model_upgd_6it_path = os.path.join(model_path, name_net_6it)

    # Download weights
    gdown.download(url_upgd_2it, f'{model_upgd_2it_path}.pth', quiet=False,fuzzy=True)
    gdown.download(url_upgd_6it, f'{model_upgd_6it_path}.pth', quiet=False,fuzzy=True)
    
    """
    model_upgd_path = './model/upgd_cnn_stl10_N0_100_N_64_M_1024_epo_2_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07'
    """

    # Load pretrained model
    load_net(model_upgd_2it_path, upgd_cnn_2it, device=device, strict=False)
    load_net(model_upgd_6it_path, upgd_cnn_6it, device=device, strict=False)
except:
    print(f'Model not found!')

# Reconstruction
with torch.no_grad():
    z_upgd_2it = upgd_cnn_2it.reconstruct(y.to(device))  # reconstruct from raw measurements
    z_upgd_6it = upgd_cnn_6it.reconstruct(y.to(device))  # reconstruct from raw measurements


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

# Reconstruction
with torch.no_grad():
    z_invnet = pinvnet.reconstruct(y.to(device))  # reconstruct from raw measurements


# %%
# DCNET network 
# -----------------------------------------------------------------------------

###############################################################################
# We compare the results with the DCNET + UNet network (see :ref:`DCNET tutorial <tuto_dcnet_split_measurements>`).

# Pretrained DC UNet (UNet denoising)
from spyrit.core.recon import DCNet
from spyrit.core.nnet import Unet
denoi = Unet()
dcnet_unet = DCNet(noise_op, prep_op, Cov, denoi)
dcnet_unet = dcnet_unet.to(device)

# Load previously trained model
try:
    import gdown

    # Download weights
    url_dcnet = 'https://drive.google.com/file/d/15PRRZj5OxKpn1iJw78lGwUUBtTbFco1l/view?usp=drive_link'
    name_net = 'dc-net_unet_stl10_N0_100_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07'
    model_path = "./model"    
    if os.path.exists(model_path) is False:
        os.mkdir(model_path)
        print(f'Created {model_path}')
    model_unet_path = os.path.join(model_path, name_net)
    gdown.download(url_dcnet, f'{model_path}.pth', quiet=False,fuzzy=True)

    # Load pretrained model
    load_net(model_path, dcnet_unet, device, False)
    print(f'Model {model_path} loaded.')
except:
    print(f'Model {model_path} not found!')
    load_unet = False

# Reconstruction
with torch.no_grad():
    z_dcnet_unet = dcnet_unet.reconstruct(y.to(device))  # reconstruct from raw measurements

###############################################################################
# We plot all results side by side.

from spyrit.misc.disp import add_colorbar, noaxis

x_plot = x.view(-1,h,h).cpu().numpy()    
x_plot2 = z_dcnet_unet.view(-1,h,h).cpu().numpy() 
x_plot3 = z_upgd_2it.view(-1,h,h).cpu().numpy() 
x_plot4 = z_upgd_6it.view(-1,h,h).cpu().numpy() 
x_plot5 = z_invnet.view(-1,h,h).cpu().numpy() 

f, axs = plt.subplots(2, 3, figsize=(15,12))
im1=axs[0,0].imshow(x_plot[0,:,:], cmap='gray')
axs[0,0].set_title('Ground-truth image', fontsize=16)
noaxis(axs[0,0])
add_colorbar(im1, 'bottom')

im2=axs[0,1].imshow(x_plot2[0,:,:], cmap='gray')
axs[0,1].set_title(f'DCNet (UNet)', fontsize=16)
noaxis(axs[0,1])
add_colorbar(im2, 'bottom')

im5=axs[0,2].imshow(x_plot5[0,:,:], cmap='gray')
axs[0,2].set_title(f'PinvNet + I', fontsize=16)
noaxis(axs[0,2])
add_colorbar(im5, 'bottom')

im3=axs[1,0].imshow(x_plot3[0,:,:], cmap='gray')
axs[1,0].set_title(f'UPGD (CNN) \n num_iter=2', fontsize=16)
noaxis(axs[1,0])
add_colorbar(im3, 'bottom')

im4=axs[1,1].imshow(x_plot4[0,:,:], cmap='gray')
axs[1,1].set_title(f'UPGD (CNN) \n num_iter=6', fontsize=16)
noaxis(axs[1,1])
add_colorbar(im4, 'bottom')

plt.show()

###############################################################################
# UPGD with a small CNN is worse that UNet denoiser, as the latter has a denoiser with 
# higher capacity. Increasing the capacity of the denoiser in UPGD should lead to better results.
