r"""
Demo to replicate the results of the paper "SPyRiT - a Python Toolbox for Deep Single-Pixel Image Reconstruction"
==========================

We simulate split Hamamard measurements and a 'variance' undersampling and 
apply several reconstruction networks availale in SPyRiT: 
- PinvNet network with UNet denoiser
- DCNet network with UNet denoiser
- Unrolled proximal gradient descent (UPGD) network with UNet denoiser

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
res_path = os.path.join(spyritPath, '../results')
if os.path.exists(res_path) is False:
    os.mkdir(res_path)
    print(f'Created {res_path}')

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

fig = plt.figure(figsize=(7,7))
plt.imshow(x_plot[0,:,:], cmap='gray', vmin=-1, vmax=1)
plt.axis('off')
plt.savefig(os.path.join(res_path, 'bird_gt_64.png'), bbox_inches='tight', pad_inches=0)

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

# Download the covariance matrix and mean image
data_folder = '../../stat/'
dataId_list = [
        '63935b624d15dd536f0484a5', # for reconstruction (imageNet, 64)
        '63935a224d15dd536f048496', # for reconstruction (imageNet, 64)
        ]
cov_name = '../../stat/Cov_64x64.npy'
download_cov = False
if download_cov:
    # Generate the warehouse client
    gc = girder_client.GirderClient(apiUrl=url)

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
M = h*h // 4      # Number of measurements (2048, 1/4 of the pixels)
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%
# PinvNet network 
# -----------------------------------------------------------------------------

###############################################################################
# We reconstruct with the pseudo inverse using :class:`spyrit.core.recon.PinvNet` class 
# as in the previous tutorial. For this, we define the neural network and then perform the reconstruction.
from spyrit.core.train import load_net, save_net
from spyrit.core.recon import PinvNet
from spyrit.core.nnet import Unet
denoi_unet = Unet()

# Reconstruction with for Core module (linear net)
pinvnet = PinvNet(noise_op, prep_op)
pinvnet_unet = PinvNet(noise_op, prep_op, denoi_unet)
pinvnet_unet = pinvnet_unet.to(device)

# Load previously trained model
try:
    import gdown

    # Download weights
    url_net = 'https://drive.google.com/file/d/1kiovsHATuf_D0qCAxyuDDzs-PA1Yo9hK/view?usp=drive_link'
    name_net = 'pinv-net_unet_stl10_N0_100_m_hadam-split_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07'
    model_path = "../model"    
    if os.path.exists(model_path) is False:
        os.mkdir(model_path)
        print(f'Created {model_path}')
    model_file = os.path.join(model_path, name_net)
    gdown.download(url_net, f'{model_file}.pth', quiet=False,fuzzy=True)

    # Load pretrained model
    load_net(model_file, pinvnet_unet, device, False)
    print(f'Model {model_file} loaded.')
except:
    print(f'Model {model_file} not found!')

# Reconstruction
with torch.no_grad():
    z_pinvnet = pinvnet.reconstruct(y.to(device))  # reconstruct from raw measurements
    z_pinvnet_unet = pinvnet_unet.reconstruct(y.to(device))  # reconstruct from raw measurements


# plot
x_plot = z_pinvnet.view(-1,h,h).cpu().numpy() 
fig = plt.figure(figsize=(7,7))
plt.imshow(x_plot[0,:,:], cmap='gray', vmin=-1, vmax=1)
plt.axis('off')
plt.savefig(os.path.join(res_path, 'bird_pinvnet_64.png'), bbox_inches='tight', pad_inches=0)

# plot
x_plot = z_pinvnet_unet.view(-1,h,h).cpu().numpy() 
fig = plt.figure(figsize=(7,7))
plt.imshow(x_plot[0,:,:], cmap='gray', vmin=-1, vmax=1)
plt.axis('off')
plt.savefig(os.path.join(res_path, 'bird_pinvnet_unet_64.png'), bbox_inches='tight', pad_inches=0)

# %%
# DCNET network 
# -----------------------------------------------------------------------------

###############################################################################
# We compare the results with the DCNET + UNet network (see :ref:`DCNET tutorial <tuto_dcnet_split_measurements>`).

# Pretrained DC UNet (UNet denoising)
from spyrit.core.nnet import Unet
from spyrit.core.recon import DCNet

dcnet = DCNet(noise_op, prep_op, Cov)
dcnet = dcnet.to(device)

denoi_unet = Unet()
dcnet_unet = DCNet(noise_op, prep_op, Cov, denoi_unet)
dcnet_unet = dcnet_unet.to(device)

from spyrit.core.train import load_net, save_net

# Load previously trained model
try:
    import gdown

    # Download weights
    url_net = 'https://drive.google.com/file/d/15PRRZj5OxKpn1iJw78lGwUUBtTbFco1l/view?usp=drive_link'
    name_net = 'dc-net_unet_stl10_N0_100_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07'
    model_path = "../model"    
    if os.path.exists(model_path) is False:
        os.mkdir(model_path)
        print(f'Created {model_path}')
    model_net_path = os.path.join(model_path, name_net)

    download_net = False
    if download_net:
        gdown.download(url_net, f'{model_net_path}.pth', quiet=False,fuzzy=True)

    # Load pretrained model
    load_net(model_net_path, dcnet_unet, device, False)
    print(f'Model {model_net_path} loaded.')
except:
    print(f'Model {model_net_path} not found!')

# Reconstruction
with torch.no_grad():
    z_dcnet_unet = dcnet_unet.reconstruct(y.to(device))  # reconstruct from raw measurements
    z_dcnet = dcnet.reconstruct(y.to(device))  # reconstruct from raw measurements

# plot
x_plot = z_dcnet.view(-1,h,h).cpu().numpy() 
fig = plt.figure(figsize=(7,7))
plt.imshow(x_plot[0,:,:], cmap='gray', vmin=-1, vmax=1)
plt.axis('off')
plt.savefig(os.path.join(res_path, 'bird_dcnet_64.png'), bbox_inches='tight', pad_inches=0)

x_plot = z_dcnet_unet.view(-1,h,h).cpu().numpy() 
fig = plt.figure(figsize=(7,7))
plt.imshow(x_plot[0,:,:], cmap='gray', vmin=-1, vmax=1)
plt.axis('off')
plt.savefig(os.path.join(res_path, 'bird_dcnet_unet_64.png'), bbox_inches='tight', pad_inches=0)

# %%
# DCNet with DRUNet denoising
# -----------------------------------------------------------------------------

###############################################################################
# We define the DRUNetDen network by providing the number of channels and noise level :attr:`noise_level`, 
# which is expected to be in [0, 255]. The larger the noise level, the higher the denoising.
# The noise level is not required as it can be set later on. 

from spyrit.external.drunet import DRUNet

# DRUnet denoising
# DRUNet(noise_level=5, n_channels=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose')
noise_level = 12
denoi_drunet = DRUNet(noise_level=noise_level, n_channels=1)

# Set the device for DRUNet
denoi_drunet = denoi_drunet.to(device)

try:
    import gdown
    # Download pretrained weights
    model_path = './model'
    url_net = 'https://drive.google.com/file/d/1oSsLjPPn6lqtzraFZLZGmwP_5KbPfTES/view?usp=drive_link'

    if os.path.exists(model_path) is False:
        os.mkdir(model_path)
        print(f'Created {model_path}')

    model_file = os.path.join(model_path, 'drunet_gray.pth')
    gdown.download(url_net, model_file, quiet=False,fuzzy=True)

    # Load pretrained weights
    denoi_drunet.load_state_dict(torch.load(model_file), strict=False)       
    print(f'Model {model_file} loaded.')
except:
    print(f'Model {model_file} not found!')

# Define DCNet with DRUNet denoising
dcnet_drunet = DCNet(noise_op, prep_op, Cov, denoi=denoi_drunet) 
dcnet_drunet = dcnet_drunet.to(device)

# Set noise level
noise_level = 10
denoi_drunet.set_noise_level(noise_level)

# Reconstruction
with torch.no_grad():
    z_dcnet_drunet = dcnet_drunet.reconstruct(y.to(device))  # reconstruct from raw measurements

# plot
x_plot = z_dcnet_drunet.view(-1,h,h).cpu().numpy() 
fig = plt.figure(figsize=(7,7))
plt.imshow(x_plot[0,:,:], cmap='gray', vmin=0, vmax=1)
plt.axis('off')
#plt.colorbar()
plt.savefig(os.path.join(res_path, f'bird_dcnet_drunet_n{noise_level}_64.png'), bbox_inches='tight', pad_inches=0)

# %%
# PInvNet with DRUNet denoising
# -----------------------------------------------------------------------------

###############################################################################
# We define the DRUNetDen network by providing the number of channels and noise level :attr:`noise_level`, 
# which is expected to be in [0, 255]. The larger the noise level, the higher the denoising.
# The noise level is not required as it can be set later on. 

from spyrit.external.drunet import DRUNet

# DRUnet denoising
# DRUNet(noise_level=5, n_channels=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose')
noise_level = 5
denoi_drunet = DRUNet(noise_level=noise_level, n_channels=1)

# Set the device for DRUNet
denoi_drunet = denoi_drunet.to(device)

try:
    import gdown
    # Download pretrained weights
    model_path = './model'
    url_net = 'https://drive.google.com/file/d/1oSsLjPPn6lqtzraFZLZGmwP_5KbPfTES/view?usp=drive_link'

    if os.path.exists(model_path) is False:
        os.mkdir(model_path)
        print(f'Created {model_path}')

    model_file = os.path.join(model_path, 'drunet_gray.pth')
    gdown.download(url_net, model_file, quiet=False,fuzzy=True)

    # Load pretrained weights
    denoi_drunet.load_state_dict(torch.load(model_file), strict=False)       
    print(f'Model {model_file} loaded.')
except:
    print(f'Model {model_file} not found!')

# Define DCNet with DRUNet denoising
pinvnet_drunet = PinvNet(noise_op, prep_op, denoi=denoi_drunet) 
pinvnet_drunet = pinvnet_drunet.to(device)

# Set noise level
#noise_level = 5
#pinvnet_drunet.set_noise_level(noise_level)

# Reconstruction
with torch.no_grad():
    z_pinvnet_drunet = pinvnet_drunet.reconstruct(y.to(device))  # reconstruct from raw measurements

# plot
x_plot = z_pinvnet_drunet.view(-1,h,h).cpu().numpy() 
fig = plt.figure(figsize=(7,7))
plt.imshow(x_plot[0,:,:], cmap='gray', vmin=0, vmax=1)
plt.axis('off')
#plt.colorbar()
plt.savefig(os.path.join(res_path, f'bird_pinvnet_drunet_n{noise_level}_64.png'), bbox_inches='tight', pad_inches=0)


# %%
# DRUNet denoising
# -----------------------------------------------------------------------------

from spyrit.external.drunet import UNetRes as drunet

# Define denoising network
n_channels = 1                   # 1 for grayscale image    
drunet_den = drunet(in_nc=n_channels+1, out_nc=n_channels)  

# Load pretrained model
try:       
    model_path = './model'
    model_file = os.path.join(model_path, 'drunet_gray.pth')
    drunet_den.load_state_dict(torch.load(model_file), strict=True)       
    print(f'Model {model_file} loaded.')
except:
    print(f'Model {model_file} not found!')
    load_drunet = False

# Set the device for DRUNet
drunet_den = drunet_den.to(device)

# 2nd step - Denoising
# Convert to [0,1]
x_sample = 0.5*(z_pinvnet + 1).cpu()

# Create noise-level map and concatenate to the image
noise_level = 12
x_sample = torch.cat((x_sample, torch.FloatTensor([noise_level/255.]).repeat(1, 1, x_sample.shape[2], x_sample.shape[3])), dim=1)        
x_sample = x_sample.to(device)

with torch.no_grad():
    z_pinvnet_den = drunet_den(x_sample)

# plot
x_plot = z_pinvnet_den.view(-1,h,h).cpu().numpy() 
fig = plt.figure(figsize=(7,7))
plt.imshow(x_plot[0,:,:], cmap='gray', vmin=0, vmax=1)
plt.axis('off')
#plt.colorbar()
plt.savefig(os.path.join(res_path, f'bird_pinvnet_drunet_den_n{noise_level}_64.png'), bbox_inches='tight', pad_inches=0)

###############################################################################

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
from spyrit.core.recon import UPGD

# Denoising network
denoi_cnn = ConvNet()

# UPGD for 3 iterations
# Start with fix stepsizes [1e-5, 5e-6, 1e-6], then train them after 15 epochs
upgd_cnn_3it = UPGD(noise_op, prep_op, denoi_cnn, num_iter = 3)
upgd_cnn_3it = upgd_cnn_3it.to(device)

# Load previously trained models
try:
    import gdown
    model_path = "./model"    
    if os.path.exists(model_path) is False:
        os.mkdir(model_path)
        print(f'Created {model_path}')

    url_upgd_3it = 'https://drive.google.com/file/d/1nPAjSjIgBRjazDBCvEkK7NlcI9_-v6TZ/view?usp=drive_link'
    name_net_3it = 'upgd_cnn_stl10_N0_100_m_hadam-split_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_uit_3_la_1e-05'

    model_upgd_3it_path = os.path.join(model_path, name_net_3it)    

    # Download weights
    gdown.download(url_upgd_3it, f'{model_upgd_3it_path}.pth', quiet=False,fuzzy=True)

    """
    model_upgd_path = './model/upgd_cnn_stl10_N0_100_N_64_M_1024_epo_2_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07'
    """

    # Load pretrained model
    load_net(model_upgd_3it_path, upgd_cnn_3it, device=device, strict=False)    

    # Print stepsizes
    print(f'Stepsizes: {upgd_cnn_3it.lambs}')

except:
    print(f'Model not found!')

# Reconstruction
with torch.no_grad():
    z_upgd_3it = upgd_cnn_3it.reconstruct(y.to(device))  # reconstruct from raw measurements

###############################################################################
# We plot all results side by side.

from spyrit.misc.disp import add_colorbar, noaxis

x_plot = x.view(-1,h,h).cpu().numpy()    
x_plot2 = z_dcnet_unet.view(-1,h,h).cpu().numpy() 
x_plot3 = z_upgd_3it.view(-1,h,h).cpu().numpy() 
x_plot5 = z_invnet_unet.view(-1,h,h).cpu().numpy() 

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
axs[0,2].set_title(f'PinvNet (UNet)', fontsize=16)
noaxis(axs[0,2])
add_colorbar(im5, 'bottom')

im3=axs[1,0].imshow(x_plot3[0,:,:], cmap='gray')
axs[1,0].set_title(f'UPGD (UNet) \n num_iter=3', fontsize=16)
noaxis(axs[1,0])
add_colorbar(im3, 'bottom')

# im4=axs[1,2].imshow(x_plot6[0,:,:], cmap='gray')
# axs[1,2].set_title(f'PinvNet (CNN)', fontsize=16)
# noaxis(axs[1,2])
# add_colorbar(im4, 'bottom')

plt.show()

###############################################################################
# UPGD with a small CNN is worse that UNet denoiser, as the latter has a denoiser with 
# higher capacity. Increasing the capacity of the denoiser in UPGD should lead to better results.
