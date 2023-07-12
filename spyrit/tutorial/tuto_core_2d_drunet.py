r"""
01. Tutorial 2D - Image reconstruction for single-pixel imaging using pretrained DRUNet denoising network
======================
This tutorial shows how to simulate data and perform image reconstruction with DC-DRUNet 
(data completion with pretrained DRUNet denoising network) for single-pixel imaging. 
For data simulation, it loads an image from ImageNet and simulated measurements based on 
an undersampled Hadamard operator. You can select number of counts and undersampled factor. 

Image reconstruction is preformed using the following methods: 
    Pseudo-inverse
    PInvNet:        Linear net
    DCNet:          Data completion net with unit matrix denoising
    DCUNet:         Data completion with UNet denoising, trained on stl10 dataset.
                    Refer to tuto_run_train_colab.ipynb for an example to train DCUNet.
    DCDRUNet:       Data completion with pretrained DRUNet denoising.

    DRUNet taken from https://github.com/cszn/DPIR
    Deep Plug-and-Play Image Restoration (DPIR) toolbox
    June 2023

"""



import numpy as np
import os
import matplotlib.pyplot as plt
from spyrit.core.meas import HadamSplit
from spyrit.core.noise import NoNoise, Poisson, PoissonApproxGauss
from spyrit.core.prep import SplitPoisson
from spyrit.core.recon import PseudoInverse, PinvNet, DCNet, DCDRUNet
from spyrit.misc.statistics import Cov2Var, data_loaders_stl10, transform_gray_norm
from spyrit.misc.disp import imagesc
from spyrit.misc.sampling import meas2img2
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net

import torch
import torchvision
# pip install girder-client
# pip install gdown
import girder_client
import gdown

from spyrit.external.drunet import UNetRes as drunet
from spyrit.external.drunet import uint2single, single2tensor4
#from spyrit.external import drunet_utils as util    

H = 64                          # Image height (assumed squared image)
M = H**2 // 4                   # Num measurements = subsampled by factor 2
B = 10                          # Batch size
alpha = 100                     # ph/pixel max: number of counts
download_cov = True             # Dwonload covariance matrix;
                                # otherwise, set to unit matrix
ind_img = 1                     # Image index for image selection

imgs_path = './spyrit/images'

cov_name = './stat/Cov_64x64.npy'

# use GPU, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################################
# So far we have been able to estimate our posterion mean. What about its
# uncertainties (i.e., posterion covariance)?

# Uncomment to download stl10 dataset
# A batch of images
# dataloaders = data_loaders_stl10('../../../data', img_size=H, batch_size=10)  
# dataloader = dataloaders['train']

# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=H)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = min(B, len(dataset)))

# Select image
x0, _ = next(iter(dataloader))
x0 = x0[ind_img:6,:,:,:]
x = x0.detach().clone()
b,c,h,w = x.shape
x = x.view(b*c,h*w)
print(f'Shape of incoming image (b*c,h*w): {x.shape}')

# Operators 
#
# Order matrix with shape (H, H) used to compute the permutation matrix 
# (as undersampling taking the first rows only)

if (download_cov is True):
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
    for dataId in dataId_list:
        myfile = gc.getFile(dataId)
        gc.downloadFile(dataId, data_folder + myfile['name'])

    print(f'Created {data_folder}') 

try:
    Cov  = np.load(cov_name)
except:
    Cov = np.eye(H*H)
    print(f"Cov matrix {cov_name} not found! Set to the identity")

Ord = Cov2Var(Cov)

# Measurement operator: 
# Computes linear measurements y=Px, where P is a linear operator (matrix) with positive entries      
# such that P=[H_{+}; H_{-}]=[max(H,0); max(0,-H)], H=H_{+}-H_{-}
meas_op = HadamSplit(M, H, Ord)

# Simulates raw split measurements from images in the range [0,1] assuming images provided in range [-1,1]
# y=0.5*H(1 + x)
# noise = NoNoise(meas_op) # noiseless
#noise = Poisson(meas_op, alpha)
noise = PoissonApproxGauss(meas_op, alpha) # faster than Poisson

# Preprocess the raw data acquired with split measurement operator assuming Poisson noise
prep = SplitPoisson(alpha, meas_op)

# Reconstruction with pseudoinverse
pinv = PseudoInverse()

# Reconstruction with for Core module (linear net)
pinvnet = PinvNet(noise, prep)

# Reconstruction with for DCNet (linear net + denoising net)
dcnet = DCNet(noise, prep, Cov)

# Pretreined DC UNet (UNet denoising)
denoi = Unet()
dcunet = DCNet(noise, prep, Cov, denoi)

# Load previously trained UNet model

# Path to model
models_path = "./model"
model_unet_path = os.path.join(models_path, "dc-net_unet_imagenet_var_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07_light")
if os.path.exists(models_path) is False:
    os.mkdir(models_path)
    print(f'Created {models_path}')

try:
    # Download weights
    url_unet = 'https://drive.google.com/file/d/1LBrjU0B-Tecd4GBRozX9-24LTRzIiMzA/view?usp=drive_link'
    gdown.download(url_unet, f'{model_unet_path}.pth', quiet=False,fuzzy=True)

    # Load model from path
    load_net(model_unet_path, dcunet, device, False)
    print(f'Model {model_unet_path} loaded.')
    load_unet = True
except:
    print(f'Model {model_unet_path} not found!')
    load_unet = False

# DCDRUNet
#
# Download DRUNet weights
url_drunet = 'https://drive.google.com/file/d/1oSsLjPPn6lqtzraFZLZGmwP_5KbPfTES/view?usp=drive_link'
model_drunet_path = os.path.join(models_path, 'drunet_gray.pth')
try:
    gdown.download(url_drunet, model_drunet_path, quiet=False,fuzzy=True)

    # Define denoising network
    n_channels = 1                   # 1 for grayscale image    
    denoi_drunet = drunet(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',                     
                downsample_mode="strideconv", upsample_mode="convtranspose")  

    # Load pretrained model
    denoi_drunet.load_state_dict(torch.load(model_drunet_path), strict=True)       
    print(f'Model {model_drunet_path} loaded.')
    load_drunet = True
except:
    print(f'Model {model_drunet_path} not found!')
    load_drunet = False

if load_drunet is True:
    denoi_drunet.eval()         
    for k, v in denoi_drunet.named_parameters():             
        v.requires_grad = False  
    print(sum(map(lambda x: x.numel(), denoi_drunet.parameters())) )  

    # Define DCDRUNet
    #noise_level = 10
    #dcdrunet = DCDRUNet(noise, prep, Cov, denoi_drunet, noise_level=noise_level)
    dcdrunet = DCDRUNet(noise, prep, Cov, denoi_drunet)

# Simulate measurements
y = noise(x)
m = prep(y)
print(f'Shape of simulated measurements y: {y.shape}')
print(f'Shape of preprocessed data m: {m.shape}')

# Reconstructions
#
# Pseudo-inverse
z_pinv = pinv(m, meas_op)
print(f'Shape of reconstructed image z: {z_pinv.shape}')

# Pseudo-inverse net
pinvnet = pinvnet.to(device)

x = x0.detach().clone()
x = x.to(device)
z_pinvnet = pinvnet(x)
# z_pinvnet = pinvnet.reconstruct(y)

# DCNet
#y = pinvnet.acquire(x)         # or equivalently here: y = dcnet.acquire(x)
#m = pinvnet.meas2img(y)        # zero-padded images (after preprocessing)
dcnet = dcnet.to(device)
z_dcnet = dcnet.reconstruct(y.to(device))  # reconstruct from raw measurements

# DC UNET 
if (load_unet is True):
    dcunet = dcunet.to(device)
    with torch.no_grad():
        z_dcunet = dcunet.reconstruct(y.to(device))  # reconstruct from raw measurements

# DC DRUNET 
if (load_drunet is True):
    # Reconstruct with DCDRUNet
    # Uncomment to set a new noise level: The higher the noise, the higher the denoising
    noise_level = 10
    dcdrunet.set_noise_level(noise_level)
    dcdrunet = dcdrunet.to(device)
    with torch.no_grad():
        # reconstruct from raw measurements
        z_dcdrunet = dcdrunet.reconstruct(y.to(device))  

    denoi_drunet = denoi_drunet.to(device)

    # -----------
    # Denoise original image with DRUNet
    noise_level = 10
    x_sample = 0.5*(x[0,0,:,:] + 1).cpu().numpy()
    imagesc(x_sample  ,'Ground-truth image normalized', show=False)

    x_sample = uint2single(255*x_sample)
    x_sample = single2tensor4(x_sample[:,:,np.newaxis])
    x_sample = torch.cat((x_sample, torch.FloatTensor([noise_level/255.]).repeat(1, 1, x_sample.shape[2], x_sample.shape[3])), dim=1)        
    x_sample = x_sample.to(device)
    z_den_drunet = denoi_drunet(x_sample)

# Plots
x_plot = x.view(-1,H,H).cpu().numpy()    
imagesc(x_plot[0,:,:],'Ground-truth image normalized', show=False)

m_plot = y.numpy()   
m_plot = meas2img2(m_plot.T, Ord)
m_plot = np.moveaxis(m_plot,-1, 0)
imagesc(m_plot[0,:,:],'Simulated Measurement', show=False)

m_plot = m.numpy()   
m_plot = meas2img2(m_plot.T, Ord)
m_plot = np.moveaxis(m_plot,-1, 0)
imagesc(m_plot[0,:,:],'Preprocessed data', show=False)

m_plot = m.numpy()   
m_plot = meas2img2(m_plot.T, Ord)
m_plot = np.moveaxis(m_plot,-1, 0)

z_plot = z_pinv.view(-1,H,H).numpy()
imagesc(z_plot[0,:,:],'Pseudo-inverse reconstruction', show=False)

z_plot = z_pinvnet.view(-1,H,H).cpu().numpy()
imagesc(z_plot[0,:,:],'Pseudo-inverse net reconstruction', show=False)

z_plot = z_dcnet.view(-1,H,H).cpu().numpy()
imagesc(z_plot[0,:,:],'DCNet reconstruction', show=False)

if (load_unet is True):
    z_plot = z_dcunet.view(-1,H,H).detach().cpu().numpy()
    imagesc(z_plot[0,:,:],'DC UNet reconstruction', show=False)

if (load_drunet is True):
    # DRUNet denoising
    z_plot = z_den_drunet.view(-1,H,H).detach().cpu().numpy()
    imagesc(z_plot[0,:,:],'DRUNet denoising of original image', show=False)

    # DCDRUNet
    z_plot = z_dcdrunet.view(-1,H,H).detach().cpu().numpy()
    imagesc(z_plot[0,:,:],f'DC DRUNet reconstruction noise level={noise_level}', show=False)

plt.show()

###############################################################################
# Note that here we have been able to compute a sample posterior covariance
# from its estimated samples. By displaying it we can see  how both the overall
# variances and the correlation between different parameters have become
# narrower compared to their prior counterparts.
