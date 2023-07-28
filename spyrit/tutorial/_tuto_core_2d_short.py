r"""
01. Tutorial 2D - Image reconstruction for single-pixel imaging
======================
This tutorial focuses on Bayesian inversion, a special type of inverse problem
that aims at incorporating prior information in terms of model and data
probabilities in the inversion process.

It shows how to simulate data and perform image reconstruction with spyrit toolbox. 
For data simulation, it loads an image from ImageNet and simulated measurements based on 
an undersampled Hadamard operator. You can select number of counts and undersampled factor. 

Image reconstruction is preformed using the following methods: 
    Pseudo-inverse
    PInvNet:        Linear net
    DCNet:          Data completion net with unit matrix denoising
    DCUNet:         Data completion with UNet denoising, trained on stl10 dataset.
                    Refer to tuto_run_train_colab.ipynb for an example to train DCUNet.

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from spyrit.core.meas import HadamSplit
from spyrit.core.noise import NoNoise, Poisson, PoissonApproxGauss
from spyrit.core.prep import SplitPoisson
from spyrit.core.recon import PseudoInverse, PinvNet, DCNet
from spyrit.misc.statistics import Cov2Var, data_loaders_stl10, transform_gray_norm
from spyrit.misc.disp import imagesc
from spyrit.misc.sampling import meas2img2
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net

import torch
import torchvision

H = 64                          # Image height (assumed squared image)
M = H**2 // 4                   # Num measurements = subsampled by factor 2
B = 10                          # Batch size
alpha = 100                     # ph/pixel max: number of counts
load_unet = True                # Load pretrained UNet denoising
download_cov = False            # Download covariance matrix;
                                # otherwise, set to unit matrix

spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, '../images')
print(imgs_path)

cov_name = os.path.join(spyritPath, '../../stat/Cov_64x64.npy')

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
x0 = x0[1:6,:,:,:]
x = x0.detach().clone()
b,c,h,w = x.shape
x = x.view(b*c,h*w)
print(f'Shape of incoming image (b*c,h*w): {x.shape}')

"""
Data Simulation:
    1) Split Linear Measurements:
            y = Px = [H_{+}; H_{-}]x
        
        spyrit.core.meas
        meas_op = LinearSplit(H), matrix H
        meas_op = HadamSplit(M, h, Ord), M:#meas, h=height, Ord=Ordering matrix for undersampling

        y = meas_op(x)

    2) No Noisy/Noisy raw measurements (handling negative images):
        Handles the fact that images are between [-1, 1] and construct measurements 
        from measurements operator.
        Simulates raw measurements as expected by the single pixel camera (no negative measurements)

        Noiseless:
                y = 0.5*H(1+x)

            spyrit.core.noise
            meas_op = HadamSplit(M, h, Ord)
            y = NoNoise(meas_op)(x) 

        Noisy:
                y = Poisson((alpha/2)*H(1+x))

            spyrit.core.noise
            meas_op = HadamSplit(M, h, Ord)    
            y = Poisson(meas_op, alpha)(x)

    3) Preprocess measurements (before reconstruction): 
        Proceprocess to compensates for image normalization and splitting
        Mixes split measurements.
            m = (y+ - y-)/alpha - H*I
        
            spyrit.core.prep
            meas_op = HadamSplit(M, h, Ord)    
            m = SplitPoisson(alpha, meas_op)(y)

    4) Reconstruct

        Standard reconstruction:
            z = PseudoInverse()(m, meas_op)
        
        Inverse Net:
            Noiseless:
            pinv_net = PinvNet(NoNoise(meas_op), SplitPoisson(alpha, meas_op))
            z = pinv_net(x)

            Noisy:
            pinv_net = PinvNet(Poisson(meas_op, alpha), SplitPoisson(alpha, meas_op))
            z_invnet = pinv_net.reconstruct(y)

        DCNet:
            dcnet = DCNet(Poisson(meas_op, alpha), SplitPoisson(alpha, meas_op), Cov)
            y = dcnet.acquire(x) 
            z_dc = dcnet.reconstruct(y)
            """

# Operators 
#
# Order matrix with shape (H, H) used to compute the permutation matrix 
# (as undersampling taking the first rows only)

if (download_cov is True):
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
dcnet_unet = DCNet(noise, prep, Cov, denoi)

# Load previously trained model
try:
    model_path = "./model/dc-net_unet_imagenet_var_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07_light"
    #model_path = './model/dc-net_unet_stl10_N0_100_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07.pth'
    #dcnet_unet.load_state_dict(torch.load(model_path), loa)
    load_net(model_path, dcnet_unet, device, False)
    print(f'Model {model_path} loaded.')
except:
    print(f'Model {model_path} not found!')
    load_unet = False

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
    dcnet_unet = dcnet_unet.to(device)
    with torch.no_grad():
        z_dcunet = dcnet_unet.reconstruct(y.to(device))  # reconstruct from raw measurements

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
    plt.show()

###############################################################################
# Note that here we have been able to compute a sample posterior covariance
# from its estimated samples. By displaying it we can see  how both the overall
# variances and the correlation between different parameters have become
# narrower compared to their prior counterparts.
