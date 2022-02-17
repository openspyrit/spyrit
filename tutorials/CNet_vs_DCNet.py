# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 12:12:42 2021

@author: ducros
"""

#%%
from __future__ import print_function, division
import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
#import spyrit.misc.walsh_hadamard as wh
from spyrit.learning.model_Had_DCAN import *
from spyrit.learning.nets import *
from spyrit.misc.metrics import psnr, psnr_, batch_psnr

#%%
#- Acquisition
img_size = 64 # image size

#- simulations
N0 = 50   # mean of the maximun number of photons
M = 1024  # number of measurements

#- Model and data paths
ph = [50, 10]
data_root = Path('../data/')
stats_root = Path('../stats_walsh/')
model_root = '../models_v1.2/'

#- Plot options
plt.rcParams['pdf.fonttype'] = 42   # Save plot using type 1 font
plt.rcParams['text.usetex'] = True  # Latex

#%% A batch of STL-10 test images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

testset = \
    torchvision.datasets.STL10(root=data_root, split='test',download=True, transform=transform)
testloader =  torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False)

inputs, _ = next(iter(testloader))
b,c,h,w = inputs.shape

#%%
Cov = np.load(stats_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))
Mean = np.load(stats_root / Path("Average_{}x{}.npy".format(img_size, img_size)))
H =  wh.walsh2_matrix(img_size)/img_size
Cov /= img_size*img_size # THIS NEEDS TO BE NORMALIAZED FOR CONSISTENCY!

#%% Simulate measurements
model = DenoiCompNet(img_size, M, Mean, Cov, N0=N0, H=H)
model2 = noiCompNet(img_size, M, Mean, Cov, N0=N0, H=H)

model = model.to(device)
model2 = model2.to(device)
inputs = inputs.to(device)

#%% Imgae index
ind = [1,2,3,4]
#ind = [72,73,74,75]

#%%
torch.manual_seed(0) # for reproducibility
#torch.seed()         # for random measurements
meas = model.forward_acquire(inputs, b, c, h, w)    # with pos/neg coefficients

#%% Load networks
# 50 photons, NO DENOISING
title  = f'NET_c0mp_N0_{ph[0]:.1f}_sig_0.0_N_64_M_1024_epo_40_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07'
load_net(model_root / Path(title), model2, device)
recon_1 = model2.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()

# 10 photons, no variability, NO DENOISING
title  = f'NET_c0mp_N0_{ph[1]:.1f}_sig_0.0_N_64_M_1024_epo_40_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07'
load_net(model_root / Path(title), model2, device)
recon_2 = model2.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()

# 50 photons, no variability,  DENOISING
title  = f'NET_c0mp_N0_{ph[0]:.1f}_sig_0.0_Denoi_N_64_M_1024_epo_40_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07'
load_net(model_root / Path(title), model, device)
recon_1_denoi = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()

# 10 photons, no variability, DENOISING
title  = f'NET_c0mp_N0_{ph[1]:.1f}_sig_0.0_Denoi_N_64_M_1024_epo_40_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07'
load_net(model_root / Path(title), model, device)
recon_2_denoi = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()

#%% Plot
f, axs = plt.subplots(4, 5, figsize=(10,10),  dpi= 100)

for i_ind,v_ind in enumerate(ind): 
   
    
    img = inputs[v_ind, 0, :, :].cpu().detach().numpy()
    
    #-- Recon
    rec_1 = recon_1[v_ind, 0, :, :]
    rec_1_denoi = recon_1_denoi[v_ind, 0, :, :]
    rec_2_denoi = recon_2_denoi[v_ind, 0, :, :]
    rec_2 = recon_2[v_ind, 0, :, :]
    
    #- Plot   
    axs[i_ind, 0].imshow(img, cmap='gray')
    axs[i_ind, 0].set_title("Ground-truth")
    axs[i_ind, 0].get_xaxis().set_visible(False)
    axs[i_ind, 0].get_yaxis().set_visible(False)
    axs[i_ind, 0].axis('off')
    axs[i_ind, 1].imshow(rec_1, cmap='gray')
    axs[i_ind, 1].set_title(f"{ph[0]} ph: ${psnr_(img,rec_1):.2f}$ dB")
    axs[i_ind, 2].imshow(rec_1_denoi, cmap='gray')
    axs[i_ind, 2].set_title(f"{ph[0]} ph denoi: ${psnr_(img,rec_1_denoi):.2f}$ dB")
    axs[i_ind, 3].imshow(rec_2, cmap='gray')
    axs[i_ind, 3].set_title(f"{ph[1]} ph: ${psnr_(img,rec_2):.2f}$ dB")
    axs[i_ind, 4].imshow(rec_2_denoi, cmap='gray')
    axs[i_ind, 4].set_title(f"{ph[1]} ph denoi: ${psnr_(img,rec_2_denoi):.2f}$ dB")

# remove axes
for ax in iter(axs.flatten()):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #ax.axis('off')

f.subplots_adjust(wspace=0, hspace=0)
plt.suptitle(f"Measurements simulated with ${N0}$ photons")
#plt.savefig("net_denoising.pdf", bbox_inches=0)