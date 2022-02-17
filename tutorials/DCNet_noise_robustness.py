# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 12:12:42 2021

@author: ducros
"""

#%%
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path

from spyrit.learning.model_Had_DCAN import *
from spyrit.learning.nets import *
from spyrit.misc.metrics import psnr, psnr_, batch_psnr

#%%
#- Acquisition
img_size = 64       # image size
M = 1024            # number of measurements
ind = 72            # image index
ph = [50, 10, 2]    # nb of photons

#- Model and data paths
data_root = Path('../data/')
stats_root = Path('../models_v1.2/')
model_root =  Path('../models_v1.2/')

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
inputs = inputs.to(device)

#%%
Cov = np.load(stats_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))
Mean = np.load(stats_root / Path("Average_{}x{}.npy".format(img_size, img_size)))
H =  wh.walsh2_matrix(img_size)/img_size
Cov /= img_size*img_size # THIS NEEDS TO BE NORMALIAZED FOR CONSISTENCY!

#%% Reconstruction network
prefix = 'NET_c0mp_'
suffix = f'N_64_M_{M:d}_epo_40_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07'

# noiseless 
title_free  = prefix + suffix

# 50 photons
title_50  = prefix + 'N0_50.0_sig_0.0_Denoi_' + suffix

# 10 photons
title_10  = prefix + 'N0_10.0_sig_0.0_Denoi_' + suffix

# 2 photons
title_2  = prefix + 'N0_2.0_sig_0.0_Denoi_' + suffix

#%% Plot
f, axs = plt.subplots(3, 5, figsize=(10,7),  dpi= 100)

img = inputs[ind, 0, :, :].cpu().detach().numpy()

for ph_i, ph_v in enumerate(ph):
    
    #-- Simulate measurements
    model = DenoiCompNet(img_size, M, Mean, Cov, N0=ph_v, H=H)
    model = model.to(device)
    torch.manual_seed(0)    # for reproducibility
    #torch.seed()           # for random measurements
    meas = model.forward_acquire(inputs, b, c, h, w)    # with pos/neg coefficients
    
    #-- Reconstruction
    load_net(model_root / Path(title_free), model, device)
    recon_free = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()
    load_net(model_root / Path(title_50), model, device)
    recon_50 = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()
    load_net(model_root / Path(title_10), model, device)
    recon_10 = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()
    load_net(model_root / Path(title_2), model, device)
    recon_2 = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()
    #
    print(model.N0)
    #
    rec_free  = recon_free[ind, 0, :, :]
    rec_50 = recon_50[ind, 0, :, :]
    rec_10 = recon_10[ind, 0, :, :]
    rec_2 = recon_2[ind, 0, :, :]
    
    #- Plot   
    axs[ph_i, 0].imshow(img, cmap='gray')
    axs[ph_i, 0].set_title("Ground-truth")
    axs[ph_i, 1].imshow(rec_free, cmap='gray')
    axs[ph_i, 1].set_title(f"NET no noise: ${psnr_(img,rec_free):.2f}$ dB")
    axs[ph_i, 2].imshow(rec_50, cmap='gray')
    axs[ph_i, 2].set_title(f"50 ph: ${psnr_(img,rec_50):.2f}$ dB")
    axs[ph_i, 3].imshow(rec_10, cmap='gray')
    axs[ph_i, 3].set_title(f"10 ph: ${psnr_(img,rec_10):.2f}$ dB")
    axs[ph_i, 4].imshow(rec_2, cmap='gray')
    axs[ph_i, 4].set_title(f"2 ph: ${psnr_(img,rec_2):.2f}$ dB")

    
# remove axes
for ax in iter(axs.flatten()):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')

# row labels
rows = ['{} photons'.format(row) for row in ph]
for ax, row in zip(axs[:,0], rows):
    ax.set_ylabel(row,  size='large')#, rotation=0,)
    ax.get_yaxis().set_visible(True)
    ax.axis('on')
    #
    #ax.xaxis.set_visible(False)
    plt.setp(ax.spines.values(), visible=False)  # make spines (the box) invisible
    ax.tick_params(left=False, labelleft=False)  # remove ticks and labels for the left axis
    ax.patch.set_visible(False) #remove background patch (only needed for non-white background)
    

f.subplots_adjust(wspace=0, hspace=0)
#plt.suptitle(f"Measurement with ${N0}$ photons $\pm {sig}$")
#plt.savefig("net_denoise.pdf", bbox_inches=0)