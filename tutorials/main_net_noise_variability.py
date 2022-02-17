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
from scipy.sparse.linalg import aslinearoperator
import pylops

#mu = 1.5
def TV(y, H, img_size, mu = 0.15, lamda = [0.1, 0.1], niter = 20, niterinner = 10):
    ny = img_size;
    nx = img_size;
    A = aslinearoperator(H);
    H_p = pylops.LinearOperator(A)
    Dop = \
        [pylops.FirstDerivative(ny * nx, dims=(ny, nx), dir=0, edge=False,
                                kind='backward', dtype=np.float64),
         pylops.FirstDerivative(ny * nx, dims=(ny, nx), dir=1, edge=False,
                                kind='backward', dtype=np.float64)]
    xinv, niter = \
    pylops.optimization.sparsity.SplitBregman(H_p, Dop, y.flatten(),
                                              niter, niterinner,
                                              mu=mu, epsRL1s=lamda,
                                              tol=1e-4, tau=1., show=False,
                                              **dict(iter_lim=5, damp=1e-4))
    return xinv;

#%%
#- Acquisition
img_size = 64 # image size
batch_size = 256
M = 512  #number of measurements

#- Model and data paths
data_root = Path('../data/')
stats_root = Path('../stats_walsh/')

#- Plot options
plt.rcParams['pdf.fonttype'] = 42   # Save plot using type 1 font
plt.rcParams['text.usetex'] = True  # Latex
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = \
    torchvision.datasets.STL10(root=data_root, split='train+unlabeled',download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)

testset = \
    torchvision.datasets.STL10(root=data_root, split='test',download=True, transform=transform)
testloader =  torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)

dataloaders = {'train':trainloader, 'val':testloader}

#%%
inputs, _ = next(iter(dataloaders['val']))
b,c,h,w = inputs.shape

Cov = np.load(stats_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))
Mean = np.load(stats_root / Path("Average_{}x{}.npy".format(img_size, img_size)))
H =  wh.walsh2_matrix(img_size)/img_size

#%% STL-10 Images and network
N0 = 50 # mean of the maximun number of photons
sig = 0.5 # range in prct of the maximun number of photons

M = 64*64//4
Ord = Cov2Var(Cov)
model = noiCompNet(img_size, M, Mean, Cov, variant=0, N0=N0, sig=sig, H=H, Ord=Ord)

model = model.to(device)
inputs = inputs.to(device)

#%%
meas = model.forward_acquire(inputs, b, c, h, w)    # with pos/neg coefficients
hadam = model.forward_preprocess(meas, b, c, h, w)  # hadamard coefficient normalized

recon_mmse = model.forward_reconstruct_mmse(meas, b, c, h, w).cpu().detach().numpy()
recon_pinv = model.forward_reconstruct_pinv(meas, b, c, h, w).cpu().detach().numpy()


#%% Load and recon
# noiseless
model_root = '../models/NET_c0mp_N_64_M_1024_epo_40_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07' 
title  = 'model_epoch_40'
load_net(model_root / Path(title), model, device)
recon_nonoise = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()

# 50 photons, 50% variability
model_root = '../models/NET_c0mp_N0_50.0_sig_0.5_N_64_M_1024_epo_50_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07'
title  = 'model_epoch_40'
load_net(model_root / Path(title), model, device)
recon_50_var = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()

# 50 photons, no variability
model_root = '../models/NET_c0mp_N0_50.0_sig_0.0_N_64_M_1024_epo_50_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07'
title  = 'model_epoch_40'
load_net(model_root / Path(title), model, device)
recon_50_novar = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()

# 10 photons, 50% variability
model_root = '../models/'
title  = 'NET_c0mp_N0_10.0_sig_0.5_N_64_M_1024_epo_40_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07'
load_net(model_root / Path(title), model, device)
recon_10_var = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()

# 10 photons, no variability
model_root = '../models/'
title  = 'NET_c0mp_N0_10.0_sig_0.0_N_64_M_1024_epo_40_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07'
load_net(model_root / Path(title), model, device)
recon_10_novar = model.forward_reconstruct(meas, b, c, h, w).cpu().detach().numpy()


#%% Recon from Walsh-ordered 2D
ind = [72,73,74,75]


#%% Plot
f, axs = plt.subplots(4, 9, figsize=(10,10),  dpi= 100)

for i_ind,v_ind in enumerate(ind): 
   
    #-- Recon
    img = inputs[v_ind, 0, :, :].cpu().detach().numpy()
    had = hadam[v_ind, 0, :].cpu().detach().numpy()

    rec_mmse = recon_mmse[v_ind, 0, :, :]
    rec_pinv = recon_pinv[v_ind, 0, :, :]
    rec_nonoise  = recon_nonoise[v_ind, 0, :, :]
    rec_50_var   = recon_50_var[v_ind, 0, :, :]
    rec_50_novar = recon_50_novar[v_ind, 0, :, :]
    rec_10_var   = recon_10_var[v_ind, 0, :, :]
    rec_10_novar = recon_10_novar[v_ind, 0, :, :]
    
    # compare to rec_pinv : OK, same results !
    #rec_l2 = wh.iwalsh2(meas2img(had, Ord))
    
    H_sub = subsample(H, Ord, M)
    rec_tv = TV(had, H_sub, img_size, mu = 1e-2, niter = 20).reshape(img_size,img_size)   
    rec_tv /= img_size # CHECK THIS OUT
    
    #- Plot   
    axs[i_ind, 0].imshow(img, cmap='gray')
    axs[i_ind, 0].set_title("Ground-truth")
    axs[i_ind, 0].get_xaxis().set_visible(False)
    axs[i_ind, 0].get_yaxis().set_visible(False)
    axs[i_ind, 0].axis('off')
    axs[i_ind, 1].imshow(rec_pinv, cmap='gray')
    axs[i_ind, 1].set_title(f"PINV: ${psnr_(img,rec_pinv):.2f}$ dB")
    axs[i_ind, 2].imshow(rec_mmse, cmap='gray')
    axs[i_ind, 2].set_title(f"MMSE: ${psnr_(img,rec_mmse):.2f}$ dB")
    axs[i_ind, 3].imshow(rec_tv, cmap='gray')
    axs[i_ind, 3].set_title(f"CS: ${psnr_(img,rec_tv):.2f}$ dB")
    axs[i_ind, 4].imshow(rec_nonoise, cmap='gray')
    axs[i_ind, 4].set_title(f"NET no noise: ${psnr_(img,rec_nonoise):.2f}$ dB")
    axs[i_ind, 5].imshow(rec_50_var, cmap='gray')
    axs[i_ind, 5].set_title(f"NET 50 ph var: ${psnr_(img,rec_50_var):.2f}$ dB")
    axs[i_ind, 6].imshow(rec_50_novar, cmap='gray')
    axs[i_ind, 6].set_title(f"NET 50 ph novar: ${psnr_(img,rec_50_novar):.2f}$ dB")
    axs[i_ind, 7].imshow(rec_10_var, cmap='gray')
    axs[i_ind, 7].set_title(f"NET 10 ph var: ${psnr_(img,rec_10_var):.2f}$ dB")
    axs[i_ind, 8].imshow(rec_10_novar, cmap='gray')
    axs[i_ind, 8].set_title(f"NET 10 ph novar: ${psnr_(img,rec_10_novar):.2f}$ dB")

f.subplots_adjust(wspace=0, hspace=0)
plt.suptitle(f"Measurement with ${N0}$ photons $\pm {sig}$")
#plt.savefig("net.pdf", bbox_inches=0)

#%%
# Load training history
train_path = model_root/Path('TRAIN_c0mp'+suffix+'.pkl')
train_net_prob = read_param(train_path)
train_path = model_root / Path('TRAIN_pinv'+suffix+'.pkl')
train_net_pinv = read_param(train_path)
train_path = model_root/ Path('TRAIN_free'+suffix+'.pkl')
train_net_free = read_param(train_path)

#plt.rcParams.update({'font.size': 20})

# Plot
fig, ax = plt.subplots(figsize=(10,6))
ax.set_xlabel('Time (epochs)')
ax.set_ylabel('Loss (MSE)')
ax.plot(train_net_pinv.val_loss,'g--', linewidth=4)
ax.plot(train_net_prob.val_loss,'r-.', linewidth=4)
ax.plot(train_net_free.val_loss,'m', linewidth=4)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.grid(True)
ax.legend(('pinvNET', 'compNET', 'freeNET'),  loc='upper right')
#fig.savefig('loss_test.pdf', dpi=fig.dpi, bbox_inches='tight')# pad_inches=0.1)

#%%


suffix = '_N_64_M_1024_epo_120_lr_0.001_sss_20_sdr_0.2_bs_512_reg_1e-07'
# Load training history
train_path = model_root/Path('TRAIN_c0mp_N0_250.0_sig_0.5_Denoi'+suffix+'.pkl')
train_net_prob = read_param(train_path)
train_path = model_root / Path('TRAIN_c0mp_N0_250.0_sig_0.5'+suffix+'.pkl')
train_net_pinv = read_param(train_path)
#train_path = model_root/ Path('TRAIN_free'+suffix+'.pkl')
#train_net_free = read_param(train_path)

#plt.rcParams.update({'font.size': 20})

# Plot
fig, ax = plt.subplots(figsize=(10,6))
ax.set_xlabel('Time (epochs)')
ax.set_ylabel('Loss (MSE)')
ax.plot(train_net_pinv.val_loss,'g--', linewidth=4)
ax.plot(train_net_prob.val_loss,'r-.', linewidth=4)
#ax.plot(train_net_free.val_loss,'m', linewidth=4)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.grid(True)
ax.legend(('none', 'denoised'),  loc='upper right')
#fig.savefig('loss_test.pdf', dpi=fig.dpi, bbox_inches='tight')# pad_inches=0.1)

