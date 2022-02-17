# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 17:55:12 2021

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
batch_size = 128
M = 512  #number of measurements

#- Model and data paths
data_root = Path('./data/')
stats_root = Path('./stats_walsh/')

#- Plot options
plt.rcParams['pdf.fonttype'] = 42   # Save plot using type 1 font
plt.rcParams['text.usetex'] = True  # Latex
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

transform = transforms.Compose(
    [#transforms.functional.to_grayscale,
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
M = 64*64//4
Ord = Cov2Var(Cov)
model = compNet(img_size, M, Mean, Cov, variant=0, H=H, Ord=Ord)

model = model.to(device)
inputs = inputs.to(device)

meas = model.forward_acquire(inputs, b, c, h, w)    # with pos/neg coefficients
hadam = model.forward_preprocess(meas, b, c, h, w)  # hadamard coefficient normalized
recon_mmse = model.forward_reconstruct_mmse(meas, b, c, h, w)
recon_pinv = model.forward_reconstruct_pinv(meas, b, c, h, w)


#%%
model_root = './models/'
suffix = '_N_64_M_1024_epo_30_lr_0.001_sss_20_sdr_0.2_bs_512_reg_1e-07'
title = 'NET_pinv'
#title = 'NET_free'
#title = 'NET_c0mp'
load_net(model_root / Path(title+suffix), model, device)
recon_net = model.forward_reconstruct(meas, b, c, h, w)

#%% Recon from Walsh-ordered 2D
ind = [72,73,74,75]
# M = [2048, 1024, 512,256] 
# sig = [1,]#[0.1, 0.25, 2, 16] 
# #eps = np.random.standard_normal((M,))


#%% Plot
f, axs = plt.subplots(4, 5, figsize=(10,10),  dpi= 100)

np.random.seed(0)
Ord = np.random.rand(h,w)
Ord = Cov2Var(Cov)

for i_ind,v_ind in enumerate(ind): 
   
    #-- Recon
    img = inputs[v_ind, 0, :, :].cpu().detach().numpy()
    had = hadam[v_ind, 0, :].cpu().detach().numpy()

    rec_mmse = recon_mmse[v_ind, 0, :, :].cpu().detach().numpy()
    rec_pinv = recon_pinv[v_ind, 0, :, :].cpu().detach().numpy()
    rec_net = recon_net[v_ind, 0, :, :].cpu().detach().numpy()
    
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
    axs[i_ind, 4].imshow(rec_net, cmap='gray')
    axs[i_ind, 4].set_title(f"NET: ${psnr_(img,rec_net):.2f}$ dB")


f.subplots_adjust(wspace=0, hspace=0)
plt.savefig("net.pdf", bbox_inches=0)

#%%
# Load training history
train_path = model_root/Path('TRAIN_c0mp'+suffix+'.pkl')
#train_net_prob = read_param(train_path)
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
#ax.plot(train_net_prob.val_loss,'r-.', linewidth=4)
ax.plot(train_net_free.val_loss,'m', linewidth=4)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.grid(True)
ax.legend(('pinvNET', 'compNET', 'freeNET'),  loc='upper right')
#fig.savefig('loss_test.pdf', dpi=fig.dpi, bbox_inches='tight')# pad_inches=0.1)