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
img_size = 64   # image size
M = 1024        # number of measurements

#- Model and data paths
data_root = Path('../data/')
stats_root = Path('../stats_walsh/')

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
model = compNet(img_size, M, Mean, Cov, H=H)
model = model.to(device)
inputs = inputs.to(device)

meas = model.forward_acquire(inputs, b, c, h, w)    # with pos/neg coefficients
hadam = model.forward_preprocess(meas, b, c, h, w)  # hadamard coefficient normalized
recon_mmse = model.forward_reconstruct_mmse(meas, b, c, h, w)
recon_pinv = model.forward_reconstruct_pinv(meas, b, c, h, w)


#%%
#model_root = '../models/'
#model_title = 'NET_c0mp_N_64_M_1024_epo_30_lr_0.001_sss_20_sdr_0.2_bs_512_reg_1e-07'
model_root = '../models_v1.2/' 
model_title  = 'NET_c0mp_N_64_M_1024_epo_40_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07'

load_net(model_root / Path(model_title), model, device)
recon_net = model.forward_reconstruct(meas, b, c, h, w)

#%% Recon from Walsh-ordered 2D
ind = [72,73,74,75]

#%% Plot
f, axs = plt.subplots(4, 5, figsize=(10,10),  dpi= 100)

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

# remove axes
for ax in iter(axs.flatten()):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')
    
f.subplots_adjust(wspace=0, hspace=0)
#plt.savefig("net.pdf", bbox_inches=0)