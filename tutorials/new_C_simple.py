# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 19:55:16 2021

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
import pdb
from spyrit.misc.statistics import stat_walsh_stl10

#%%
#- Acquisition
img_size = 64 # image size
M = 1024    # number of measurements
N0 = 10     # Image intensity (in photons)

#- Model and data paths
data_root = Path('../data/')
stats_root = Path('../stats_walsh/')

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
testloader =  torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

inputs, _ = next(iter(testloader))


b,c,h,w = inputs.shape

#%%

stat_walsh_stl10()

Cov = np.load(stats_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))
Mean = np.load(stats_root / Path("Average_{}x{}.npy".format(img_size, img_size)))
H =  wh.walsh2_matrix(img_size)/img_size
Cov /= img_size*img_size # THIS NEEDS TO BE NORMALIAZED FOR CONSISTENCY!

#%% Simulate measurements
model = noiCompNet(img_size, M, Mean, Cov, N0=N0, sig=0, H=H)
model = model.to(device)
inputs = inputs.to(device)

torch.manual_seed(0)    # for reproducibility
raw = model.forward_acquire(inputs, b, c, h, w) # with pos/neg coefficients

#%% Reconstruction 
recon = model.forward_reconstruct_mmse(raw, b, c, h, w)

#%% Select two images in the batch
i_im = [71,72]
img = inputs[i_im, 0, :, :].cpu().detach().numpy().astype(np.float32, copy=False)
rec = recon[i_im, 0, :, :].cpu().detach().numpy()

#%% Error
#-- plot
f, axs = plt.subplots(1, 4, figsize=(12,8),  dpi= 100)
axs[0].imshow(img[0], cmap='gray') 
axs[1].imshow(rec[0], cmap='gray')
axs[2].imshow(img[1], cmap='gray')
axs[3].imshow(rec[1], cmap='gray');
axs[0].set_title("ground-truth")
axs[1].set_title("Completion recon")
axs[2].set_title("ground-truth")
axs[3].set_title("Completion recon")
f.suptitle(f'Simulations assuming {N0} photons')
