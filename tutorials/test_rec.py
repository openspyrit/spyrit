# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:03:14 2021

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

#%%
#- Acquisition
img_size = 64 # image size
batch_size = 512
M = 512  #number of measurements

#- Model and data paths
data_root = Path('./data/')
stats_root = Path('./stats_walsh/')

#- Save plot using type 1 font
plt.rcParams['pdf.fonttype'] = 42

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

#%% completion network
M = 64*64//4 
np.random.seed(0)
Ord = np.random.rand(h,w)
Ord = Cov2Var(Cov)
model = compNet(img_size, M, Mean, Cov, variant=2, H=H, Ord=Ord)
model = model.to(device)
inputs = inputs.to(device)

raw = model.forward_acquire(inputs, b, c, h, w) # with pos/neg coefficients
meas = model.forward_preprocess(raw, b, c, h, w)

i_im = 71
img = inputs[i_im, 0, :, :].cpu().detach().numpy().astype(np.float32, copy=False)
m = meas[i_im, 0, :].cpu().detach().numpy()

#%% Recon from net
recon = model.forward_maptoimage(raw, b, c, h, w)
rec1 = recon[i_im, 0, :, :].cpu().detach().numpy()

#%% Recon from Walsh-ordered 2D
y = meas2img(m, Ord)
rec2 = wh.iwalsh2(y)

#%% Error
#-- numpy
err1 = img - rec1
err2 = img - rec2
print(f'Error from net: {np.linalg.norm(err1)/np.linalg.norm(img)}')
print(f'Error from inverse : {np.linalg.norm(err2)/np.linalg.norm(img)}')

#-- plot
f, axs = plt.subplots(1, 4, figsize=(12,8),  dpi= 100)
axs[0].imshow(img, cmap='gray') 
axs[1].imshow(rec1, cmap='gray')
axs[2].imshow(rec2, cmap='gray')
axs[3].imshow(rec1-rec2, cmap='gray');
axs[0].set_title("ground-truth")
axs[1].set_title("recon from NET")
axs[2].set_title("recon from inverse")
axs[3].set_title("NET_inverse")