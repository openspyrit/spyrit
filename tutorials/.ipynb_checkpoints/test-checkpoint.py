# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:05:04 2021

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
from spyrit.misc.disp import *

#%%
#- Acquisition
img_size = 64 # image size
batch_size = 256
M = 512  #number of measurements

#- Model and data paths
data_root = Path('./data/')
stats_root = Path('./stats_walsh/')

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
Cov /= img_size**2

#%%
M = 64*64//4
Ord = Cov2Var(Cov)
N0 = 10 # mean of the maximun number of photons
model = DenoiCompNet(img_size, M, Mean, Cov, variant=0, N0=N0, sig=0, H=H, Ord=Ord)
model = model.to(device)
inputs = inputs.to(device)

#%%
model_root = './models/'
title  = 'NET_c0mp_N0_10.0_sig_0.0_Denoi_N_64_M_1024_epo_40_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07'
load_net(model_root / Path(title), model, device)
img = inputs[1, 0, :, :].cpu().detach().numpy()

torch.manual_seed(0) # for reproducibility
rec = model(inputs)[1, 0, :, :].cpu().detach().numpy()

#%%
imagesc(img)
imagesc(rec)
print(psnr_(img,rec))

#%%
b,c,h,w = inputs.shape

torch.manual_seed(0) # for reproducibility
meas = model.forward_acquire(inputs, b, c, h, w)
rec = model.forward_reconstruct_mmse(meas, b, c, h, w)[1,0,:,:].cpu().detach().numpy()
imagesc(rec)
print(psnr(img,rec))