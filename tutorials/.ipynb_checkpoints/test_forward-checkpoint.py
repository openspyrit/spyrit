# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:03:14 2021

@author: ducros
"""

#%%
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
import spyrit.misc.walsh_hadamard as wh
from spyrit.learning.model_Had_DCAN import *

#%%
#- Acquisition
img_size = 64 # image size
batch_size = 512
M = 512  #number of measurements

#- Model and data paths
data_root = Path('../data/')
stats_root = Path('../stats_walsh/')

#- Save plot using type 1 font
plt.rcParams['pdf.fonttype'] = 42

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

transform = torchvision.transforms.Compose(
    [torchvision.transforms.functional.to_grayscale,
     torchvision.transforms.Resize((img_size, img_size)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0.5], [0.5])])

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
M = 64*64#//4 
#Ord = np.random.rand(h,w)
Ord = Cov2Var(Cov)
model = compNet(img_size, M, Mean, Cov, H=H, Ord=Ord)
model = model.to(device)
inputs = inputs.to(device)

m = model.forward_acquire(inputs, b, c, h, w)
m = model.forward_preprocess(m, b, c, h, w)


#%% Walsh-ordered 2D
i_im = 71
img_size = 64
image = inputs[i_im, 0, :, :];

#-- network
m1 = m[i_im, 0, :,].cpu().detach().numpy()
had_net = meas2img(m1,Ord)

#%%
#-- numpy
img = image.cpu().detach().numpy()#.astype(np.float32, copy=False)
H = wh.walsh_matrix(len(img)).astype(np.float32, copy=False)
had = wh.walsh2(img,H)

#-- torch
outputs = wh.walsh2_torch(inputs)
had_t = outputs[i_im, 0, :, :];
had_t = had_t.cpu().detach().numpy()

#-- error
err = had - had_t
print(np.linalg.norm(err)/np.linalg.norm(img))
err = had - had_net
print(np.linalg.norm(err)/np.linalg.norm(img))

#-- plot
f, axs = plt.subplots(1, 4, figsize=(12,8),  dpi= 100)
axs[0].imshow(img, cmap='gray'); 
axs[1].imshow(had, cmap='gray');
axs[2].imshow(had_t, cmap='gray');
axs[3].imshow(had_net, cmap='gray');
axs[1].set_title("numpy float32")
axs[2].set_title("torch float32")
axs[3].set_title("network")