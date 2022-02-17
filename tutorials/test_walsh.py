# -*- coding: utf-8 -*-
"""
Computes Hadamard tranforms of images and compares implementations based on numpy and torch.

Created on Mon Jun 21 17:04:22 2021

@author: Nicolas Ducros
"""

#%%
#from __future__ import print_function, division
import torch
import numpy as np
import torchvision
#from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
import spyrit.misc.walsh_hadamard as wh

#%%
#- Acquisition
img_size = 64 # image size
batch_size = 512

#- Model and data paths
data_root = Path('../data/')

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

#%% Walsh ordered 2D on STL-10
inputs, classes = next(iter(dataloaders['val']))
img = inputs[81, 0, :, :];
img = img.cpu().detach().numpy()

img_size = 64
had = wh.walsh2(img)
im1 = wh.iwalsh2(had)
err = img - im1
print(np.linalg.norm(err)/np.linalg.norm(img))

f, axs = plt.subplots(1, 4, figsize=(12,8),  dpi= 100)
axs[0].imshow(img, cmap='gray'); 
axs[1].imshow(had, cmap='gray');
axs[2].imshow(im1, cmap='gray');
axs[3].imshow(err, cmap='gray');
axs[1].set_title("hadamard transform")
axs[2].set_title("inverse transform")
axs[3].set_title("difference")


#%% Walsh-ordered 2D on STL-10
inputs, classes = next(iter(dataloaders['val']))
i_im = 71
img_size = 64
img = inputs[i_im, 0, :, :];

#-- numpy
img = img.cpu().detach().numpy().astype(np.float32, copy=False)
H = wh.walsh_matrix(len(img)).astype(np.float32, copy=False)
had = wh.walsh2(img,H)
#-- torch
outputs = wh.walsh2_torch(inputs)
had_t = outputs[i_im, 0, :, :];
had_t = had_t.cpu().detach().numpy()
#-- error
err = had - had_t
print(np.linalg.norm(err)/np.linalg.norm(img))

#-- plot
f, axs = plt.subplots(1, 4, figsize=(12,8),  dpi= 100)
axs[0].imshow(img, cmap='gray'); 
axs[1].imshow(had, cmap='gray');
axs[2].imshow(had_t, cmap='gray');
axs[3].imshow(err, cmap='gray');
axs[1].set_title("numpy float32")
axs[2].set_title("torch float32")
axs[3].set_title("difference")

#%% Computation times
import time
#-- numpy
H = wh.walsh_matrix(len(img)).astype(np.float32, copy=False)
time_start = time.perf_counter()
outputs = wh.walsh2(img,H)
time_np = (time.perf_counter() - time_start)
print(f'1 Numpy image: {time_np}')

#-- torch on CPU
H = torch.from_numpy(H)
time_start = time.perf_counter()
outputs = wh.walsh2_torch(inputs,H)
time_tcpu = (time.perf_counter() - time_start)
print(f'{inputs.shape[0]} Torch batch (cpu): {time_tcpu}')

#-- torch on GPU
H = H.to(device)
inputs_cuda = inputs.to(device)
time_start = time.perf_counter()
outputs = wh.walsh2_torch(inputs_cuda,H)
time_tgpu = (time.perf_counter() - time_start)
print(f'{inputs.shape[0]} Torch batch (cuda): {time_tgpu}')

#-- ratios
print(f'{inputs.shape[0]} Numpy / {inputs.shape[0]} torch (cpu): {inputs.shape[0]*time_np/time_tcpu}')
print(f'{inputs.shape[0]} Numpy / {inputs.shape[0]} torch (cuda): {inputs.shape[0]*time_np/time_tgpu}')
