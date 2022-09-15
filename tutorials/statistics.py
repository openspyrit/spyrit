# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms
from pathlib import Path
from spyrit.misc.disp import imagesc


#%% Running the cell twice give the same results
from spyrit.misc.statistics import data_loaders_ImageNet

img_size = 64 # image size
batch_size = 128
data_root =  Path('../../datasets/ILSVRC2012_v10102019')
stat_root =  Path('../../stat/ILSVRC2012_v10102019')
dataloaders = data_loaders_ImageNet(data_root / 'test', data_root / 'val', 
                                    img_size=img_size, 
                                    batch_size=batch_size, 
                                    seed=7)

# Plot a train image
torch.manual_seed(2) # reproductibility of random crop
inputs, _ = next(iter(dataloaders['train']))
b,c,h,w = inputs.shape

x = inputs.view(b*c,w*h)

img = x[4,:].numpy().reshape((h,w))
imagesc(img)

# Plot a validation image
torch.manual_seed(2) # reproductibility of random crop
inputs, _ = next(iter(dataloaders['val']))
b,c,h,w = inputs.shape

x = inputs.view(b*c,w*h)

img = x[4,:].numpy().reshape((h,w))
imagesc(img)

#%% Compute covariance matrix from stl10
img_size = 64 # image size
batch_size = 1024
data_root =  Path('../../datasets/')
stat_root =  Path('../../stat/stl10')

from spyrit.misc.statistics import stat_walsh_stl10
stat_walsh_stl10(stat_root = stat_root, 
                 data_root = data_root,
                 img_size = img_size, 
                 batch_size = batch_size)

#%% Compute covariance matrix from ImageNet
img_size = 64 # image size
batch_size = 1024
data_root =  Path('../../datasets/ILSVRC2012_v10102019/test')
stat_root =  Path('../../stat/ILSVRC2012_v10102019')

from spyrit.misc.statistics import stat_walsh_ImageNet

stat_walsh_ImageNet(stat_root = stat_root, 
                    data_root = data_root,
                    img_size = img_size, 
                    batch_size = batch_size, 
                    n_loop=2)
