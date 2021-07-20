from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import imageio
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import PIL
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import wget
import zipfile
from pathlib import Path

# Import important functions from spyrit
from spyrit.learning.model_Had_DCAN import *# compNet, Stat_had, Weight_Decay_Loss
from spyrit.learning.nets import *
from spyrit.misc.disp import *
from spyrit.misc.metrics import *

# User-defined global parameters
#- Acquisition
img_size = 64 # image size
M = 333       # number of neasurements
#- Training
num_epochs = 60
lr = 1e-3 
step_size = 20
gamma = 0.2
batch_size =256
reg = 1e-7
#- Model and data paths
model_root = Path('./models/')
stats_root = Path('./stats/')
data_root = Path('./data/')
#- Save plot using type 1 font
plt.rcParams['pdf.fonttype'] = 42

print(model_root)
print(data_root)

# Download models and covariance
if not(Path('./models/').exists()):
    wget.download('https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2020_ISBI_CNet/2020_ISBI_CNet.zip')
    with zipfile.ZipFile('2020_ISBI_CNet.zip', 'r') as zip_ref:
        zip_ref.extractall()
    Path('2020_ISBI_CNet.zip').unlink()

# Load STL-10 dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])


trainset = \
    torchvision.datasets.STL10(root=data_root, split='train',download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)

testset = \
    torchvision.datasets.STL10(root=data_root, split='test',download=True, transform=transform)
testloader =  torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)

dataloaders = {'train':trainloader, 'val':testloader}

# Load trained networks
print('Loading Cov and Mean')    
Cov_had = np.load(stats_root / "Cov_{}x{}.npy".format(img_size, img_size))
Mean_had = np.load(stats_root / 'Average_{}x{}.npy'.format(img_size, img_size))

# Statistical completion
suffix = '_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
         img_size, M, num_epochs, lr, step_size, gamma, batch_size, reg)


model = compNet(img_size,M, Mean_had,Cov_had)
model = model.to(device)
title = model_root/('NET_c0mp'+ suffix)
load_net(title, model, device)

psnr_net_prob, psnr_prob = dataset_psnr(dataloaders['val'], model, device);
     
print_mean_std(psnr_net_prob,'compNet: ')
print_mean_std(psnr_prob,'comp: ')

print('Number of trainable parameters: {}'.format(count_trainable_param(model)))
print('Total number of parameters: {}'.format(count_param(model)))

#Pseudo Inverse
model = compNet(img_size,M, Mean_had,Cov_had,2)
model = model.to(device)
title = model_root/('NET_pinv'+ suffix)
load_net(title, model, device)

psnr_net_pinv, psnr_pinv = dataset_psnr(dataloaders['val'], model, device)

print_mean_std(psnr_net_pinv,'pinvNet: ')
print_mean_std(psnr_pinv,'PInv: ')
print('Number of trainable parameters: {}'.format(count_trainable_param(model)))
print('Total number of parameters: {}'.format(count_param(model)))

# Fully learnt (free Net)
model = compNet(img_size,M, Mean_had,Cov_had,3)
model = model.to(device)
title = model_root/('NET_free'+ suffix)
load_net(title, model, device)
#    
psnr_net_free, psnr_free = dataset_psnr(dataloaders['val'], model, device)

print_mean_std(psnr_net_free,'freeNet: ')
print_mean_std(psnr_free,'free: ')

print('Number of trainable parameters: {}'.format(count_trainable_param(model)))
print('Total number of parameters: {}'.format(count_param(model)))


#%% Load measured data
meas = dataset_meas(dataloaders['val'], model, device) #dataloaders['train']
meas = np.array(meas)

#%%
n1 = 2; #2,12 or 2,7
n2 = 7;


# Load training history
train_path = model_root/('TRAIN_c0mp'+suffix+'.pkl')
train_net_prob = read_param(train_path)
train_path = model_root/('TRAIN_pinv'+suffix+'.pkl')
train_net_pinv = read_param(train_path)
train_path = model_root/('TRAIN_free'+suffix+'.pkl')
train_net_free = read_param(train_path)


