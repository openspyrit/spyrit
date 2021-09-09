# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 12:12:42 2021

@author: ducros + amador
"""
#############
## -- Imports
#############
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import dct

# -- Pytorch tools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# -- Dataloading tools
import torchvision
from torchvision import datasets, models, transforms

# -- Spyrit packages
from spyrit.learning.model_Had_DCAN import *  # models
from spyrit.misc.metrics import *  # psnr metrics
from spyrit.misc.walsh_hadamard import *  # Hadamard order matrix
from spyrit.learning.nets import *  # traning, load, visualization...

###########################
# -- Acquisition parameters
###########################
img_size = 64  # image size
batch_size = 256
M = 1024  # number of measurements
N0 = 50  # maximum photons/pixel in training stage
sig = 0.0  # std of maximum photons/pixel

#########################
# -- Model and data paths
#########################
data_root = Path('/home/licho/Documentos/Stage/Codes/STL10')
stats_root = Path('/home/licho/Documentos/Stage/Codes/Test')
model_root = Path('/home/licho/Documentos/Stage/Codes/Semaine20/Training_free')

########################
# -- STL10 database load
########################
print("Loading STL-10 DATA")
device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(9)

transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = \
    torchvision.datasets.STL10(root=data_root, split='train+unlabeled', download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = \
    torchvision.datasets.STL10(root=data_root, split='test', download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

dataloaders = {'train': trainloader, 'val': testloader}
print("dataloaders are ready")

################################################################
# -- Precalculated data, Hadamard matrix, and coefficients order
################################################################
Cov = np.load(stats_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))
Mean = np.load(stats_root / Path("Average_{}x{}.npy".format(img_size, img_size)))
H = walsh2_matrix(img_size)/img_size
Cov = Cov / img_size ** 2
Ord = Cov2Var(Cov)

#####################
# -- Models statement
#####################

# --network architecture --> ['c0mp', 'comp','pinv', 'free'] --> [0, 1, 2, 3]
net_type = ['NET_c0mp', 'NET_comp', 'NET_pinv', 'NET_free']

# -- Optimisation parameters :
net_arch = 3
# Number of training epochs
num_epochs = 30
# Regularisation Parameter
reg = 1e-7
# Learning Rate
lr = 1e-3
# Scheduler Step Size
step_size = 10
# Scheduler Decrease Rate
gamma = 0.5

# -- Load net
suffix = '_N0_{}_sig_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, img_size, M, num_epochs, lr, step_size, gamma, batch_size, reg)

title = model_root / (net_type[net_arch] + suffix)

################################
# model 1 : free (Test : 100 ph)
################################
free_100 = noiCompNet(img_size, M, Mean, Cov, variant=net_arch, N0=100, sig=sig, H=H, Ord=Ord)
free_100 = free_100.to(device)

# -- Load net
load_net(title, free_100, device)

###############################
# model 2 : free (Test : 25 ph)
###############################
free_25 = noiCompNet(img_size, M, Mean, Cov, variant=net_arch, N0=25, sig=sig, H=H, Ord=Ord)
free_25 = free_25.to(device)

# -- Load net
load_net(title, free_25, device)

###############################
# model 3 : free (Test : 10 ph)
###############################
free_10 = noiCompNet(img_size, M, Mean, Cov, variant=net_arch, N0=10, sig=sig, H=H, Ord=Ord)
free_10 = free_10.to(device)

# -- Load net
load_net(title, free_10, device)

################
# -- Test images
################
inputs, _ = next(iter(dataloaders['val']))
inputs = inputs.to(device)
b, c, h, w = inputs.shape

#############################
# -- Acquisition measurements
#############################
num_img = 200  # [4,19,123]
b = 1
img_test = inputs[num_img, 0, :, :].view([b, c, h, w])
m_100 = free_100.forward_acquire(img_test, b, c, h, w)  # measures with pos/neg coefficients
m_25 = free_25.forward_acquire(img_test, b, c, h, w)
m_10 = free_10.forward_acquire(img_test, b, c, h, w)

#####################
# -- Model evaluation
#####################

f_free_100 = free_100.forward_reconstruct(m_100, b, c, h, w)
f_free_25 = free_25.forward_reconstruct(m_25, b, c, h, w)
f_free_10 = free_10.forward_reconstruct(m_10, b, c, h, w)

###########################
# -- Displaying the results
###########################
# numpy ground-true : We select an image for visual test
GT = img_test.view([h, w]).cpu().detach().numpy()

fig, axs = plt.subplots(nrows=1, ncols=4, constrained_layout=True)
fig.suptitle('Comparaison des reconstructions en appliquant différents methodes proposées. '
             'Acquisition effectué avec {} motifs. Reseau entraine avec {} photons.'.format(M, N0), fontsize='large')

#################################

ax = axs[0]
ax.imshow(GT, cmap='gray')
ax.set_title('Vérité Terrain')

ax = axs[1]
im = f_free_100[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('100 ph')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[2]
im = f_free_25[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('25 ph')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[3]
im = f_free_10[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('10 ph')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

plt.tight_layout()
plt.show()

#######################
# Load training history
#######################
train_path_free = model_root / ('TRAIN_free' + suffix + '.pkl')
train_NET_free = read_param(train_path_free)

plt.rcParams.update({'font.size': 12})

##################
# -- Training Plot
##################
fig1, ax = plt.subplots(figsize=(10, 6))
plt.title('Loss curve for the free model. Training with {} photons'.format(N0), fontsize=16)
ax.set_xlabel('Time (epochs)')
ax.set_ylabel('Loss (MSE)')
ax.plot(train_NET_free.val_loss, 'k', linewidth=1.5)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.grid(True)
ax.legend(('2500 ph : 40m 12s', '50 ph : 40m 02s', '10 ph : 40m 27s'),  loc='upper right')

