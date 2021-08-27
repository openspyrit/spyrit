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
import seaborn as sns
import pandas as pd
import sys
sys.path.append('../..')

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
from spyrit.learning.nets import *  # traning, load, visualization...
from spyrit.misc.walsh_hadamard import *  # Hadamard order matrix

###########################
# -- Acquisition parameters
###########################
img_size = 64  # image size
batch_size = 256
M = 1024  # number of measurements
N0 = 50  # maximum photons/pixel in training stage
N0_test = 10  # Noise test level
sig = 0.0  # std of maximum photons/pixel
sig_test = 0.0  # std noise test

#########################
# -- Model and data paths
#########################
data_root = Path('/home/licho/Documentos/Stage/Codes/STL10/')
stats_root = Path('/home/licho/Documentos/Stage/Codes/Test/')
model_root = Path('/home/licho/Documentos/Stage/Codes/Semaine18/Training_Iterative_models/')

My_NVMS_file = Path(stats_root) / (
    'NVMS_N_{}_M_{}.npy'.format(img_size, M))
NVMS = np.load(My_NVMS_file) / N0_test
print('loaded :NVMS_N_{}_M_{}.npy'.format(img_size, M))

################################################################
# -- Precalculated data, Hadamard matrix, and coefficients order
################################################################
Cov = np.load(stats_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))
Mean = np.load(stats_root / Path("Average_{}x{}.npy".format(img_size, img_size)))
H = walsh2_matrix(img_size)/img_size
Cov = Cov / img_size ** 2
Ord = Cov2Var(Cov)

even_index = range(0, 2 * M, 2);
uneven_index = range(1, 2 * M, 2);

########################
# -- STL10 database load
########################
print("Loading STL-10 DATA")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

################
# -- Test images
################
inputs, _ = next(iter(dataloaders['val']))
inputs = inputs.to(device)
b, c, h, w = inputs.shape

#####################
# -- Models statement
#####################
# --network architecture --> ['c0mp', 'comp','pinv', 'free'] --> [0, 1, 2, 3]
net_type = ['NET_c0mp', 'NET_comp', 'NET_pinv', 'NET_free']

# -- Optimisation parameters :
net_arch = 0
# Number of gradient descent iterations
Niter = 5
# Step size of gradient descent
tau = 0.001
# Number of training epochs
num_epochs = 20
# Regularisation Parameter
reg = 1e-7
# Learning Rate
lr = 1e-3
# Scheduler Step Size
step_size = 5
# Scheduler Decrease Rate
gamma = 0.2

##################
# Iterative models
##################

#####################################################################
# model 1 : MMSE + Denoising stage with diagonal matrix approximation
#####################################################################
mmse_diag_iter = DenoiCompNetIter(img_size, M, Mean, Cov, Niter=Niter, tau=tau, variant=net_arch, N0=N0_test, sig=sig_test, H=H, Ord=Ord)
mmse_diag_iter = mmse_diag_iter.to(device)

# -- Load net
suffix_mmse_diag_iter = '_N0_{}_sig_{}_DenoiIter_Niter_{}_tau_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, Niter, tau, img_size, M, num_epochs, lr, step_size, gamma, batch_size, reg)

title_mmse_diag_iter = model_root / (net_type[net_arch] + suffix_mmse_diag_iter)
load_net(title_mmse_diag_iter, mmse_diag_iter, device)

#################################################################################
# model 2 : MMSE + Denoising stage with a first order taylor approximation + NVMS
#################################################################################
mmse_NVMS_iter = DenoiCompNetIterNVMS(img_size, M, Mean, Cov, NVMS=NVMS, Niter=Niter, tau=tau, variant=net_arch, N0=N0_test, sig=sig_test, H=H, Ord=Ord)
mmse_NVMS_iter = mmse_NVMS_iter.to(device)

# -- Load net
suffix_mmse_NVMS_iter = '_N0_{}_sig_{}_DenoiIterNVMS_Niter_{}_tau_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, Niter, tau, img_size, M, num_epochs, lr, step_size, gamma, batch_size, reg)

title_mmse_NVMS_iter = model_root / (net_type[net_arch] + suffix_mmse_NVMS_iter)
load_net(title_mmse_NVMS_iter, mmse_NVMS_iter, device)

# -- Load a model with NVMS's layers adapted to noise level test

mmse_NVMS_stock = DenoiCompNetNVMS(img_size, M, Mean, Cov, NVMS=NVMS, variant=net_arch, N0=N0_test, sig=sig_test, H=H, Ord=Ord)
mmse_NVMS_stock = mmse_NVMS_stock.to(device)

# -- Upload layers weights
mmse_NVMS_iter.fcP0.weight = mmse_NVMS_stock.fcP0.weight
mmse_NVMS_iter.fcP1.weight = mmse_NVMS_stock.fcP1.weight
mmse_NVMS_iter.fcP2.weight = mmse_NVMS_stock.fcP2.weight

#############################
# -- Acquisition measurements
#############################
num_img = 4  # [4,19,123]
b = 1
img_test = inputs[num_img, 0, :, :].view([b, c, h, w])
m = mmse_diag_iter.forward_acquire(img_test, b, c, h, w)  # measures with pos/neg coefficients

#####################
# -- Model evaluation
#####################

f_mmse_diag = mmse_diag_iter.forward_reconstruct(m, b, c, h, w)
f_mmse_nvms = mmse_NVMS_iter.forward_reconstruct(m, b, c, h, w)

###########################
# -- Displaying the results
###########################
# numpy ground-true : We select an image for visual test
GT = img_test.view([h, w]).cpu().detach().numpy()

fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
fig.suptitle('Comparaison des reconstructions en appliquant différents noyaux proposées. '
             'Acquisition effectué avec {} motifs et {} photons. Réseau convolutionel entraîné avec {} photons'.format(M, N0_test, N0), fontsize='large')

ax = axs[0]
im = f_mmse_diag[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('MMSE (Diagonal)')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[1]
im = f_mmse_nvms[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('MMSE (Taylor-NVMS)')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[2]
ax.imshow(GT, cmap='gray')
ax.set_title('Vérité Terrain')

plt.show()

####################################
# -- PSNR test on the validation set
####################################

psnr_mmse_diag_denoi, psnr_NET_mmse_diag_denoi = dataset_psnr(dataloaders['val'], mmse_diag_iter, device, denoise=True, M=M)
print_mean_std(psnr_mmse_diag_denoi, 'MMSE + diag denoi')
print_mean_std(psnr_NET_mmse_diag_denoi, 'MMSE + diag denoi + FCNN')

psnr_mmse_nvms_denoi, psnr_NET_mmse_nvms_denoi = dataset_psnr(dataloaders['val'], mmse_NVMS_iter, device, denoise=True, M=M)
print_mean_std(psnr_mmse_nvms_denoi, 'MMSE + NVMS denoi')
print_mean_std(psnr_NET_mmse_nvms_denoi, 'MMSE + NVMS denoi + FCNN')

#################
# -- PSNR boxplot
#################
plt.rcParams.update({'font.size': 8})
plt.figure()
sns.set_style("whitegrid")
axes = sns.boxplot(data=pd.DataFrame([psnr_mmse_diag_denoi, psnr_NET_mmse_diag_denoi, \
                                      psnr_mmse_nvms_denoi, psnr_NET_mmse_nvms_denoi]).T)

axes.set_xticklabels(['MMSE + diag denoi', 'MMSE + diag denoi + FCNN', \
                      'MMSE + NVMS denoi', 'MMSE + NVMS denoi + FCNN'])
axes.set_ylabel('PSNR')

#######################
# Load training history
#######################
train_path_MMSE_diag_denoi = model_root / ('TRAIN_c0mp' + suffix_mmse_diag_iter + '.pkl')
train_NET_MMSE_diag_denoi = read_param(train_path_MMSE_diag_denoi)

train_path_MMSE_nvms_denoi = model_root / ('TRAIN_c0mp' + suffix_mmse_NVMS_iter + '.pkl')
train_NET_MMSE_nvms_denoi = read_param(train_path_MMSE_nvms_denoi)

plt.rcParams.update({'font.size': 12})

##################
# -- Training Plot
##################
fig1, ax = plt.subplots(figsize=(10, 6))
plt.title('Comparison of loss curves for Pseudo inverse, and MMSE models from training with {} photons'.format(N0), fontsize=16)
ax.set_xlabel('Time (epochs)')
ax.set_ylabel('Loss (MSE)')
ax.plot(train_NET_MMSE_diag_denoi.val_loss, 'b', linewidth=1.5)
ax.plot(train_NET_MMSE_nvms_denoi.val_loss, 'c', linewidth=1.5)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.grid(True)
ax.legend((' MMSE + denoi (diagonal approximation) :  38m 60s',\
           ' MMSE + denoi (Taylor inversion) + NVMS : 39m 19s'),  loc='upper right')












