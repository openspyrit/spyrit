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
N0_test = 2500  # Noise test level
sig = 0.0  # std of maximum photons/pixel
sig_test = 0.0  # std noise test

#########################
# -- Model and data paths
#########################
data_root = Path('/home/amador/Documents/python-virtual-environments/STL10')
stats_root = Path('/home/amador/Documents/Stage/Codes/spyrit-doc/Test')
model_root = Path('/home/amador/Documents/Stage/Codes/Semaine17/Training_models/fix50ph_Models/')

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

##########################
# model 1 : Pseudo inverse
##########################
net_arch = 2
pinv = noiCompNet(img_size, M, Mean, Cov, variant=net_arch, N0=N0_test, sig=sig_test, H=H, Ord=Ord)
pinv = pinv.to(device)

# -- Load net
suffix_Pinv = '_N0_{}_sig_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, img_size, M, num_epochs, lr, step_size, gamma, batch_size, reg)

title_Pinv = model_root / (net_type[net_arch] + suffix_Pinv)
load_net(title_Pinv, pinv, device)

########################################
# model 2 : MMSE without denoising stage
########################################
net_arch = 0
mmse = noiCompNet(img_size, M, Mean, Cov, variant=net_arch, N0=N0_test, sig=sig_test, H=H, Ord=Ord)
mmse = mmse.to(device)

# -- Load net
suffix_mmse = '_N0_{}_sig_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, img_size, M, num_epochs, lr, step_size, gamma, batch_size, reg)

title_mmse = model_root / (net_type[net_arch] + suffix_mmse)
load_net(title_mmse, mmse, device)

#####################################################################
# model 3 : MMSE + Denoising stage with diagonal matrix approximation
#####################################################################
net_arch = 0
mmse_diag = DenoiCompNet(img_size, M, Mean, Cov, variant=net_arch, N0=N0_test, sig=sig_test, H=H, Ord=Ord)
mmse_diag = mmse_diag.to(device)

# -- Load net
suffix_mmse_diag = '_N0_{}_sig_{}_Denoi_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, img_size, M, num_epochs, lr, step_size, gamma, batch_size, reg)

title_mmse_diag = model_root / (net_type[net_arch] + suffix_mmse_diag)
load_net(title_mmse_diag, mmse_diag, device)

########################################################################
# model 3 : MMSE + Denoising stage with first order Taylor approximation
########################################################################
net_arch = 0
mmse_taylor = DenoiCompNetApprox(img_size, M, Mean, Cov, variant=net_arch, N0=N0_test, sig=sig_test, H=H, Ord=Ord)
mmse_taylor = mmse_taylor.to(device)

# -- Load net
suffix_mmse_taylor = '_N0_{}_sig_{}_DenoiApprox_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, img_size, M, num_epochs, lr, step_size, gamma, batch_size, reg)

title_mmse_taylor = model_root / (net_type[net_arch] + suffix_mmse_taylor)
load_net(title_mmse_taylor, mmse_taylor, device)


#############################################################
# model 4 : MMSE + Denoising stage with full matrix inversion
#############################################################
net_arch = 0
mmse_full = DenoiCompNetFull(img_size, M, Mean, Cov, variant=net_arch, N0=N0_test, sig=sig_test, H=H, Ord=Ord)
mmse_full = mmse_full.to(device)

# -- Load net
suffix_mmse_full = '_N0_{}_sig_{}_DenoiFull_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, img_size, M, num_epochs, lr, step_size, gamma, batch_size, reg)

title_mmse_full = model_root / (net_type[net_arch] + suffix_mmse_full)
load_net(title_mmse_full, mmse_full, device)

#################################################################################
# model 5 : MMSE + Denoising stage with a first order taylor approximation + NVMS
#################################################################################
net_arch = 0
mmse_NVMS = DenoiCompNetNVMS(img_size, M, Mean, Cov, NVMS=NVMS, variant=net_arch, N0=N0_test, sig=sig_test, H=H, Ord=Ord)
mmse_NVMS = mmse_NVMS.to(device)

# -- Load net
suffix_mmse_NVMS = '_N0_{}_sig_{}_DenoiNVMS_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, img_size, M, num_epochs, lr, step_size, gamma, batch_size, reg)

title_mmse_NVMS = model_root / (net_type[net_arch] + suffix_mmse_NVMS)
load_net(title_mmse_NVMS, mmse_NVMS, device)

# -- Load a model with NVMS's layers adapted to noise level test

mmse_NVMS_stock = DenoiCompNetNVMS(img_size, M, Mean, Cov, NVMS=NVMS, variant=net_arch, N0=N0_test, sig=sig_test, H=H, Ord=Ord)
mmse_NVMS_stock = mmse_NVMS_stock.to(device)

# -- Upload layers weights
mmse_NVMS.fcP0.weight = mmse_NVMS_stock.fcP0.weight
mmse_NVMS.fcP1.weight = mmse_NVMS_stock.fcP1.weight
mmse_NVMS.fcP2.weight = mmse_NVMS_stock.fcP2.weight

#############################
# -- Acquisition measurements
#############################
num_img = 4  # [4,19,123]
b = 1
img_test = inputs[num_img, 0, :, :].view([b, c, h, w])
m = mmse_diag.forward_acquire(img_test, b, c, h, w)  # measures with pos/neg coefficients
m, var = mmse_diag.forward_variance(m, b, c, h, w)  # Variance calculus
hadam = mmse_diag.forward_preprocess(m, b, c, h, w)  # hadamard coefficient normalized

#####################
# -- Model evaluation
#####################

# -- Pseudo inverse before FCN
f_pinv = pinv.forward_maptoimage(hadam, b, c, h, w)

# -- mmse without denoising stage before FCN
f_mmse = mmse.forward_maptoimage(hadam, b, c, h, w)

# -- mmse + diag approx before FCN
diag_denoi = mmse_diag.forward_denoise(hadam, var, b, c, h, w)
f_mmse_diag = mmse_diag.forward_maptoimage(diag_denoi, b, c, h, w)

# -- mmse + first order Taylor approximation before FCN
taylor_denoi = mmse_taylor.forward_denoise(hadam, var, b, c, h, w)
f_mmse_taylor = mmse_taylor.forward_maptoimage(taylor_denoi, b, c, h, w)

# -- mmse + full denoi inverse before FCN
full_denoi = mmse_full.forward_denoise(hadam, var, b, c, h, w)
f_mmse_full = mmse_full.forward_maptoimage(full_denoi, b, c, h, w)

# -- mmse + NVMS denoi before FCN
nvms_denoi = mmse_NVMS.forward_denoise(hadam, var, b, c, h, w)
f_mmse_nvms = mmse_NVMS.forward_maptoimage(nvms_denoi, b, c, h, w)

# -- Pseudo inverse + FCN
net_pinv = pinv.forward_postprocess(f_pinv, b, c, h, w)

# -- mmse + FCN
# net_mmse = mmse.forward_postprocess(f_mmse, b, c, h, w)
net_mmse = mmse.forward_postprocess(f_mmse, b, c, h, w)

# -- mmse + diag approx + FCN
net_mmse_denoi = mmse_diag.forward_postprocess(f_mmse_diag, b, c, h, w)

# -- mmse + first order Taylor approximation + FCN
net_mmse_taylor = mmse_taylor.forward_postprocess(f_mmse_taylor, b, c, h, w)

# -- mmse + full denoi + FCN
net_mmse_full = mmse_full.forward_postprocess(f_mmse_full, b, c, h, w)

# -- mmse + NVMS denoi + FCN
net_mmse_nvms = mmse_NVMS.forward_postprocess(f_mmse_nvms, b, c, h, w)

###########################
# -- Displaying the results
###########################
# numpy ground-true : We select an image for visual test
GT = img_test.view([h, w]).cpu().detach().numpy()

fig, axs = plt.subplots(nrows=2, ncols=7, constrained_layout=True)
fig.suptitle('Comparaison des reconstructions en appliquant différents noyaux proposées. '
             'Acquisition effectué avec {} motifs et {} photons. Réseau convolutionel entraîné avec {} photons'.format(M, N0_test, N0), fontsize='large')

ax = axs[0, 0]
im = f_pinv[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('Pinv')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[0, 1]
im = f_mmse[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('MMSE (without denoising)')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[0, 2]
im = f_mmse_taylor[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('MMSE (Taylor)')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[0, 3]
im = f_mmse_diag[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('MMSE (Diagonal)')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[0, 4]
im = f_mmse_nvms[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('MMSE (Taylor-NVMS)')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[0, 5]
im = f_mmse_full[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('MMSE (Inverse complète)')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[0, 6]
ax.imshow(GT, cmap='gray')
ax.set_title('Vérité Terrain')

##############

ax = axs[1, 0]
im = net_pinv[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('Pinv + FCNN')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[1, 1]
im = net_mmse[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('MMSE (without denoising) + FCNN')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[1, 2]
im = net_mmse_taylor[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('MMSE (Taylor) + FCNN')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[1, 3]
im = net_mmse_denoi[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('MMSE (Diagonal) + FCNN')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[1, 4]
im = net_mmse_nvms[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('MMSE (Taylor-NVMS) + FCNN')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[1, 5]
im = net_mmse_full[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('MMSE (Inverse complète) + FCNN')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[1, 6]
ax.imshow(GT, cmap='gray')
ax.set_title('Vérité Terrain')

plt.show()

####################################
# -- PSNR test on the validation set
####################################
psnr_Pinv, psnr_NET_Pinv = dataset_psnr(dataloaders['val'], pinv, device)
print_mean_std(psnr_Pinv, 'Pinv')
print_mean_std(psnr_NET_Pinv, 'Pinv + FCNN')

psnr_mmse, psnr_NET_mmse = dataset_psnr(dataloaders['val'], mmse, device)
print_mean_std(psnr_mmse, 'MMSE without denoising')
print_mean_std(psnr_NET_mmse, 'MMSE + FCNN')

psnr_mmse_taylor_denoi, psnr_NET_mmse_taylor_denoi = dataset_psnr(dataloaders['val'], mmse_taylor, device, denoise=True, M=M)
print_mean_std(psnr_mmse_taylor_denoi, 'MMSE + taylor denoi')
print_mean_std(psnr_NET_mmse_taylor_denoi, 'MMSE + taylor denoi + FCNN')

psnr_mmse_diag_denoi, psnr_NET_mmse_diag_denoi = dataset_psnr(dataloaders['val'], mmse_diag, device, denoise=True, M=M)
print_mean_std(psnr_mmse_diag_denoi, 'MMSE + diag denoi')
print_mean_std(psnr_NET_mmse_diag_denoi, 'MMSE + diag denoi + FCNN')

psnr_mmse_nvms_denoi, psnr_NET_mmse_nvms_denoi = dataset_psnr(dataloaders['val'], mmse_NVMS, device, denoise=True, M=M)
print_mean_std(psnr_mmse_nvms_denoi, 'MMSE + NVMS denoi')
print_mean_std(psnr_NET_mmse_nvms_denoi, 'MMSE + NVMS denoi + FCNN')

psnr_mmse_full_denoi, psnr_NET_mmse_full_denoi = dataset_psnr(dataloaders['val'], mmse_full, device, denoise=True, M=M)
print_mean_std(psnr_mmse_full_denoi, 'MMSE + full denoi')
print_mean_std(psnr_NET_mmse_full_denoi, 'MMSE + full denoi + FCNN')

#################
# -- PSNR boxplot
#################
plt.rcParams.update({'font.size': 8})
plt.figure()
sns.set_style("whitegrid")
axes = sns.boxplot(data=pd.DataFrame([psnr_mmse, psnr_NET_mmse, \
                                      psnr_mmse_taylor_denoi, psnr_NET_mmse_taylor_denoi, \
                                      psnr_mmse_diag_denoi, psnr_NET_mmse_diag_denoi, \
                                      psnr_mmse_nvms_denoi, psnr_NET_mmse_nvms_denoi, \
                                      psnr_mmse_full_denoi, psnr_NET_mmse_full_denoi]).T)

axes.set_xticklabels(['MMSE', 'MMSE + FCNN', \
                      'MMSE + taylor denoi', 'MMSE + taylor denoi + FCNN', \
                      'MMSE + diag denoi', 'MMSE + diag denoi + FCNN', \
                      'MMSE + NVMS denoi', 'MMSE + NVMS denoi + FCNN', \
                      'MMSE + full denoi', 'MMSE + full denoi + FCNN'])
axes.set_ylabel('PSNR')

#######################
# Load training history
#######################
train_path_Pinv = model_root/('TRAIN_pinv'+suffix_Pinv+'.pkl')
train_NET_Pinv = read_param(train_path_Pinv)

train_path_MMSE = model_root / ('TRAIN_c0mp' + suffix_mmse + '.pkl')
train_NET_MMSE = read_param(train_path_MMSE)

train_path_MMSE_diag_denoi = model_root / ('TRAIN_c0mp' + suffix_mmse_diag + '.pkl')
train_NET_MMSE_diag_denoi = read_param(train_path_MMSE_diag_denoi)

train_path_MMSE_taylor_denoi = model_root / ('TRAIN_c0mp' + suffix_mmse_taylor + '.pkl')
train_NET_MMSE_taylor_denoi = read_param(train_path_MMSE_taylor_denoi)

train_path_MMSE_full_denoi = model_root / ('TRAIN_c0mp' + suffix_mmse_full + '.pkl')
train_NET_MMSE_full_denoi = read_param(train_path_MMSE_full_denoi)

train_path_MMSE_nvms_denoi = model_root / ('TRAIN_c0mp' + suffix_mmse_NVMS + '.pkl')
train_NET_MMSE_nvms_denoi = read_param(train_path_MMSE_nvms_denoi)

plt.rcParams.update({'font.size': 12})

##################
# -- Training Plot
##################
fig1, ax = plt.subplots(figsize=(10, 6))
plt.title('Comparison of loss curves for Pseudo inverse, and MMSE models from training with {} photons'.format(N0), fontsize=16)
ax.set_xlabel('Time (epochs)')
ax.set_ylabel('Loss (MSE)')
ax.plot(train_NET_Pinv.val_loss, 'g', linewidth=1.5)
ax.plot(train_NET_MMSE.val_loss, 'y', linewidth=1.5)
ax.plot(train_NET_MMSE_diag_denoi.val_loss, 'b', linewidth=1.5)
ax.plot(train_NET_MMSE_taylor_denoi.val_loss, 'm', linewidth=1.5)
ax.plot(train_NET_MMSE_full_denoi.val_loss, 'r', linewidth=1.5)
ax.plot(train_NET_MMSE_nvms_denoi.val_loss, 'c', linewidth=1.5)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.grid(True)
ax.legend(('Pseudo inverse : 38m 23s', \
          ' MMSE without denoising : 39m 2s', \
          ' MMSE + denoi (diagonal approximation) :  41m 32s',\
          ' MMSE + denoi (Taylor approximation) :  112m 45s',\
          ' MMSE + denoi (full inversion) :  149m 5s',\
          ' MMSE + denoi (Taylor inversion) + NVMS : 38m 41s'),  loc='upper right')












