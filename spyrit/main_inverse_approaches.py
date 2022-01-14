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
CR = 1024  # Number of Hadamard coefficients in the acquisition step in training stage 25%
N0 = 50 # Maximum photons/pixel level in training stage
N0_test = 5  # Noise test level
sig = 0.0  # std of maximum photons/pixel

#########################
# -- Model and data paths
#########################
data_root = Path('/home/licho/Documentos/Stage/Codes/STL10/')
stats_root = Path('/home/licho/Documentos/Stage/Codes/Test/')
model_root = Path('/home/licho/Documentos/Stage/Article/training_models-main/')

# Calculate the Noise Variance Matrix Stabilization
NVMS = np.diag((img_size ** 2) * np.ones(CR)) / N0_test

################################################################
# -- Precalculated data, Hadamard matrix, and coefficients order
################################################################
Cov = np.load(stats_root / Path("Cov_{}x{}.npy".format(img_size, img_size)))
Mean = np.load(stats_root / Path("Average_{}x{}.npy".format(img_size, img_size)))
H = walsh2_matrix(img_size)/img_size
Cov = Cov / img_size ** 2
Ord = Cov2Var(Cov)

########################
# -- STL10 database load
########################
print("Loading STL-10 DATA")
device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

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
net_arch = 0  # Bayesian solution
num_epochs = 30  # Number of training epochs
num_epochs_comp = 6 # Number of training epochs in the iterative case
reg = 1e-7  # Regularisation Parameter
lr = 1e-3  # Learning Rate
step_size = 10  # Scheduler Step Size
gamma = 0.5  # Scheduler Decrease Rate
Niter_simple = 1  # Number of net iterations for simple schema
Niter_comp = 5  # Number of net iterations for compound schema

########################
# -- Loading MMSE models
########################

###############################################################################
# model 1 : Denoising stage with full matrix inversion -- vanilla version (k=0)
###############################################################################

denoiCompNetFull = DenoiCompNet(img_size, CR, Mean, Cov, NVMS, Niter=Niter_simple, variant=net_arch, denoi=2, N0=N0_test, sig=sig, H=H, Ord=Ord)
denoiCompNetFull = denoiCompNetFull.to(device)

# -- Load net
suffix0 = '_N0_{}_sig_{}_Denoi_Full_Niter_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, Niter_simple, img_size, CR, num_epochs, lr, step_size, gamma, batch_size, reg)

title0 = model_root / (net_type[net_arch] + suffix0)
load_net(title0, denoiCompNetFull, device)

####################################################################
# model 2 : Denoising stage with diagonal matrix approximation (k=0)
####################################################################

denoiCompNet_simple = DenoiCompNet(img_size, CR, Mean, Cov, NVMS, Niter=Niter_simple, variant=net_arch, denoi=1, N0=N0_test, sig=sig, H=H, Ord=Ord)
denoiCompNet_simple = denoiCompNet_simple.to(device)

# -- Load net
suffix1 = '_N0_{}_sig_{}_Denoi_Diag_Niter_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, Niter_simple, img_size, CR, num_epochs, lr, step_size, gamma, batch_size, reg)

title1 = model_root / (net_type[net_arch] + suffix1)
load_net(title1, denoiCompNet_simple, device)

####################################################################
# model 3 : Denoising stage with diagonal matrix approximation (k=5)
####################################################################

denoiCompNet_iter = DenoiCompNet(img_size, CR, Mean, Cov, NVMS, Niter=Niter_comp, variant=net_arch, denoi=1, N0=N0_test, sig=sig, H=H, Ord=Ord)
denoiCompNet_iter = denoiCompNet_iter.to(device)

# -- Load net
suffix2 = '_N0_{}_sig_{}_Denoi_Diag_Niter_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, Niter_comp, img_size, CR, num_epochs_comp, lr, step_size, gamma, batch_size, reg)

title2 = model_root / (net_type[net_arch] + suffix2)
load_net(title2, denoiCompNet_iter, device)

########################################################################################################
# model 4 : Denoising stage with a first order taylor approximation + NVMS (with max matrix) and k=0.  #
########################################################################################################

denoiCompNetNVMS_simple = DenoiCompNet(img_size, CR, Mean, Cov, NVMS=NVMS, Niter=Niter_simple, variant=net_arch, denoi=0, N0=N0_test, sig=sig, H=H, Ord=Ord)
denoiCompNetNVMS_simple = denoiCompNetNVMS_simple.to(device)

# -- Load net
suffix3 = '_N0_{}_sig_{}_Denoi_NVMS_Max_Niter_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, Niter_simple, img_size, CR, num_epochs, lr, step_size, gamma, batch_size, reg)

title3 = model_root / (net_type[net_arch] + suffix3)
load_net(title3, denoiCompNetNVMS_simple, device)

#######################################################################################################
# model 5 : Denoising stage with a first order taylor approximation + NVMS (with max matrix) and k=5 #
#######################################################################################################

denoiCompNetNVMS_iter = DenoiCompNet(img_size, CR, Mean, Cov, NVMS=NVMS, Niter=Niter_comp, variant=net_arch, denoi=0, N0=N0_test, sig=sig, H=H, Ord=Ord)
denoiCompNetNVMS_iter = denoiCompNetNVMS_iter.to(device)

# -- Load net
suffix4 = '_N0_{}_sig_{}_Denoi_NVMS_Max_Niter_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, Niter_comp, img_size, CR, num_epochs_comp, lr, step_size, gamma, batch_size, reg)

title4 = model_root / (net_type[net_arch] + suffix4)
load_net(title4, denoiCompNetNVMS_iter, device)

##########################
# -- Upload layers weights
##########################

if N0 != N0_test:
    P0, P1, P2 = denoiCompNetNVMS_simple.forward_denoise_operators(Cov, NVMS, img_size, CR)
    denoiCompNetNVMS_simple.fcP0 = P0
    denoiCompNetNVMS_simple.fcP1 = P1
    denoiCompNetNVMS_simple.fcP2 = P2

    denoiCompNetNVMS_iter.fcP0 = P0
    denoiCompNetNVMS_iter.fcP1 = P1
    denoiCompNetNVMS_iter.fcP2 = P2

#############################
# -- Acquisition measurements
#############################

# -- Image test selection
num_img = 34 # 70, 4, 115, 116, 117, 34, 64 (low frequencies and 2 ph: 9, 19)
b = 1
img_test = inputs[num_img, 0, :, :].view([b, c, h, w])
m = denoiCompNet_simple.forward_acquire(img_test, b, c, h, w)  # measures with pos/neg coefficients
x, var = denoiCompNet_simple.forward_variance(m, b, c, h, w) # Variance calculus
x = denoiCompNet_simple.forward_preprocess(x, b, c, h, w)  # Hadamard coefficient normalized

##########################
# -- Firs layer evaluation
##########################

x_diag = denoiCompNet_simple.forward_denoise(x, var, b, c, h, w)
f_diag = denoiCompNet_simple.forward_maptoimage(x_diag, b, c, h, w)

x_nvms = denoiCompNetNVMS_simple.forward_denoise(x, var, b, c, h, w)
f_nvms = denoiCompNetNVMS_simple.forward_maptoimage(x_nvms, b, c, h, w)

x_full = denoiCompNetFull.forward_denoise(x, var, b, c, h, w)
f_full = denoiCompNetFull.forward_maptoimage(x_full, b, c, h, w)

####################
# -- FCNN evaluation
####################

# -- mmse + diag approx + FCNN
net_denoi = denoiCompNet_simple.forward_postprocess(f_diag, b, c, h, w)

# -- mmse + NVMS denoi (Max) + FCNN
net_nvms = denoiCompNetNVMS_simple.forward_postprocess(f_nvms, b, c, h, w)

# -- mmse + full denoi + FCNN
net_full = denoiCompNetFull.forward_postprocess(f_full, b, c, h, w)

# -- mmse + diag approx + FCNN (k=5)
net_diag_iter = denoiCompNet_iter.forward_reconstruct(m, b, c, h, w)

# -- mmse + NVMS denoi (max) + FCNN (k=5)
net_nvms_iter = denoiCompNetNVMS_iter.forward_reconstruct(m, b, c, h, w)

###########################
# -- Displaying the results
###########################
# numpy ground-true : We select an image for visual test
GT = img_test.view([h, w]).cpu().detach().numpy()

fig, axs = plt.subplots(nrows=3, ncols=3, constrained_layout=True)
fig.suptitle('Test with {} patterns and {} photons. Training with {} patterns and {} photons '.format(CR, N0_test, CR, N0), fontsize=18)

ax = axs[0, 0]
im = f_diag[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=12)

ax = axs[0, 1]
im = f_nvms[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=12)

ax = axs[0, 2]
im = f_full[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=12)

##############

ax = axs[1, 0]
im = net_denoi[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=12)

ax = axs[1, 1]
im = net_nvms[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=12)

ax = axs[1, 2]
im = net_full[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=12)

##############

ax = axs[2, 0]
im = net_diag_iter[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=12)

ax = axs[2, 1]
im = net_nvms_iter[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=12)

ax = axs[2, 2]
ax.imshow(GT, cmap='gray')
ax.set_title('Ground Truth', fontsize=18)

plt.figtext(0.12,0.91, "Diagonal approximation", ha="left", va="top", fontsize=16, color="k")
plt.figtext(0.5,0.91, "Taylor approximation (NVMS)", ha="center", va="top", fontsize=16, color="k")
plt.figtext(0.85,0.91, "Full inverse", ha="right", va="top", fontsize=16, color="k")
plt.figtext(0.5,0.94, "First layer evaluation.", ha="center", va="top", fontsize=14, color="k")
plt.figtext(0.5,0.645, "Fully convolutional neural network  (1 iterations)", ha="center", va="top", fontsize=14, color="k")
plt.figtext(0.5,0.33, "Fully convolutional neural network  (5 iterations)", ha="center", va="top", fontsize=14, color="k")
plt.tight_layout()
plt.show()


################
# Paper images #
################

# -- First Layer Evaluation

fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True)

ax = axs[0]
im = f_diag[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('Diag', fontsize=18)
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=16)

ax = axs[1]
im = f_nvms[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('Proposed', fontsize=18)
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=16)

ax = axs[2]
im = f_full[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('Full', fontsize=18)
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=16)

"""
ax = axs[3]
ax.imshow(GT, cmap='gray')
ax.set_title('Ground Truth', fontsize=18)
"""

plt.show()

# ----- Net results (K=1)

# fig.suptitle('Comparaison des reconstructions en appliquant différents noyaux proposées. '
#             'Acquisition effectué avec {} motifs et {} photons. Réseau convolutionel entraîné avec {} photons'.format(M, N0_test, N0), fontsize='large')

fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True)

ax = axs[0]
im = net_denoi[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('Diag', fontsize=18)
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=16)

ax = axs[1]
im = net_nvms[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('Proposed', fontsize=18)
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=16)

"""
ax = axs[2]
im = net_full[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('Full', fontsize=18)
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im), fontsize=16)
"""

ax = axs[2]
ax.imshow(GT, cmap='gray')
ax.set_title('Ground Truth', fontsize=18)

plt.show()

####################################
# -- PSNR test on the validation set
####################################
psnr_diag_denoi, psnr_net_diag_denoi, ssim_diag_denoi, ssim_net_diag_denoi = dataset_psnr_ssim_2(dataloaders['val'], denoiCompNet_simple, device, M=CR_test)
print_mean_std(psnr_diag_denoi, 'PSNR: diag denoi')
print_mean_std(psnr_net_diag_denoi, 'PSNR: diag denoi + FCNN')
print_mean_std(ssim_diag_denoi, 'SSIM: diag denoi')
print_mean_std(ssim_net_diag_denoi, 'SSIM: diag denoi + FCNN')

psnr_nvms_denoi, psnr_net_nvms_denoi, ssim_nvms_denoi, ssim_net_nvms_denoi = dataset_psnr_ssim_2(dataloaders['val'], denoiCompNetNVMS_simple, device, M=CR_test)
print_mean_std(psnr_nvms_denoi, 'PSNR: NVMS Max denoi')
print_mean_std(psnr_net_nvms_denoi, 'PSNR: NVMS Max denoi + FCNN')
print_mean_std(ssim_nvms_denoi, 'SSIM: NVMS Max denoi')
print_mean_std(ssim_net_nvms_denoi, 'SSIM: NVMS Max denoi + FCNN')

psnr_full_denoi, psnr_net_full_denoi, ssim_full_denoi, ssim_net_full_denoi = dataset_psnr_ssim_2(dataloaders['val'], denoiCompNetFull, device, M=M)
print_mean_std(psnr_full_denoi, 'PSNR: Full denoi')
print_mean_std(psnr_net_full_denoi, 'PSNR: MMSE + full denoi + FCNN')
print_mean_std(ssim_full_denoi, 'SSIM: Full denoi')
print_mean_std(ssim_net_full_denoi, 'SSIM: Full denoi + FCNN')

#############
# -- K != 1 #
#############

psnr_diag_denoi_iter, ssim_diag_denoi_iter = dataset_psnr_ssim_iterative(dataloaders['val'], denoiCompNet_iter, device, M=CR)
print_mean_std(psnr_diag_denoi_iter, 'PSNR: diag denoi iter')
print_mean_std(ssim_diag_denoi_iter, 'SSIM: diag denoi iter')

psnr_nvms_denoi, ssim_nvms_denoi = dataset_psnr_ssim_iterative(dataloaders['val'], denoiCompNetNVMS_iter, device, M=CR)
print_mean_std(psnr_nvms_denoi, 'PSNR: NVMS Max denoi iter')
print_mean_std(ssim_nvms_denoi, 'SSIM: NVMS Max denoi iter')

psnr_full_denoi_iter, ssim_full_denoi_iter = dataset_psnr_ssim_iterative(dataloaders['val'], denoiCompNetFull, device, M=CR)
print_mean_std(psnr_full_denoi_iter, 'PSNR: full denoi + FCNN')
print_mean_std(ssim_full_denoi_iter, 'SSIM: full denoi + FCNN')

#######################
# Load training history
#######################
train_path_MMSE_diag_denoi = model_root / ('TRAIN_c0mp' + suffix_ + '.pkl')
train_NET_MMSE_diag_denoi = read_param(train_path_MMSE_diag_denoi)

train_path_MMSE_full_denoi = model_root / ('TRAIN_c0mp' + suffix_mmse_full + '.pkl')
train_NET_MMSE_full_denoi = read_param(train_path_MMSE_full_denoi)

train_path_MMSE_nvms_denoi = model_root / ('TRAIN_c0mp' + suffix_mmse_NVMS + '.pkl')
train_NET_MMSE_nvms_denoi = read_param(train_path_MMSE_nvms_denoi)

plt.rcParams.update({'font.size': 12})

##################
# -- Training Plot
##################
fig1, ax = plt.subplots(figsize=(10, 6))
#plt.title('Comparison of loss curves for Pseudo inverse, and MMSE models from training with {} photons'.format(N0), fontsize=16)
ax.set_xlabel('Time (epochs)', fontsize=18)
ax.set_ylabel('Loss (MSE)', fontsize=18)
ax.plot(train_NET_MMSE_diag_denoi.val_loss, 'b', linewidth=1.5)
ax.plot(train_NET_MMSE_nvms_denoi.val_loss, 'c', linewidth=1.5)
ax.plot(train_NET_MMSE_full_denoi.val_loss, 'r', linewidth=1.5)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.grid(True)
ax.legend((' Approx Diagonal. Entraînement en : 38m 14s', \
           ' Approx Taylor (NVMS). Entraînement en : 38m 26s',\
           ' Inverse Complète.  Entraînement en :  147m 53s'),  loc='upper right', fontsize=18)