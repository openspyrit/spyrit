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
N0_test = 50
sig = 0.0  # std of maximum photons/pixel

#########################
# -- Model and data paths
#########################
data_root = Path('/home/licho/Documentos/Stage/Codes/STL10/')
stats_root = Path('/home/licho/Documentos/Stage/Codes/Test')
model_root = Path('/home/licho/Documentos/Stage/Codes/Semaine20/Iterative_training/')

My_NVMS_file = Path(stats_root) / (
    'NVMS_N_{}_M_{}.npy'.format(img_size, M))
NVMS = np.load(My_NVMS_file) / N0_test
print('loaded :NVMS_N_{}_M_{}.npy'.format(img_size, M))

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
net_arch = 0
# Number of gradient descent iterations
Niter = 5
# Number of training epochs
num_epochs_1 = 30
# Number of training epochs
num_epochs_2 = 15
# Regularisation Parameter
reg = 1e-7
# Learning Rate
lr = 1e-3
# Scheduler Step Size
step_size = 10
# Scheduler Decrease Rate
gamma = 0.5

#################################################################
# model 0 : MMSE + Denoising stage with diagonal matrix inversion
#################################################################
mmse_diag_stock = DenoiCompNet(img_size, M, Mean, Cov, variant=net_arch, N0=N0_test, sig=sig, H=H, Ord=Ord)
mmse_diag_stock = mmse_diag_stock.to(device)

# -- Load net
suffix_mmse_diag_stock = '_N0_{}_sig_{}_Denoi_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, img_size, M, num_epochs_1, lr, step_size, gamma, batch_size, reg)

title_mmse_diag_stock = model_root / (net_type[net_arch] + suffix_mmse_diag_stock)
load_net(title_mmse_diag_stock, mmse_diag_stock, device)

#################################################################
# model 1 : MMSE + Denoising stage with diagonal matrix inversion
#################################################################
mmse_diag = DenoiCompNetIter(img_size, M, Mean, Cov, Niter=Niter, variant=net_arch, N0=N0_test, sig=sig, H=H, Ord=Ord)
mmse_diag = mmse_diag.to(device)

# -- Load net
suffix_mmse_diag = '_N0_{}_sig_{}_IterDenoi_Niter_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, Niter, img_size, M, num_epochs_2, lr, step_size, gamma, batch_size, reg)

title_mmse_diag = model_root / (net_type[net_arch] + suffix_mmse_diag)
load_net(title_mmse_diag, mmse_diag, device)


##########################################################
# model 2 : MMSE + Denoising stage with NVMS approximation
##########################################################
mmse_NVMS = DenoiCompNetIterNVMS(img_size, M, Mean, Cov, NVMS, Niter=Niter, variant=net_arch, N0=N0_test, sig=sig, H=H, Ord=Ord)
mmse_NVMS = mmse_NVMS.to(device)

# -- Load net
suffix_mmse_NVMS = '_N0_{}_sig_{}_IterDenoiNVMS_Niter_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, Niter, img_size, M, num_epochs_2, lr, step_size, gamma, batch_size, reg)

title_mmse_NVMS = model_root / (net_type[net_arch] + suffix_mmse_NVMS)
load_net(title_mmse_NVMS, mmse_NVMS, device)

# -- Load a model with NVMS's layers adapted to noise level test
if N0 != N0_test:
    mmse_NVMS_stock = DenoiCompNetNVMS(img_size, M, Mean, Cov, NVMS=NVMS, variant=net_arch, N0=N0_test, sig=sig,
                                       H=H, Ord=Ord)
    mmse_NVMS_stock = mmse_NVMS_stock.to(device)

    # -- Upload layers weights
    mmse_NVMS.fcP0.weight = mmse_NVMS_stock.fcP0.weight
    mmse_NVMS.fcP1.weight = mmse_NVMS_stock.fcP1.weight
    mmse_NVMS.fcP2.weight = mmse_NVMS_stock.fcP2.weight

"""
####################
# Variational models
####################

########################################################
# model 3 : TV-L2 Regularisation with gradient conjugate
########################################################
epsilon = 1e-3  # Gradient regularisation parameter
Lambda = 1e2  # Regularisation parameter : 1e2

Reg_TV_L2_Conj = RegTVL2GRAD(img_size, M, Mean, Cov, NVMS=NVMS, reg=Lambda, step_size=tau, epsilon=epsilon, Niter=Niter, N0=N0, sig=sig, H=H, Ord=Ord)
Reg_TV_L2_Conj = Reg_TV_L2_Conj.to(device)

# -- Load net
suffix_mmse_NVMS_TV = '_N0_{}_sig_{}_DenoiIterNVMS_Niter_{}_tau_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(\
    N0, sig, Niter, tau, img_size, M, num_epochs, lr, step_size, gamma, batch_size, reg)

title_mmse_NVMS_TV = model_root / (net_type[net_arch] + suffix_mmse_NVMS_TV)
load_net(title_mmse_NVMS_TV, Reg_TV_L2_Conj, device)

# -- Load a model with NVMS's layers adapted to noise level test
if N0 != N0_test:
    mmse_NVMS_stock = DenoiCompNetNVMS(img_size, M, Mean, Cov, NVMS=NVMS, variant=net_arch, N0=N0_test, sig=sig,
                                       H=H, Ord=Ord)
    mmse_NVMS_stock = mmse_NVMS_stock.to(device)

    # -- Upload layers weights
    Reg_TV_L2_Conj.fcP0.weight = mmse_NVMS_stock.fcP0.weight
    Reg_TV_L2_Conj.fcP1.weight = mmse_NVMS_stock.fcP1.weight
    Reg_TV_L2_Conj.fcP2.weight = mmse_NVMS_stock.fcP2.weight

"""

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
m = mmse_diag.forward_acquire(img_test, b, c, h, w)  # measures with pos/neg coefficients

#####################
# -- Model evaluation
#####################
f_mmse_diag_stock = mmse_diag_stock.forward_reconstruct(m, b, c, h, w)
f_mmse_diag = mmse_diag.forward_reconstruct(m, b, c, h, w)
f_mmse_NVMS = mmse_NVMS.forward_reconstruct(m, b, c, h, w)

###########################
# -- Displaying the results
###########################
# numpy ground-true : We select an image for visual test
GT = img_test.view([h, w]).cpu().detach().numpy()

fig, axs = plt.subplots(nrows=1, ncols=4, constrained_layout=True)
fig.suptitle('Comparaison des reconstructions en appliquant différents methodes proposées. '
             'Acquisition effectué avec {} motifs et {} photons.'.format(M, N0_test), fontsize='large')

#################################
ax = axs[0]
im = f_mmse_diag_stock[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('mmse + diag denoi')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[1]
im = f_mmse_diag[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('mmse + diag denoi Iterative')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[2]
im = f_mmse_NVMS[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('mmse + NVMS Iterative')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

ax = axs[3]
ax.imshow(GT, cmap='gray')
ax.set_title('Vérité Terrain')

plt.tight_layout()
plt.show()

#######################
# Load training history
#######################
train_path_mmse_diag_stock = model_root / ('TRAIN_c0mp' + suffix_mmse_diag_stock + '.pkl')
train_net_mmse_diag_stock = read_param(train_path_mmse_diag_stock)

train_path_mmse_diag = model_root / ('TRAIN_c0mp' + suffix_mmse_diag + '.pkl')
train_net_mmse_diag = read_param(train_path_mmse_diag)

train_path_mmse_NVMS = model_root / ('TRAIN_c0mp' + suffix_mmse_NVMS + '.pkl')
train_net_mmse_NVMS = read_param(train_path_mmse_NVMS)

plt.rcParams.update({'font.size': 12})

##################
# -- Training Plot
##################
fig1, ax = plt.subplots(figsize=(10, 6))
plt.title('Comparison of loss curves for MMSE models from training with {} photons'.format(N0), fontsize=16)
ax.set_xlabel('Time (epochs)')
ax.set_ylabel('Loss (MSE)')
ax.plot(train_net_mmse_diag_stock.val_loss, 'g', linewidth=1.5)
ax.plot(train_net_mmse_diag.val_loss, 'b', linewidth=1.5)
ax.plot(train_net_mmse_NVMS.val_loss, 'r', linewidth=1.5)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.grid(True)
ax.legend((' MMSE + denoi (diagonal approximation). Training complete in 41m 32s, Best val Loss: 2.29',\
           ' MMSE + denoi (diagonal approximation) Iter. Training complete in 42m 53s, Best val Loss: 2.19',\
           ' MMSE + denoi (Taylor inversion) + NVMS Iter. Training complete in 42m 59s, Best val Loss: 2.19'),  loc='upper right')