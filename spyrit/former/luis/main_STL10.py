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
from spyrit.learning.model_Had_DCAN_1 import *  # models
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
sig = 0.0  # std of maximum photons/pixel

#########################
# -- Model and data paths
#########################
data_root = Path('/media/licho/0A286AF6286AE065/Users/Luis/Documents/Deep Learning/STL10/')
stats_root = Path('/media/licho/0A286AF6286AE065/Users/Luis/Documents/Deep Learning/Test/Stats_Walsh/')
model_root = Path('/media/licho/0A286AF6286AE065/Users/Luis/Documents/Deep Learning/training_models-main/')

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
b = 1
# -- Image test selection
num_img = 19 # 97, 70, 4, 115, 116, 117, 34, 64 (low frequencies and 2 ph: 9, 19) (high frequencies : 217, 21, 76, 177, 48, 245)
img_test = inputs[num_img, 0, :, :].view([b, c, h, w])
# numpy ground-true : We select an image for visual test
GT = img_test.view([h, w]).cpu().detach().numpy()

#####################
# -- Models statement
#####################
# --network architecture --> ['c0mp', 'comp','pinv', 'free'] --> [0, 1, 2, 3]
net_type = ['NET_c0mp', 'NET_comp', 'NET_pinv', 'NET_free']
net_arch = 0  # Bayesian solution
num_epochs = 30  # Number of training epochs
reg = 1e-7  # Regularisation Parameter
lr = 1e-3  # Learning Rate
step_size = 10  # Scheduler Step Size
gamma = 0.5  # Scheduler Decrease Rate
Niter_simple = 1  # Number of net iterations for simple schema

########
# Test #
########

f_diag = []
f_nvms = []
f_full = []

# Noise test level
N0_test = [100, 10, 5, 1]

for i in range(len(N0_test)):
    # Calculate the Noise Variance Matrix Stabilization
    NVMS = np.diag((img_size ** 2) * np.ones(CR)) / N0_test[i]

    ########################
    # -- Loading MMSE models
    ########################

    ###############################################################################
    # model 1 : Denoising stage with full matrix inversion -- vanilla version (k=0)
    ###############################################################################

    denoiCompNetFull = DenoiCompNet_NVMS(img_size, CR, Mean, Cov, NVMS, Niter=Niter_simple, variant=net_arch, denoi=2,
                                         N0=N0_test[i], sig=sig, H=H, Ord=Ord)
    denoiCompNetFull = denoiCompNetFull.to(device)

    # -- Load net
    suffix0 = '_N0_{}_sig_{}_Denoi_Full_Niter_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format( \
        N0, sig, Niter_simple, img_size, CR, num_epochs, lr, step_size, gamma, batch_size, reg)

    title0 = model_root / (net_type[net_arch] + suffix0)
    load_net(title0, denoiCompNetFull, device)

    ####################################################################
    # model 2 : Denoising stage with diagonal matrix approximation (k=0)
    ####################################################################

    denoiCompNet_simple = DenoiCompNet_NVMS(img_size, CR, Mean, Cov, NVMS, Niter=Niter_simple, variant=net_arch,
                                            denoi=1, N0=N0_test[i], sig=sig, H=H, Ord=Ord)
    denoiCompNet_simple = denoiCompNet_simple.to(device)

    # -- Load net
    suffix1 = '_N0_{}_sig_{}_Denoi_Diag_Niter_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format( \
        N0, sig, Niter_simple, img_size, CR, num_epochs, lr, step_size, gamma, batch_size, reg)

    title1 = model_root / (net_type[net_arch] + suffix1)
    load_net(title1, denoiCompNet_simple, device)

    ########################################################################################################
    # model 3 : Denoising stage with a first order taylor approximation + NVMS (with max matrix) and k=0.  #
    ########################################################################################################

    denoiCompNetNVMS_simple = DenoiCompNet_NVMS(img_size, CR, Mean, Cov, NVMS=NVMS, Niter=Niter_simple,
                                                variant=net_arch, denoi=0, N0=N0_test[i], sig=sig, H=H, Ord=Ord)
    denoiCompNetNVMS_simple = denoiCompNetNVMS_simple.to(device)

    # -- Load net
    suffix3 = '_N0_{}_sig_{}_Denoi_NVMS_Niter_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format( \
        N0, sig, Niter_simple, img_size, CR, num_epochs, lr, step_size, gamma, batch_size, reg)

    title3 = model_root / (net_type[net_arch] + suffix3)
    load_net(title3, denoiCompNetNVMS_simple, device)

    ##########################
    # -- Upload layers weights
    ##########################

    if N0 != N0_test[i]:
        P0, P1, P2 = denoiCompNetNVMS_simple.forward_denoise_operators(Cov, NVMS, img_size, CR)
        denoiCompNetNVMS_simple.fcP0 = P0
        denoiCompNetNVMS_simple.fcP1 = P1
        denoiCompNetNVMS_simple.fcP2 = P2

    #############################
    # -- Acquisition measurements
    #############################

    m = denoiCompNet_simple.forward_acquire(img_test, b, c, h, w)  # measures with pos/neg coefficients
    x, var = denoiCompNet_simple.forward_variance(m, b, c, h, w)  # Variance
    x = denoiCompNet_simple.forward_preprocess(x, b, c, h, w)  # Hadamard coefficient normalized

    ##########################
    # -- Firs layer evaluation
    ##########################

    x_diag = denoiCompNet_simple.forward_denoise(x, var, b, c, h, w)
    f_diag.append(denoiCompNet_simple.forward_maptoimage(x_diag, b, c, h, w))

    x_nvms = denoiCompNetNVMS_simple.forward_denoise(x, var, b, c, h, w)
    f_nvms.append(denoiCompNetNVMS_simple.forward_maptoimage(x_nvms, b, c, h, w))

    x_full = denoiCompNetFull.forward_denoise(x, var, b, c, h, w)
    f_full.append(denoiCompNetFull.forward_maptoimage(x_full, b, c, h, w))

    if i == len(N0_test) - 1:
        ####################
        # -- FCNN evaluation
        ####################

        # -- mmse + diag approx + FCNN
        f_diag.append(denoiCompNet_simple.forward_postprocess(f_diag[i], b, c, h, w))

        # -- mmse + NVMS denoi (Max) + FCNN
        f_nvms.append(denoiCompNetNVMS_simple.forward_postprocess(f_nvms[i], b, c, h, w))

        # -- mmse + full denoi + FCNN
        f_full.append(denoiCompNetFull.forward_postprocess(f_full[i], b, c, h, w))


###########################
# -- Displaying the results
###########################

model_label = ['Diag approximation', 'Taylor approximation', 'Bayesian solution']
fig, axs = plt.subplots(len(model_label), len(N0_test) + 2, figsize=(10, 6), constrained_layout=False)

# - Ground Truth
for j in range(3):
    axs[j, len(N0_test) + 1].imshow(GT, cmap='gray')

    # Remove axis of GT
    axs[j, len(N0_test) + 1].axis('off')

    # -- row label
    axs[j, 0].set_ylabel(model_label[j], fontsize=16)

    # -- Column label
    if j == 0:
        axs[j, len(N0_test) + 1].set_title('Ground Truth', fontsize=16)

# - Models results
for i in range(0, len(f_full)):
    # -- Columns labels (Noise level)
    if i < 4:
        axs[0, i].set_title("ph = %.0f" % N0_test[i], fontsize=16)
    else:
        axs[0, i].set_title("FCNN at ph = %.0f" % N0_test[i - 1], fontsize=16)

    # -- Diagonal approximation
    im1 = f_diag[i][0, 0, :, :].cpu().detach().numpy()
    axs[0, i].imshow(im1, cmap='gray')
    # Turn off tick labels
    axs[0, i].set_yticklabels([])
    axs[0, i].set_xticklabels([])
    axs[0, i].set_xlabel("PSNR = %.2f" % psnr_(GT, im1), fontsize=14)

    # -- Taylor approximation
    im2 = f_nvms[i][0, 0, :, :].cpu().detach().numpy()
    axs[1, i].imshow(im2, cmap='gray')
    # Turn off tick labels
    axs[1, i].set_yticklabels([])
    axs[1, i].set_xticklabels([])
    axs[1, i].set_xlabel("PSNR = %.2f" % psnr_(GT, im2), fontsize=14)

    # -- Bayesian solution
    im3 = f_full[i][0, 0, :, :].cpu().detach().numpy()
    axs[2, i].imshow(im3, cmap='gray')
    # Turn off tick labels
    axs[2, i].set_yticklabels([])
    axs[2, i].set_xticklabels([])
    axs[2, i].set_xlabel("PSNR = %.2f" % psnr_(GT, im3), fontsize=14)

plt.show()



