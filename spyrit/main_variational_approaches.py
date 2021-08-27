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

###########################
# -- Acquisition parameters
###########################
img_size = 64  # image size
batch_size = 256
M = 1024  # number of measurements
N0 = 100  # maximum photons/pixel in training stage
sig = 0.0  # std of maximum photons/pixel

#########################
# -- Model and data paths
#########################
data_root = Path('/home/licho/Documentos/Stage/Codes/STL10/')
stats_root = Path('/home/licho/Documentos/Stage/Codes/Test/')

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

even_index = range(0, 2 * M, 2)
uneven_index = range(1, 2 * M, 2)

####################
# Variational models
####################

#######################################
# model 1 : Regularisation L1 with ISTA
#######################################
DCT = dct(np.eye(img_size ** 2), axis=0)  # Discrete Cosine Transform
Lambda = 1e4  # Regularisation parameter. This parameter also controls the regularity in the noise presence
Niter = 500  # Number of iterations for ISTA algorithm

Reg_L1_ISTA_DCT = RegL1ISTA(img_size, M, Mean, Cov, Basis=DCT, reg=Lambda, Niter=Niter, N0=N0, sig=sig, H=H, Ord=Ord)
Reg_L1_ISTA_DCT = Reg_L1_ISTA_DCT.to(device)

"""
#######################################
# model 2 : TV-L2 Regularisation
#######################################
epsilon = 1e-3 # Gradient regularisation parameter
Lambda = 7e1 # Regularisation parameter
step_size = 1e-6 # Gradient descent step size
Niter = 500 # Number of iterations for gradient descent algorithm

Reg_TV_L2 = RegTVL2GRAD(img_size, M, Mean, Cov, reg=Lambda, eta=step_size, Niter=Niter, eps=epsilon, N0=N0, sig=sig, H=H, Ord=Ord)
Reg_TV_L2 = Reg_TV_L2 .to(device)
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
num_img = 34  # [4,19,123]
b = 1
img_test = inputs[num_img, 0, :, :].view([b, c, h, w])
m = Reg_L1_ISTA_DCT.forward_acquire(img_test, b, c, h, w)  # measures with pos/neg coefficients
hadam = Reg_L1_ISTA_DCT.forward_preprocess(m, b, c, h, w)  # hadamard coefficient normalized

#####################
# -- Model evaluation
#####################

f_Reg_L1 = Reg_L1_ISTA_DCT.forward_maptoimage(hadam, b, c, h, w)
# f_Reg_TV_L2 = Reg_TV_L2.forward_reconstruct(hadam, b, c, h, w)

###########################
# -- Displaying the results
###########################
# numpy ground-true : We select an image for visual test
GT = img_test.view([h, w]).cpu().detach().numpy()

fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
fig.suptitle('Comparaison des reconstructions en appliquant différents methodes proposées. '
             'Acquisition effectué avec {} motifs et {} photons.'.format(M, N0), fontsize='large')

ax = axs[0]
im = f_Reg_L1[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('Regularisation L1 en appliquant la transformé de cosinus')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))

"""
ax = axs[1]
im = f_Reg_TV_L2[0, 0, :, :].cpu().detach().numpy()
ax.imshow(im, cmap='gray')
ax.set_title('Regilarisation TV-L2')
ax.set_xlabel('PSNR =%.3f' % psnr_(GT, im))
"""

ax = axs[1]
ax.imshow(GT, cmap='gray')
ax.set_title('Vérité Terrain')

plt.show()