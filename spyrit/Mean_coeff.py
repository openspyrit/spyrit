###########
# --Imports
###########

import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('../..')

# -- Pytorch tools
import torch
import torch.nn as nn

# -- Dataloading tools
import torchvision
from torchvision import datasets, models, transforms

# -- Spyrit packages
from spyrit.learning.model_Had_DCAN import * # models
from spyrit.learning.nets import * # traning, load, visualization...

#########################################
# -- STL-10 (Loading the Compressed Data)
#########################################
# Loading and normalizing STL10 :
# The output of torchvision datasets are PILImage images of range [0, 1].
# RGB images transformed into grayscale images.

print("Loading STL-10 DATA")

img_size = 64  # Height-width dimension of the unknown image

data_root = Path('/home/licho/Documentos/Stage/Codes/STL10/')
transform = transforms.Compose(
    [transforms.functional.to_grayscale,
     transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = datasets.STL10(root=data_root,
                          split='train+unlabeled',
                          transform=transform)

# -- Sample size
batch_size = 256

dataloader = \
    torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

print('Dataloaders are ready')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#####################################################
# -- Precomputed data (Average and covariance matrix)
#####################################################
# -- Path to precomputed data (Average and covariance matrix -- for model)
precompute_root = Path('/home/licho/Documentos/Stage/Codes/Test/')
Cov = np.load(precompute_root / "Cov_{}x{}.npy".format(img_size, img_size))

###########################
# -- Acquisition parameters
###########################
print('Loading Acquisition parameters ( Covariance and Mean matrices) ')
# -- Compressed Reconstruction via CNN (CR = 1/8)
CR = 1024  # Number of patterns ---> M = 1024 measures

# -- This is the same acquire function of the Learning model
def theoretical_acquisition(x, M, b, c, h, w, P):
    # --Scale input image
    x = (x + 1) / 2
    ###############
    # --Acquisition
    ###############
    x = x.view(b * c, 1, h, w)
    Px = P.to(x.device)
    x = Px(x)
    x = F.relu(x)
    x = x.view(b * c, 1, 2 * M)

    return x

# -- Hadamard Matrix definition (fht hadamard transform needs to be normalized)
H = img_size *  Hadamard_Transform_Matrix(img_size)
# -- Selection of Hadamard coefficients
Var = Cov2Var(Cov)
Perm = Permutation_Matrix(Var)
Pmat = np.dot(Perm, H)
Pmat = Pmat[:CR, :]
Pconv = matrix2conv(Pmat)
# -- Pattern matrix statement
P, T = split(Pconv, 1)
P.bias.requires_grad = False
P.weight.requires_grad = False

def sum_coeff(dat, n, M, bs, Pattern, dev):
    # -- Index of measurements
    even_index = range(0, 2 * M, 2);
    uneven_index = range(1, 2 * M, 2);

    L = dat.dataset.data.shape[0] # -- We takes M measures for any database image
    N = np.int(L / bs)
    S = np.zeros((L,1)) # -- Vector for sum storage
    j = 0 # -- Batch counter
    for inputs, labels in dat:
        inputs = inputs.to(dev)
        # --We takes b * c theoretical measures on STL-10 database
        b, c, h, w = inputs.shape
        x = theoretical_acquisition(inputs, M, b, c, h, w, Pattern)

        # -- Measurement recombination
        x = x[:, :, even_index] + x[:, :, uneven_index]

        # -- Sum of Hadarmard coefficients
        if j < N:
            S[j * bs: (j + 1) * bs] = torch.mean(x,2)

        else:
            S[j * bs:] = torch.mean(x, 2)

        j = j + 1 # -- Update of batch counter

        print(f'batch : {j}')

    return S


cumsum_coef = sum_coeff(dat=dataloader, n=img_size, M=CR, bs=batch_size, Pattern=P, dev=device)
# Gaussian KDE
kde = KernelDensity(kernel='gaussian', bandwidth=100).fit(cumsum_coef)

plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots()
X_plot = np.linspace(np.min(cumsum_coef), np.max(cumsum_coef), 50)[:, np.newaxis]
bins = np.linspace(np.min(cumsum_coef), np.max(cumsum_coef), 50)
log_dens = kde.score_samples(X_plot)
ax.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax.text(-3.5, 0.31, "Gaussian Kernel Density")
plt.xlabel('Intensité total par image')
plt.ylabel('Densité de probabilité')
