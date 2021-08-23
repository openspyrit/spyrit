###########
# --Imports
###########

import numpy as np
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
from spyrit.learning.model_Had_DCAN import *  # models
from spyrit.learning.nets import *  # training, load, visualization...

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#####################################################
# -- Precomputed data (Average and covariance matrix)
#####################################################
# -- Path to precomputed data (Average and covariance matrix -- for model)
precompute_root = Path('/home/licho/Documentos/Stage/Codes/Test')
Cov = np.load(precompute_root / "Cov_{}x{}.npy".format(img_size, img_size))

###########################
# -- Acquisition parameters
###########################
print('Loading Acquisition parameters ( Covariance and Mean matrices) ')
# -- Compressed Reconstruction via CNN (CR = 1/8)
CR = 1024  # Number of patterns ---> M = 1024 measures

##################################
# -- Acquisition matrix definition
##################################

H = img_size * Hadamard_Transform_Matrix(img_size)
Ord = Cov2Var(Cov)
# -- Selection of Hadamard coefficients
Perm = Permutation_Matrix(Ord)
Pmat = np.dot(Perm, H)
Pmat = Pmat[:CR, :]
Pconv = matrix2conv(Pmat)
P, T = split(Pconv, 1)
P.bias.requires_grad = False
P.weight.requires_grad = False


# -- This is the same acquire function of the learning model
def forward_acquisition(x, b, c, h, w, m, Pattern):
    ##############################
    # --Scale input image to [0,1]
    ##############################
    x = (x + 1) / 2

    ########################
    # --Acquisition patterns
    ########################
    x = x.view(b * c, 1, h, w)
    Px = Pattern.to(x.device)
    x = Px(x)
    x = F.relu(x)
    x = x.view(b * c, 1, 2 * m)

    #########################################################
    # --Measurement noise (Gaussian approximation of Poisson)
    #########################################################
    x = x + torch.sqrt(x) * torch.randn_like(x)
    return x


"""
NVMS : This method precalculate a Noise Variance Matrix Stabilization (NVMS),
       which is a matrix that takes the mean of the variance of the noised measurements, 
       for a given photon level N0 on a batch of the STL-10 database. This method allows  
       to stabilize the signal dependent variance matrix in the denoising stage. 
"""


def NVMS(m, Pattern):
    # -- Hadamard Matrix definition (fht hadamard transform needs to be normalized)
    j = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # --We takes b * c measures on the STL-10 database
        b, c, h, w = inputs.shape
        x = forward_acquisition(inputs, b, c, h, w, m, Pattern)

        # --Index of measurements
        even_index = range(0, 2 * m, 2)
        uneven_index = range(1, 2 * m, 2)

        # -- Measurement recombination and variance calculation on whole batch
        x = x[:, :, even_index] + x[:, :, uneven_index]

        # -- Noise Variance Matrix Stabilization by mean estimation
        if j == 0:
            S = torch.sum(x, 0)
            C = b * c

        else:
            S += torch.sum(x, 0)
            C += b * c

        j += 1
        print(f'batch : {j}')

    mean_variance = torch.div(S, C)
    return mean_variance


Noise_Variance = NVMS(m=CR, Pattern=P).cpu().detach().numpy()
Noise_Variance = np.reshape(Noise_Variance, (CR,))
Noise_Variance = np.diag(Noise_Variance)

np.save(precompute_root / 'NVMS_N_{}_M_{}.npy'.format(img_size, CR), Noise_Variance)
print('Saved : NVMS_N_{}_M_{}.npy'.format(img_size, CR))
