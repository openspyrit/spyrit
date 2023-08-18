r"""
.. _tuto_train_pseudoinverse_cnn_linear:
04. Train pseudoinverse solution + CNN denoising from linear measurements
======================
This tutorial shows how to train the pseudoinverse with a CNN denoiser for 
reconstruction of linear measurements used in tutorial :ref:`tuto_pseudoinverse_cnn_linear`. 
We have used a small CNN as an example, which can be replaced by the denoiser 
of your choice, for example Unet. Training is performed on the STL-10 dataset. 

You can use Tensorboard for Pytorch to visualize the training process (losses) as well 
as intermediate results (reconstructed images at different epochs).

The measurement operator is chosen as a Hadamard matrix with positive coefficients. 
Note that this matrix can be replaced any the desired matrix. 
"""

# %% 
# We define a dataloader for STL-10 dataset using :func:`spyrit.misc.statistics.data_loaders_stl10`.
# This will download the dataset if it is not already downloaded. It is based on torch.utils.data.DataLoader 
# and it creates a generator that returns a batch of images and labels at each iteration.

from spyrit.misc.statistics import data_loaders_stl10
from pathlib import Path

# Parameters
h = 64                  # image size hxh 
data_root = Path("./data")   # path to data folder (where the dataset is stored)
batch_size = 512

# Dataloader for STL-10 dataset
dataloaders = data_loaders_stl10(data_root, 
                                    img_size=h, 
                                    batch_size=batch_size, 
                                    seed=7,
                                    shuffle=True, download=True)  


# %% 
# Define a measurement operator
#------------------------------

###############################################################################
# We consider the sample operator as in the previous tutorials 
# (see :ref:`tuto_pseudoinverse_linear`). 
# We consider the case where the measurement matrix is the positive
# component of a Hadamard matrix, which if often used in single-pixel imaging.
# First, we compute a full Hadamard matrix that computes the 2D transforme of an
# image of size :attr:`h` and take its positive part. 
# Then, we subsample the rows of the measurement matrix to simulate an 
# accelerated acquisition. 
# To keep the low-frequency Hadamard coefficients, we choose a sampling map 
# with ones in the top left corner and zeros elsewhere.
# After permutation of the full Hadamard matrix, we keep only its first 
# :attr:`M` rows

from spyrit.misc.walsh_hadamard import walsh2_matrix
import numpy as np
import math 
from spyrit.misc.disp import imagesc
from spyrit.misc.sampling import Permutation_Matrix

und = 4                # undersampling factor
M = h**2 // und        # number of measurements (undersampling factor = 4)

F = walsh2_matrix(h)
F = np.where(F>0, F, 0)


Sampling_map = np.ones((h,h))
M_xy = math.ceil(M**0.5)
Sampling_map[:,M_xy:] = 0
Sampling_map[M_xy:,:] = 0

# imagesc(Sampling_map, 'low-frequency sampling map')

Perm = Permutation_Matrix(Sampling_map)
F = Perm@F 
H = F[:M,:]

print(f"Shape of the measurement matrix: {H.shape}")

###############################################################################
# Then, we instantiate a :class:`spyrit.core.meas.Linear` measurement operator 
# and a :class:`spyrit.core.noise.NoNoise` noise operator for noiseless case.
# We recall that we can simulate the measurements by using the noise operator.
from spyrit.core.meas import Linear
from spyrit.core.noise import NoNoise
meas_op = Linear(H, pinv=True)  
noise = NoNoise(meas_op)        

###############################################################################
# Finally, we define the preprocessing measurements operator corresponding to an 
# image in [-1,1]. For this, we use the :class:`spyrit.core.prep.DirectPoisson` 
# with :math:`\alpha` = 1, which allows to compensate for the image normalisation 
# achieved by :class:`spyrit.core.noise.NoNoise`.

from spyrit.core.prep import DirectPoisson
N0 = 1.0            # Mean maximum total number of photons
prep = DirectPoisson(N0, meas_op) # "Undo" the NoNoise operator

# %% 
# PinvNet Network 
# ---------------

###############################################################################
# We consider the :class:`spyrit.core.recon.PinvNet` class that reconstructs an
# image by computing the pseudoinverse solution, which is fed to a neural 
# network denoiser. First, we define the denoiser as a small CNN using the 
# :class:`spyrit.core.nnet.ConvNet` class. Then, we define the PinvNet network
# with the noise and preprocessing operators defined above and with the denoiser.

import torch
from spyrit.core.nnet import ConvNet
from spyrit.core.recon import PinvNet

denoiser = ConvNet()
model = PinvNet(noise, prep, denoi=denoiser)

# Send to GPU if available 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)


# %%
# Define a Loss function optimizer and scheduler
#
# ----------------

###############################################################################
# In order to train the network, we need to define a loss function, an optimizer
# and a scheduler. We use the Mean Square Error (MSE) loss function, weigh decay 
# loss and the Adam optimizer. We use a scheduler to decrease the learning rate
# by a factor of :attr:`gamma` every :attr:`step_size` epochs.

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from spyrit.core.train import save_net, Weight_Decay_Loss

# Parameters
lr = 1e-3
step_size = 10
gamma = 0.5

loss = nn.MSELoss()
criterion = Weight_Decay_Loss(loss)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# %%
# Train the network
#
# ----------------

###############################################################################
# To train the network, we use the :func:`~spyrit.core.train.train_model` function, 
# which handles the training process. We loop over our data iterator, feed the inputs to the
# network and optimize. You can train for one epoch only to check that everything works fine. 
# The training process can be monitored using Tensorboard by typing in the console:
# tensorboard --logdir runs

from spyrit.core.train import train_model
from datetime import datetime

# Parameters
model_root = Path("./model")# path to model saving files
num_epochs = 1              # number of training epochs (num_epochs = 30)
checkpoint_interval = 5     # interval between saving model checkpoints 
tb_path = True              

# Path for Tensorboard experiment tracking logs
now = datetime.now().strftime('%Y-%m-%d_%H-%M')
tb_path = f'runs/runs_stdl10_n1_m1024/{now}'

# Train the network
model, train_info = train_model(model, criterion, \
            optimizer, scheduler, dataloaders, device, model_root, num_epochs=num_epochs,\
            disp=True, do_checkpoint=checkpoint_interval, tb_path=tb_path)
    
# %%
#  Saving the model so that it can later be utilized
#
# ----------------

###############################################################################
# We save the model so that it can later be utilized. We save the network's
# architecture, the training parameters and the training history.

from spyrit.core.train import save_net

# Training parameters
train_type = 'N0_{:g}'.format(N0) 
arch = 'pinv-net'
denoi = 'cnn'
data = 'stl10'
reg = 1e-7      # Default value
suffix = 'N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}'.format(\
        h, M, num_epochs, lr, step_size,gamma, batch_size)
title = model_root / f'{arch}_{denoi}_{data}_{train_type}_{suffix}'    
print(title)

Path(model_root).mkdir(parents=True, exist_ok=True)

if checkpoint_interval:
    Path(title).mkdir(parents=True, exist_ok=True)
    
save_net(title, model)

# Save training history
import pickle
from spyrit.core.train import Train_par

params = Train_par(batch_size, lr, h, reg=reg)
params.set_loss(train_info)
train_path = model_root / f'TRAIN_{arch}_{denoi}_{data}_{train_type}_{suffix}.pkl'

with open(train_path, 'wb') as param_file:
    pickle.dump(params,param_file)
torch.cuda.empty_cache()

###############################################################################
# In a future tutorial, we will show how to train the network step by step.