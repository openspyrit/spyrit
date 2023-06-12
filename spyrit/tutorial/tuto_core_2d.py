r"""
01. Tutorial 2D
======================
This tutorial focuses on Bayesian inversion, a special type of inverse problem
that aims at incorporating prior information in terms of model and data
probabilities in the inversion process.
"""

import os
import numpy as np
import spyrit
from spyrit.core.meas import HadamSplit
from spyrit.core.noise import NoNoise
from spyrit.core.prep import SplitPoisson
from spyrit.core.recon import PseudoInverse
from spyrit.misc.statistics import data_loaders_stl10, transform_gray_norm
from spyrit.misc.disp import imagesc
from spyrit.misc.sampling import meas2img2

import torch
import torchvision
 
M = 64*64 // 2      
H = 64              
B = 10              

spyritPath = spyrit.__path__[0]
imgs_path = os.path.join(spyritPath, 'images')

###############################################################################
# So far we have been able to estimate our posterion mean. What about its
# uncertainties (i.e., posterion covariance)?

# Uncomment to download stl10 dataset
# A batch of images
# dataloaders = data_loaders_stl10('../../../data', img_size=H, batch_size=10)  
# dataloader = dataloaders['train']


# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=H)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = min(B, len(dataset)))

x, _ = next(iter(dataloader))
b,c,h,w = x.shape
x = x.view(b*c,h*w)

# Operators
Ord = np.ones((H,H))                
meas_op = HadamSplit(M, H, Ord)
noise = NoNoise(meas_op) # noiseless
prep = SplitPoisson(1.0, meas_op)
recon = PseudoInverse()

# measurements and images
y = noise(x)
m = prep(y)
z = recon(m, meas_op)

# reshape
x_plot = x.view(-1,H,H).numpy() 
m_plot = m.numpy()   
m_plot = meas2img2(m_plot.T, Ord)
m_plot = np.moveaxis(m_plot,-1, 0)
z_plot = z.view(-1,H,H).numpy()

# plot
imagesc(x_plot[0,:,:],'Ground-truth image')
imagesc(m_plot[0,:,:],'Measurement')
imagesc(z_plot[0,:,:],'Reconstructed image')

#%% Using the reconstruction network, no noise
import torch
import torchvision
import numpy as np
import spyrit
from spyrit.core.meas import HadamSplit
from spyrit.core.noise import NoNoise
from spyrit.core.prep import SplitPoisson
from spyrit.core.recon import PinvNet
from spyrit.misc.statistics import data_loaders_stl10, transform_gray_norm
from spyrit.misc.disp import imagesc
from spyrit.misc.sampling import meas2img2
 
M = 64*64 // 2
H = 64
B = 10

spyritPath = spyrit.__path__[0]
imgs_path = os.path.join(spyritPath, 'images')

# A batch of images
# dataloaders = data_loaders_stl10('../../../data', img_size=H, batch_size=10)  
# dataloader = dataloaders['train']
transform = transform_gray_norm(img_size=H)
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = min(B, len(dataset)))

x, _ = next(iter(dataloader))
# N.B.: no view here compared to previous example

# Operators
Ord = np.ones((H,H))
meas_op = HadamSplit(M, H, Ord)
noise = NoNoise(meas_op)    # noiseless
prep = SplitPoisson(1.0, meas_op)
pinv_net = PinvNet(noise, prep)

# use GPU, if available
#device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pinv_net = pinv_net.to(device)
x = x.to(device)

# measurements and images
z = pinv_net(x)

# reshape
x_plot = x.view(-1,H,H).cpu().numpy() 
#m_plot = m.numpy()   
#m_plot = meas2img2(m_plot.T, Ord)
#m_plot = np.moveaxis(m_plot,-1, 0)
z_plot = z.view(-1,H,H).cpu().numpy()

# plot
imagesc(x_plot[0,:,:], 'Ground-truth image')
#imagesc(m_plot[0,:,:],'Measurement')
imagesc(z_plot[0,:,:], f'Reconstructed image ({device})')

#%% Using the core modules only, Poisson noise
import torch
import torchvision
import numpy as np
import spyrit
from spyrit.core.meas import HadamSplit
from spyrit.core.noise import Poisson, PoissonApproxGauss
from spyrit.core.prep import SplitPoisson
from spyrit.core.recon import PseudoInverse
from spyrit.misc.statistics import data_loaders_stl10, transform_gray_norm
from spyrit.misc.disp import imagesc
from spyrit.misc.sampling import meas2img2
 

Mx = 64//1
alpha = 100.0 #ph/pixel max

M = Mx*Mx
H = 64
B = 10

spyritPath = spyrit.__path__[0]
imgs_path = os.path.join(spyritPath, 'images')

# A batch of images
# dataloaders = data_loaders_stl10('../../../data', img_size=H, batch_size=10)  
# dataloader = dataloaders['train']
transform = transform_gray_norm(img_size=H)
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = min(B, len(dataset)))

x, _ = next(iter(dataloader))
b,c,h,w = x.shape
x = x.view(b*c,h*w)

# Operators
Ord = np.zeros((H,H))        # regular sampling
Ord[0:Mx,0:Mx] = 1
meas_op = HadamSplit(M, H, Ord)
noise_op = Poisson(meas_op, alpha)
#noise_op = PoissonApproxGauss(meas_op, alpha)
split_op = SplitPoisson(alpha, meas_op)
recon_op = PseudoInverse()

# measurements and images
y = noise_op(x)
m = split_op(y)
z = recon_op(m, meas_op)

# reshape
x_plot = x.view(-1,H,H).numpy() 
m_plot = m.numpy()   
m_plot = meas2img2(m_plot.T, Ord)
m_plot = np.moveaxis(m_plot,-1, 0)
z_plot = z.view(-1,H,H).numpy()

# plot
imagesc(x_plot[0,:,:],'Ground-truth image')
imagesc(m_plot[0,:,:],'Preprocessed measurements')
imagesc(z_plot[0,:,:],'Reconstructed image')


#%% DCNet
import numpy as np
import torch
import torchvision
import spyrit
from spyrit.misc.statistics import Cov2Var
from spyrit.core.noise import NoNoise, Poisson
from spyrit.core.prep import SplitPoisson
from spyrit.core.meas import HadamSplit
from spyrit.core.recon import TikhonovMeasurementPriorDiag, PinvNet, DCNet
from spyrit.misc.statistics import data_loaders_stl10, transform_gray_norm
from spyrit.misc.disp import imagesc
from spyrit.misc.sampling import Permutation_Matrix, meas2img2


alpha = 10.0 #ph/pixel max
H = 64
M = H**2 // 8 # subsampled by factor 8
B = 10

spyritPath = spyrit.__path__[0]
imgs_path = os.path.join(spyritPath, 'images')

# init reconstrcution networks
#cov_file = '../../../stat/ILSVRC2012_v10102019/Cov_8_64x64.npy'
#cov_file = '../../../stat/stl10/Cov_64x64.npy'
#Cov = np.load(cov_file)
Cov = np.eye(H*H)
Ord = Cov2Var(Cov)
meas = HadamSplit(M, H, Ord)

noise = Poisson(meas, alpha)
#noise  = NoNoise(meas)    # noiseless
prep  = SplitPoisson(alpha, meas)
pinet = PinvNet(noise, prep)
dcnet = DCNet(noise, prep, Cov)

# A batch of images
# dataloaders = data_loaders_stl10('../../../data', img_size=H, batch_size=10)  
# dataloader = dataloaders['val']
transform = transform_gray_norm(img_size=H)
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = min(B, len(dataset)))

x, _ = next(iter(dataloader))

# use GPU, if available
#device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pinet = pinet.to(device)
dcnet = dcnet.to(device)
x = x.to(device)

# measurements
y = pinet.acquire(x)    # or equivalently here: y = dcnet.acquire(x)
m = pinet.meas2img(y)   # zero-padded images (after preprocessing)

# reconstruction
x_pi = pinet.reconstruct(y)
x_dc = dcnet.reconstruct(y)
x_dc_2 = dcnet(x)   # another reconstruction, from the ground-truth image

# reshape, send to CPU, convert 
x_plot = x.view(-1,H,H).cpu().numpy()    
m_plot = m.view(-1,H,H).cpu().numpy()
x_pi_plot = x_pi.view(-1,H,H).cpu().numpy()
x_dc_plot = x_dc.view(-1,H,H).cpu().numpy()
x_dc_2_plot = x_dc_2.view(-1,H,H).cpu().numpy()

# plot
b = 5
imagesc(x_plot[b,:,:], 'Ground-truth image')
imagesc(m_plot[b,:,:], 'Measurement')
imagesc(x_pi_plot[b,:,:], f'Pinv reconstruction #1 ({device})')
imagesc(x_dc_plot[b,:,:], f'DC reconstruction #1 ({device})')
imagesc(x_dc_2_plot[b,:,:], f'DC reconstruction #2 ({device})')

###############################################################################
# Note that here we have been able to compute a sample posterior covariance
# from its estimated samples. By displaying it we can see  how both the overall
# variances and the correlation between different parameters have become
# narrower compared to their prior counterparts.
