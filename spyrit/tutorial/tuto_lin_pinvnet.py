
r"""
1.1. Pseudoinverse for linear measurements
======================
This tutorial shows how to simulate data and perform image reconstruction. 
The measurement operator is a Hadamard matrix with positive coefficients. 
Note that this matrix can be replaced with the desired matrix. Undersampled 
measurements are simulated by selecting the undersampling factor. 

"""

import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import os
from spyrit.core.prep import DirectPoisson
from spyrit.core.recon import PinvNet
from spyrit.core.meas import Linear, HadamSplit
from spyrit.core.noise import NoNoise, Poisson
from spyrit.misc.statistics import Cov2Var, data_loaders_stl10, transform_gray_norm
from spyrit.misc.disp import imagesc
from spyrit.misc.walsh_hadamard import walsh2_matrix
from spyrit.misc.sampling import Permutation_Matrix

h = 32                  # image size hxh 
und = 2                 # undersampling factor
M = 32*32 // und        # number of measurements (undersampling factor = 2)
B = 10                  # batch size
alpha = 100             # number of mean photon counts
ind_img = 1             # Image index (modify to change the image)
mode_noise = False      # noiseless or 'poisson'

spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, '../images')
cov_name = os.path.join(spyritPath, '../../stat/Cov_64x64.npy')

# use GPU, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% 
# Load data
# ---------------------
# 
# Images *x* for training expect values in [-1,1]. The images are normalized
# using the *transform_gray_norm* function.

# A batch of images
#dataloaders = data_loaders_stl10('../../data', img_size=h, batch_size=10)  
#x, _ = next(iter(dataloaders['train']))

# N.B.: no view here compared to previous example
# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=h)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = min(B, len(dataset)))

# Select image
x0, _ = next(iter(dataloader))
x0 = x0[ind_img:6,:,:,:]
x = x0.detach().clone()
b,c,h,w = x.shape
#x = x.view(b*c,h*w)
print(f'Shape of incoming image (b*c,h*w): {x.view(b*c,h*w).shape}')


# plot
x_plot = x.view(-1,h,h).cpu().numpy() 
imagesc(x_plot[0,:,:], 'Ground-truth image normalized to [-1,1]')

# %% 
# Measurement operator
# ---------------------
#
# Measurements comprises three operators:
#
# 1. Measurement operator:
#     An operator that applies the measurement matrix *H* to the image.     
#     For instance, we can define a linear operator, *spyrit.core.meas.Linear(nn.Module)*, 
#     that applies the measurement matrix *H* to the image:
#
#         meas_op = Linear(H, pinv=True) 
#    
# 2. Normalization operator: 
#     An operator that normalizes the image *x* from [-1,1] to an image in [0,1]
# .. math::
#       \tilde{x}=\frac{x+1}{2}
#      
#     For a noiseless case, the operator *spyrit.core.NoNoise(nn.Module)* is used:
#         noise = NoNoise(meas_op)      
#
#     Measurements are then obtained for :math:`\tilde{x}` as 
# .. math::
#       y=H\tilde{x}=\frac{H(x+1)}{2}.
#     
#
# 3. Preprocessing operator:
#     The preprocessing operator allows to convert the measurements *y* to 
#     measurements *m* for the original image *x*. For instance, using the 
#     operator *spyrit.core.prep.DirectPoisson(nn.Module)*, the measurements $m$ for $x$ are 
#     then obtained as
# .. math::
#       m=2y-H*I.
#     
#        
# Similarly, for the Poisson case, :math:`y=\alpha \mathcal{P}(H\tilde{x})` and 
# :math:`m=\frac{2y}{\alpha}-H\mathbf{I}`.
#
#
#
# Prior to reconstruction, images are normalized so :math:`\tilde{x}` in [0,1] 
# using *NoNoise(nn.Module)*. By defening a linear operator equal to the identity, 
# the measurements are then the normalized images.

# Linear operator
meas_op_eye = Linear(np.eye(h*h), pinv=True) 
noise_op_eye = NoNoise(meas_op_eye)  
y_eye = noise_op_eye(x.view(b*c,h*w))

# plot
x_plot = y_eye.view(-1,h,h).cpu().numpy() 
imagesc(x_plot[0,:,:], r'Image $\tilde{x}$ in [0, 1]')


###############################################################################
# We now define the measurement linear operator from a given matrix of choice. 
# For instance, we can use a Hadamard matrix with positive coefficients or 
# a identity operator with randomly selected coefficients set to zero, as 
# in the case of impainting. Measurements *y* are then obtained as 
#
#     H = extract_hadamard_matrix(M, h)
#     meas_op = Linear(H, pinv=True) 
#     noise = NoNoise(meas_op)  
#     y = noise(x.view(b*c,h*w))
#
# Note, that if we apply directly *Linear*, we would obtained negative values 
# as *NoNoise* is not applied.

# Measurement matrix as positive Hadamard matrix
def extract_hadamard_matrix(M, h):
    # Compute a Hadamard matrix, take positive values only and undersample it
    # meas_op = HadamSplit(M, h, Ord)
    F = walsh2_matrix(h)
    F = np.where(F>0, F, 0)
    Cov = np.eye(h*h)
    Ord = Cov2Var(Cov)
    Perm = Permutation_Matrix(Ord)
    F = Perm@F 
    H = F[:M,:]
    return H

# Measurement matrix for simple random impainting
def lin_operator_inpainting(h, M):
    H = np.eye(h*h)
    # Define randon indices
    ind = np.random.choice(h*h, h*h-M, replace=False)
    #H[ind,:] = 0
    H = H[ind,:]
    return H

H = extract_hadamard_matrix(M, h)
#H = lin_operator_inpainting(h, M)
print(f"Shape of the measurement matrix: {H.shape}")

# Measurement operator
meas_op = Linear(H, pinv=True) 
meas_op.h, meas_op.w = h, h     

# Normalization operator
if mode_noise is False:
    noise = NoNoise(meas_op)        # noiseless
    prep = DirectPoisson(1.0, meas_op)
elif mode_noise == 'poisson':
    noise = Poisson(meas_op, alpha) # poisson noise
    prep = DirectPoisson(alpha, meas_op)

# Simulate measurements
# y_lin = meas_op(x.view(b*c,h*w))
y = noise(x.view(b*c,h*w))

# plot
print(f'Shape of preprocessed data y: {y.shape}')
x_plot = y.view(b*c,h//und,h).cpu().numpy() 
imagesc(x_plot[0,:,:], 'Linear noiseless measurements y=noise(x)')

# Preprocessing operator to convert measurements to original image *x* in [-1,1]
# which would be used during only training 
m = prep(y)

# plot
print(f'Shape of simulated measurements m: {m.shape}')
x_plot = m.view(b*c,h//und,h).cpu().numpy() 
imagesc(x_plot[0,:,:], 'Linear noiseless prep measurements m=DirectPoisson(x)')

# %% 
# PinvNet Network 
# ---------------------
#
# PinvNet allows to perform image reconstruction using the pseudoinverse. 
# *spyrit.core.recon.PinvNet* includes the measurement operator, 
# the noise model and reconstruction. 
# Measurements can be obtained as 
#   y = pinv_net.acquire(x)
# Alternatively, the measurements can be obtained as
#   y = noise(x)
#
# The reconstruction can be obtained as
#   z = pinv_net.reconstruct(y)
# or as 
#   z = pinv_net(x)       

pinv_net = PinvNet(noise, prep)

pinv_net = pinv_net.to(device)
x = x.to(device)

# measurements and images
with torch.no_grad():
    y = pinv_net.acquire(x)
    z = pinv_net.reconstruct(y.to(device))
#z = pinv_net(x)

# reshape
x_plot = x.view(-1,h,h).cpu().numpy() 
z_plot = z.view(-1,h,h).cpu().numpy()
z_plot[0,0,0] = 0.0

# plot
imagesc(z_plot[0,:,:], f'Reconstructed image with PinvNet')

plt.show()

###############################################################################
# Postprocessing can be added as a last layer of PinvNet, as shown in the 
# next tutorial.
