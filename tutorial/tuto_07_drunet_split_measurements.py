r"""
======================================================================
07. DCNet with plug-and-play DRUNet denoising
======================================================================
.. _tuto_dcdrunet_split_measurements:

This tutorial shows how to perform image reconstruction using a DCNet (data completion network) that includes a `DRUNet denoiser <https://github.com/cszn/DPIR>`_. DRUNet is a pretrained plug-and-play denoising network that has been pretrained for a wide range of noise levels. DRUNet admits the noise level as an input. Contratry to the DCNet described in :ref:`Tutorial 6 <tuto_dcnet_split_measurements>`, it requires no training.
"""

######################################################################
# .. figure:: ../fig/drunet.png
#    :width: 600
#    :align: center
#    :alt: DCNet with DRUNet denoising in the image domain

######################################################################
# .. note::
#
#       As in the previous tutorials, we consider a split Hadamard operator and measurements corrupted by Poisson noise (see :ref:`Tutorial 5 <tuto_acquisition_split_measurements>`).

import numpy as np
import os
from spyrit.misc.disp import imagesc
import matplotlib.pyplot as plt


# %%
# Load a batch of images
# ====================================================================

######################################################################
# Images :math:`x` for training neural networks expect values in [-1,1]. The images are normalized using the :func:`transform_gray_norm` function.

# sphinx_gallery_thumbnail_path = 'fig/drunet.png'

from spyrit.misc.statistics import transform_gray_norm
import torchvision
import torch

h = 64  # image size hxh
i = 1  # Image index (modify to change the image)
spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, "images")


# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=h)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=7)

x, _ = next(iter(dataloader))
print(f"Shape of input images: {x.shape}")

# Select image
x = x[i : i + 1, :, :, :]
x = x.detach().clone()
b, c, h, w = x.shape

# plot
x_plot = x.view(-1, h, h).cpu().numpy()
imagesc(x_plot[0, :, :], r"$x$ in [-1, 1]")

# %%
# Operators for split measurements
# ====================================================================

######################################################################
# We consider noisy measurements obtained from a split Hadamard operator, and a subsampling strategy that retaines the coefficients with the largest variance (for more details, refer to :ref:`Tutorial 5 <tuto_acquisition_split_measurements>`).

######################################################################
# First, we download the covariance matrix from our warehouse.

from spyrit.misc.load_data import download_girder

# download parameters
url = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"
dataId = "63935b624d15dd536f0484a5"
data_folder = "./stat/"
cov_name = "Cov_64x64.npy"
# download the covariance matrix
file_abs_path = download_girder(url, dataId, data_folder, cov_name)

try:
    Cov = np.load(file_abs_path)
    print(f"Cov matrix {cov_name} loaded")
except:
    Cov = np.eye(h * h)
    print(f"Cov matrix {cov_name} not found! Set to the identity")

######################################################################
# We define the measurement, noise and preprocessing operators and then simulate a measurement vector corrupted by Poisson noise. As in the previous tutorials, we simulate an accelerated acquisition by subsampling the measurement matrix by retaining only the first rows of a Hadamard matrix that is permuted looking at the diagonal of the covariance matrix.


from spyrit.core.meas import HadamSplit
from spyrit.core.noise import Poisson
from spyrit.misc.sampling import meas2img
from spyrit.misc.statistics import Cov2Var
from spyrit.core.prep import SplitPoisson

# Measurement parameters
M = 64 * 64 // 4  # Number of measurements (here, 1/4 of the pixels)
alpha = 100.0  # number of photons

# Measurement and noise operators
Ord = Cov2Var(Cov)
meas_op = HadamSplit(M, h, torch.from_numpy(Ord))
noise_op = Poisson(meas_op, alpha)
prep_op = SplitPoisson(alpha, meas_op)

# Vectorize image
x = x.view(b * c, h * w)
print(f"Shape of vectorized image: {x.shape}")

# Measurements
y = noise_op(x)  # a noisy measurement vector
m = prep_op(y)  # preprocessed measurement vector

m_plot = m.detach().numpy()
m_plot = meas2img(m_plot, Ord)
imagesc(m_plot[0, :, :], r"Measurements $m$")

# %%
# DRUNet denoising
# ====================================================================

######################################################################
# DRUNet is defined by the :class:`spyrit.external.drunet.DRUNet` class. This class inherits from the original :class:`spyrit.external.drunet.UNetRes` class introduced in [ZhLZ21]_, with some modifications to handle different noise levels.

###############################################################################
# We instantiate the DRUNet by providing the noise level, which is expected to be in [0, 255], and the number of channels. The larger the noise level, the higher the denoising.

from spyrit.external.drunet import DRUNet

noise_level = 7
denoi_drunet = DRUNet(noise_level=noise_level, n_channels=1)

# Use GPU, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
denoi_drunet = denoi_drunet.to(device)

###############################################################################
# We download the pretrained weights of the DRUNet and load them.

from spyrit.misc.load_data import download_girder

# Load pretrained model
url = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
dataID = "667ebf9ebaa5a9000705895e"  # unique ID of the file
local_folder = "./model/"
data_name = "tuto7_drunet_gray.pth"
model_drunet_abs_path = download_girder(url, dataID, local_folder, data_name)

# Load pretrained weights
denoi_drunet.load_state_dict(torch.load(model_drunet_abs_path), strict=False)

# %%
# Pluggind the DRUnet in a DCNet
# ====================================================================

######################################################################
# We define the DCNet network by providing the forward operator, preprocessing operator, covariance prior and denoising prior. The DCNet class :class:`spyrit.core.recon.DCNet` is discussed in :ref:`Tutorial 06 <tuto_dcnet_split_measurements>`.

from spyrit.core.recon import DCNet

dcnet_drunet = DCNet(noise_op, prep_op, torch.from_numpy(Cov), denoi=denoi_drunet)
dcnet_drunet = dcnet_drunet.to(device)  # Use GPU, if available

######################################################################
# Then, we reconstruct the image from the noisy measurements.

with torch.no_grad():
    z_dcnet_drunet = dcnet_drunet.reconstruct(y.to(device))

# %%
# Tunning of the denoising
# ====================================================================

######################################################################
# We reconstruct the images for another two different noise levels of DRUnet

noise_level_2 = 1
noise_level_3 = 20

with torch.no_grad():

    denoi_drunet.set_noise_level(noise_level_2)
    z_dcnet_drunet_2 = dcnet_drunet.reconstruct(y.to(device))

    denoi_drunet.set_noise_level(noise_level_3)
    z_dcnet_drunet_3 = dcnet_drunet.reconstruct(y.to(device))

######################################################################
# Plot all reconstructions
from spyrit.misc.disp import add_colorbar, noaxis

x_plot = z_dcnet_drunet.view(-1, h, h).cpu().numpy()
x_plot2 = z_dcnet_drunet_2.view(-1, h, h).cpu().numpy()
x_plot3 = z_dcnet_drunet_3.view(-1, h, h).cpu().numpy()

f, axs = plt.subplots(1, 3, figsize=(10, 5))
im1 = axs[0].imshow(x_plot2[0, :, :], cmap="gray")
axs[0].set_title(f"DRUNet\n (n map={noise_level_2})", fontsize=16)
noaxis(axs[0])
add_colorbar(im1, "bottom")

im2 = axs[1].imshow(x_plot[0, :, :], cmap="gray")
axs[1].set_title(f"DRUNet\n (n map={noise_level})", fontsize=16)
noaxis(axs[1])
add_colorbar(im2, "bottom")

im3 = axs[2].imshow(x_plot3[0, :, :], cmap="gray")
axs[2].set_title(f"DRUNet\n (n map={noise_level_3})", fontsize=16)
noaxis(axs[2])
add_colorbar(im3, "bottom")

# %%
# Alternative implementation showing the advantage of the :class:`~spyrit.external.drunet.DRUNet` class
# ====================================================================

##############################################################################
# First, we consider DCNet without denoising in the image domain (default behaviour)

dcnet = DCNet(noise_op, prep_op, torch.from_numpy(Cov))
dcnet = dcnet.to(device)

with torch.no_grad():
    z_dcnet = dcnet.reconstruct(y.to(device))

######################################################################
# Then, we instantiate DRUNet using the original class :class:`spyrit.external.drunet.UNetRes`.

from spyrit.external.drunet import UNetRes as drunet

# Define denoising network
n_channels = 1  # 1 for grayscale image
drunet_den = drunet(in_nc=n_channels + 1, out_nc=n_channels)

# Load pretrained model
try:
    drunet_den.load_state_dict(torch.load(model_drunet_abs_path), strict=True)
    print(f"Model {model_drunet_abs_path} loaded.")
except:
    print(f"Model {model_drunet_abs_path} not found!")
    load_drunet = False
drunet_den = drunet_den.to(device)

######################################################################
# To denoise the output of DCNet, we create noise-level map that we concatenate to the output of DCNet that we normalize in [0,1]

x_sample = 0.5 * (z_dcnet + 1).cpu()

#
x_sample = torch.cat(
    (
        x_sample,
        torch.FloatTensor([noise_level / 255.0]).repeat(
            1, 1, x_sample.shape[2], x_sample.shape[3]
        ),
    ),
    dim=1,
)
x_sample = x_sample.to(device)

with torch.no_grad():
    z_dcnet_den = drunet_den(x_sample)

##############################################################################
# We plot all results

x_plot = x.view(-1, h, h).cpu().numpy()
x_plot2 = z_dcnet.view(-1, h, h).cpu().numpy()
x_plot3 = z_dcnet_drunet.view(-1, h, h).cpu().numpy()
x_plot4 = z_dcnet_den.view(-1, h, h).cpu().numpy()

f, axs = plt.subplots(2, 2, figsize=(10, 10))
im1 = axs[0, 0].imshow(x_plot[0, :, :], cmap="gray")
axs[0, 0].set_title("Ground-truth image", fontsize=16)
noaxis(axs[0, 0])
add_colorbar(im1, "bottom")

im2 = axs[0, 1].imshow(x_plot2[0, :, :], cmap="gray")
axs[0, 1].set_title("No denoising", fontsize=16)
noaxis(axs[0, 1])
add_colorbar(im2, "bottom")

im3 = axs[1, 0].imshow(x_plot3[0, :, :], cmap="gray")
axs[1, 1].set_title(f"Using DRUNet with n map={noise_level}", fontsize=16)
noaxis(axs[1, 0])
add_colorbar(im3, "bottom")

im4 = axs[1, 1].imshow(x_plot4[0, :, :], cmap="gray")
axs[1, 0].set_title(f"Using UNetRes with n map={noise_level}", fontsize=16)
noaxis(axs[1, 1])
add_colorbar(im4, "bottom")

plt.show()

############################################################################### The results are identical to those obtained using :class:`~spyrit.external.drunet.DRUNet`.

###############################################################################
# .. note::
#
#       In this tutorial, we have used DRUNet with a DCNet but it can be used any other network, such as pinvNet. In addition, we have considered pretrained weights, leading to a plug-and-play strategy that does not require training. However, the DCNet-DRUNet network can be trained end-to-end to improve the reconstruction performance in a specific setting (where training is done for all noise levels at once). For more details, refer to the paper [ZhLZ21]_.

###############################################################################
# .. note::
#
#       We refer to `spyrit-examples tutorials <http://github.com/openspyrit/spyrit-examples/tree/master/tutorial>`_ for a comparison of different solutions (pinvNet, DCNet and DRUNet) that can be run in colab.

######################################################################
# .. rubric:: References for DRUNet
#
# .. [ZhLZ21] Zhang, K.; Li, Y.; Zuo, W.; Zhang, L.; Van Gool, L.; Timofte, R..: Plug-and-Play Image Restoration with Deep Denoiser Prior. In: IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10), 6360-6376, 2021.
# .. [ZhZG17] Zhang, K.; Zuo, W.; Gu, S.; Zhang, L..: Learning Deep CNN Denoiser Prior for Image Restoration. In: IEEE Conference on Computer Vision and Pattern Recognition, 3929-3938, 2017.
