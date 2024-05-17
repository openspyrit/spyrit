r"""
07. DCNet with DRUNet denoising for split measurements
==========================
.. _tuto_dcdrunet_split_measurements:

This tutorial shows how to perform image reconstruction with DCNet
(data completion network) and DRUNet denoiser,
for single-pixel imaging. DRUNet is a pretrained plug-and-play (PnP) denoising network
that has been pretrained for a wide range of noise levels and admits the noise level
as an input. Thus, it requires no training while providing state-of-the-art postprocessing
performance.

As in previous tutorials, we consider split Hadamard operator and poisson noise
(see :ref:`Acquisition - split measurements <tuto_acquisition_split_measurements>`).
"""

###############################################################################
# .. note::
#
#       DRUNet has been taken from https://github.com/cszn/DPIR
#       Deep Plug-and-Play Image Restoration (DPIR) toolbox
#       June 2023

###############################################################################
# .. rubric:: References for DRUNet
#
# .. [ZhLZ21] Zhang, K.; Li, Y.; Zuo, W.; Zhang, L.; Van Gool, L.; Timofte, R..: Plug-and-Play Image Restoration with Deep Denoiser Prior. In: IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10), 6360-6376, 2021.
# .. [ZhZG17] Zhang, K.; Zuo, W.; Gu, S.; Zhang, L..: Learning Deep CNN Denoiser Prior for Image Restoration. In: IEEE Conference on Computer Vision and Pattern Recognition, 3929-3938, 2017.

import numpy as np
import os
from spyrit.misc.disp import imagesc
import matplotlib.pyplot as plt


# %%
# Load a batch of images
# -----------------------------------------------------------------------------

###############################################################################
# Images :math:`x` for training neural networks expect values in [-1,1]. The images are normalized
# using the :func:`transform_gray_norm` function.

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
# Forward operators for split measurements
# -----------------------------------------------------------------------------

###############################################################################
# We consider noisy split measurements for a Hadamard operator and a
# “variance subsampling” strategy that preserves the coefficients with the
# largest variance, obtained from a previously estimated covariance matrix
# (for more details, refer to :ref:`Acquisition - split measurements <tuto_acquisition_split_measurements>`).

###############################################################################
# First, we download the covariance matrix and load it.

import girder_client

# api Rest url of the warehouse
url = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"

# Generate the warehouse client
gc = girder_client.GirderClient(apiUrl=url)

# Download the covariance matrix and mean image
data_folder = "./stat/"
dataId_list = [
    "63935b624d15dd536f0484a5",  # for reconstruction (imageNet, 64)
    "63935a224d15dd536f048496",  # for reconstruction (imageNet, 64)
]
cov_name = "./stat/Cov_64x64.npy"

try:
    for dataId in dataId_list:
        myfile = gc.getFile(dataId)
        gc.downloadFile(dataId, data_folder + myfile["name"])

    print(f"Created {data_folder}")

    Cov = np.load(cov_name)
    print(f"Cov matrix {cov_name} loaded")
except:
    Cov = np.eye(h * h)
    print(f"Cov matrix {cov_name} not found! Set to the identity")

###############################################################################
# We define the measurement, noise and preprocessing operators and then
# simulate a measurement vector :math:`y` corrupted by Poisson noise. As in the previous tutorial,
# we simulate an accelerated acquisition by subsampling the measurement matrix
# by retaining only the first :math:`M` rows of a Hadamard matrix :math:`\textrm{Perm} H`.

from spyrit.core.meas import HadamSplit
from spyrit.core.noise import Poisson
from spyrit.misc.sampling import meas2img2
from spyrit.misc.statistics import Cov2Var
from spyrit.core.prep import SplitPoisson

# Measurement parameters
M = 64 * 64 // 4  # Number of measurements (here, 1/4 of the pixels)
alpha = 100.0  # number of photons

# Measurement and noise operators
Ord = Cov2Var(Cov)
meas_op = HadamSplit(M, h, Ord)
noise_op = Poisson(meas_op, alpha)
prep_op = SplitPoisson(alpha, meas_op)

# Vectorize image
x = x.view(b * c, h * w)
print(f"Shape of vectorized image: {x.shape}")

# Measurements
y = noise_op(x)  # a noisy measurement vector
m = prep_op(y)  # preprocessed measurement vector

m_plot = m.detach().numpy()
m_plot = meas2img2(m_plot.T, Ord)
imagesc(m_plot, r"Measurements $m$")

# %%
# DCNet with DRUNet denoising
# -----------------------------------------------------------------------------

##############################################################################
# We consider the DCNet network with DRUNet as the denoising layer.
# The definition of the DRUNet network is given in the :class:`spyrit.external.drunet.DRUNet` class,
# provided in the submodule :mod:`spyrit.external.drunet`. This class inherits from the original
# :class:`spyrit.external.drunet.UNetRes` class provided in the paper,
# with some modifications to handle the noise level as an input
# inside the DCNet network. DRUNet allows to denoise an image with any noise level by concatenating
# a noise level map to the input.

###############################################################################
# .. image:: fig/drunet.png
#    :width: 400
#    :align: center
#    :alt: Sketch of the network architecture with a DRUNet denoising layer

###############################################################################
# We define the DRUNetDen network by providing the number of channels and noise level :attr:`noise_level`,
# which is expected to be in [0, 255]. The larger the noise level, the higher the denoising.
# The noise level is not required as it can be set later on.

from spyrit.external.drunet import DRUNet

# Use GPU, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DRUnet denoising
# DRUNet(noise_level=5, n_channels=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose')
noise_level = 7
denoi_drunet = DRUNet(noise_level=noise_level, n_channels=1)

# Set the device for DRUNet
denoi_drunet = denoi_drunet.to(device)

###############################################################################
# We download the pretrained weights of the DRUNet and load them.

try:
    import gdown

    # Download pretrained weights
    model_drunet_path = "./model"
    url_drunet = "https://drive.google.com/file/d/1fhnIDJAbh7IRSZ9tgk4JPtfGra4O1ghk/view?usp=drive_link"

    if os.path.exists(model_drunet_path) is False:
        os.mkdir(model_drunet_path)
        print(f"Created {model_drunet_path}")

    model_drunet_path = os.path.join(model_drunet_path, "drunet_gray.pth")
    gdown.download(url_drunet, model_drunet_path, quiet=False, fuzzy=True)

    # Load pretrained weights
    denoi_drunet.load_state_dict(torch.load(model_drunet_path), strict=False)
    print(f"Model {denoi_drunet} loaded.")
except:
    print(f"Model {model_drunet_path} not found!")

###############################################################################
# We define the DCNet network by providing the measurement, noise and preprocessing operators,
# and the denoiser. The DCNet class :class:`spyrit.core.recon.DCNet` was
# introduced in the previous :ref:`DCNet tutorial <tuto_dcnet_split_measurements>`.
# Then, we reconstruct the image from the noisy measurements.

from spyrit.core.recon import DCNet

# Define DCNet with DRUNet denoising
dcnet_drunet = DCNet(noise_op, prep_op, torch.from_numpy(Cov), denoi=denoi_drunet)
dcnet_drunet = dcnet_drunet.to(device)

# Reconstruction
with torch.no_grad():
    z_dcnet_drunet = dcnet_drunet.reconstruct(
        y.to(device)
    )  # reconstruct from raw measurements

###############################################################################
# We can set other noise levels and reconstruct the images to see how 
# a higher noise level increases the blurring effect.

from spyrit.misc.disp import add_colorbar, noaxis

# Set new noise levels and reconstruct
with torch.no_grad():
    noise_level_2 = 1
    denoi_drunet.set_noise_level(noise_level_2)
    z_dcnet_drunet_2 = dcnet_drunet.reconstruct(y.to(device))

    noise_level_3 = 20
    denoi_drunet.set_noise_level(noise_level_3)
    z_dcnet_drunet_3 = dcnet_drunet.reconstruct(y.to(device))

# Plot the two reconstructions
x_plot = z_dcnet_drunet.view(-1, h, h).cpu().numpy()
x_plot2 = z_dcnet_drunet_2.view(-1, h, h).cpu().numpy()
x_plot3 = z_dcnet_drunet_3.view(-1, h, h).cpu().numpy()

f, axs = plt.subplots(1, 3, figsize=(10, 5))
im1 = axs[0].imshow(x_plot2[0, :, :], cmap="gray")
axs[0].set_title(f"DCNET with DRUNet\n (n map={noise_level_2})", fontsize=16)
noaxis(axs[0])
add_colorbar(im1, "bottom")

im2 = axs[1].imshow(x_plot[0, :, :], cmap="gray")
axs[1].set_title(f"DCNET with DRUNet\n (n map={noise_level})", fontsize=16)
noaxis(axs[1])
add_colorbar(im2, "bottom")

im3 = axs[2].imshow(x_plot3[0, :, :], cmap="gray")
axs[2].set_title(f"DCNET with DRUNet\n (n map={noise_level_3})", fontsize=16)
noaxis(axs[2])
add_colorbar(im3, "bottom")

# %%
# DRUNet denoising
# -----------------------------------------------------------------------------

###############################################################################
# Alternatively, we can reconstruct with DCNet and apply DRUNet in a second step,
# which is equivalent to DCNet with a DRUNet denoising layer but it is more involved.
# For that purpose, we define the DRUNet network from its original class
# :class:`spyrit.external.drunet.UNetRes` and load the pretrained weights.

from spyrit.external.drunet import UNetRes as drunet

# Define DCNet
dcnet = DCNet(noise_op, prep_op, torch.from_numpy(Cov))
dcnet = dcnet.to(device)

# Define denoising network
n_channels = 1  # 1 for grayscale image
drunet_den = drunet(in_nc=n_channels + 1, out_nc=n_channels)

# Load pretrained model
try:
    drunet_den.load_state_dict(torch.load(model_drunet_path), strict=True)
    print(f"Model {model_drunet_path} loaded.")
except:
    print(f"Model {model_drunet_path} not found!")
    load_drunet = False

# Set the device for DRUNet
drunet_den = drunet_den.to(device)

##############################################################################
# First, we reconstruct the image with DCNet and then we denoise the image with DRUNet.

# 1st step - Reconstruction
with torch.no_grad():
    z_dcnet = dcnet.reconstruct(y.to(device))  # reconstruct from raw measurements

# 2nd step - Denoising
# Convert to [0,1]
x_sample = 0.5 * (z_dcnet + 1).cpu()

# Create noise-level map and concatenate to the image
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

###############################################################################
# We plot all results side by side.

from spyrit.misc.disp import add_colorbar, noaxis

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
axs[0, 1].set_title("DCNet (without denoising)", fontsize=16)
noaxis(axs[0, 1])
add_colorbar(im2, "bottom")

im3 = axs[1, 0].imshow(x_plot3[0, :, :], cmap="gray")
axs[1, 1].set_title(f"DCNet with DRUNet (n map={noise_level})", fontsize=16)
noaxis(axs[1, 0])
add_colorbar(im3, "bottom")

im4 = axs[1, 1].imshow(x_plot4[0, :, :], cmap="gray")
axs[1, 0].set_title(f"1) DCNet. 2) DRUNet (n map={noise_level})", fontsize=16)
noaxis(axs[1, 1])
add_colorbar(im4, "bottom")

plt.show()

###############################################################################
# We see the results by DCNet (without denoising), DCNet with DRUNet layer,
# and DCNet with DRUNet denoising in two separated networks.
# Note that the last two results are equivalent, as expected.

###############################################################################
# .. note::
#
#       We refer to `spyrit-examples tutorials <http://github.com/openspyrit/spyrit-examples/tree/master/tutorial>`_
#       for a comparison of different solutions for split measurements (pinvNet, DCNet and DRUNet) that can be run in colab.
#
#       In this tutorial, we have used the PnP DRUNet denoiser for DCNet network but it can be added
#       to any other network, such as pinvNet. In addition, we have used the pretrained weights of DRUNet
#       as a PnP strategy that does not require training. However, the DCNet-DRUNet network can be trained
#       end-to-end to improve the reconstruction performance in a specific setting (where training is done
#       for all noise levels at once). For more details, refer to the paper [ZhLZ21]_.
