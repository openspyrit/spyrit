r"""
=========================================
05. Denoised Completion Network (DC-Net)
=========================================
.. _tuto_dcnet_split_measurements:

This tutorial shows how to perform image reconstruction using a denoised
completion network (DC-Net) [1]_ with a trainable image denoiser.

.. figure:: ../fig/tuto5_dcnet.png
   :width: 600
   :align: center
   :alt: Reconstruction and neural network denoising architecture sketch using split measurements

|

.. [1] A Lorente Mur, P Leclerc, F Peyrin, and N Ducros, "Single-pixel image reconstruction from experimental data using neural networks," *Opt. Express*, Vol. 29, Issue 11, 17097-17110 (2021). `DOI <https://doi.org/10.1364/OE.424228>`_.
"""

# %%
# Load a batch of images
# -----------------------------------------------------------------------------

###############################################################################
# We load a batch of images from the :attr:`/images/` folder. Using the
# :func:`spyrit.misc.statistics.transform_gray_norm` function with the :attr:`normalize=False`
# argument returns images with values in (0,1).
import os
import torchvision
import torch.nn
from spyrit.misc.statistics import transform_gray_norm

spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, "images/")

# Grayscale images of size 64 x 64, no normalization to keep values in (0,1)
transform = transform_gray_norm(img_size=64, normalize=False)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=7)

x, _ = next(iter(dataloader))
print(f"Ground-truth images: {x.shape}")

###############################################################################
# We display the second image in the batch
from spyrit.misc.disp import imagesc

imagesc(x[1, 0, :, :], "x[1, 0, :, :]")

# %%
# Forward operators for split measurements
# =========================================

###############################################################################
# We consider Poisson noise, i.e., a noisy measurement vector given by
#
# .. math::
#       y \sim \mathcal{P}(\alpha A x),
#
# where :math:`\alpha` is a scalar value that represents the maximum image intensity (in photons), :math:`A \colon\, \mathbb{R}_+^{2M\times N}` is the acquisition matrix that contains the DMD patterns, :math:`x \in \mathbb{R}^N` is the signal of interest, :math:`2M` is the number of DMD patterns, and :math:`N` is the dimension of the signal.
#
# The larger :math:`\alpha`, the higher the signal-to-noise ratio of the measurements.

###############################################################################
# The acquisition matrix :math:`A` is chosen as a split Hadamard matrix. It is subsampled by a factor of four by retaining the rows that give, statistically, the coefficients with the largest variance. This is achieved by the :class:`~spyrit.core.meas.HadamSplit` class (see :ref:`Tutorial 5 <tuto_acquisition_operators_HadamSplit2d>` for details).

###############################################################################
# First, we download a covariance matrix (for subsampling).
from spyrit.misc.load_data import download_girder

# Get covariance matrix
url = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
dataId = "672207cbf03a54733161e95d"
data_folder = "./stat/"

file_abs_path = download_girder(url, dataId, data_folder)
Cov = torch.load(file_abs_path, weights_only=True)

######################################################################
# Then, we choose a subsampling factor of four and specify the subsampling strategy using the :attr:`order` attribute. Finally, we set the noise model using the :attr:`noise_model` attribute. We use the :class:`spyrit.core.noise.Poisson` class and set :math:`\alpha` to 100 photons.
from spyrit.core.torch import Cov2Var
from spyrit.core.meas import HadamSplit2d
from spyrit.core.noise import Poisson

M = 64 * 64 // 4
alpha = 100.0  # image intensity

Variance = Cov2Var(Cov)
noise_model = Poisson(alpha)
meas_op = HadamSplit2d(64, M=M, order=Variance, noise_model=noise_model)

######################################################################
# We simulate the measurements
y = meas_op(x)

# %%
# Pseudo inverse solution with preprocessing
# =========================================

######################################################################
# We compute the pseudo inverse solution using :class:`spyrit.core.recon.PinvNet`,
# which can include a preprocessing step
#
# .. math::
#       m = \texttt{Prep}(y).

###############################################################################
# We consider the :class:`spyrit.core.prep.UnsplitRescale` class that intends
# to "undo":
#
# * The splitting of an acquisition matrix (see :class:`spyrit.core.meas.LinearSplit`)
# * The scaling that controls the SNR of Poisson-corrupted measurements (see :class:`spyrit.core.noise.Poisson`).
#
# For this, we use the :class:`spyrit.core.prep.UnsplitRescale` class that computes
#
# .. math::
#       m = \frac{(y_+-y_-)}{\alpha},
# where :math:`y_+ = H_+ x` and :math:`y_- = H_- x`.
from spyrit.core.recon import PinvNet
from spyrit.core.prep import UnsplitRescale

prep_op = UnsplitRescale(alpha)
pinvnet = PinvNet(meas_op, prep_op)
###############################################################################
# Use GPU, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

pinvnet = pinvnet.to(device)
y = y.to(device)

###############################################################################
# Reconstruction
with torch.no_grad():
    x_pinv = pinvnet.reconstruct(y)

###############################################################################
# We display the second image in the batch
imagesc(x_pinv[1, 0, :, :].cpu(), "pinv")

######################################################################
# .. note::
#   Thanks to preprocessing, the reconstructed image has values in the range (0, 1), like the ground truth image.


# %%
# Denoised Completion Network (DC-Net)
# =========================================

######################################################################
# A DC-Net is based on four sequential steps:
#
# 1. Denoising in the measurement domain.
#
# 2. Estimation of the missing measurements from the denoised ones.
#
# 3. Image-domain mapping.
#
# 4. (Learned) Denoising in the image domain.
#
# Typically, only the last step involves learnable parameters.

# %%
# Denoised Completion
# =========================================

######################################################################
# The first three steps implement denoised completion, which corresponds to Tikhonov regularization. Considering linear measurements :math:`m = Hx`, where :math:`H` is the  measurement matrix and :math:`x` is the unknown image, it estimates :math:`x` from :math:`y` by minimizing
#
# .. math::
#    \| m - Hx \|^2_{\Gamma^{-1}} + \|x\|^2_{\Sigma^{-1}},
#
# where :math:`\Sigma` is a covariance prior and :math:`\Gamma` is the noise covariance. Denoised completation can be performed using  the :class:`~spyrit.core.recon.TikhonovMeasurementPriorDiag` class (see documentation for more details).

######################################################################
# In practice, it is more convenient to use the :class:`spyrit.core.recon.DCNet` class, which relies on a forward operator, a preprocessing operator, and a covariance prior.
from spyrit.core.recon import DCNet

dcnet = DCNet(meas_op, prep_op, Cov / 4, device=device)

######################################################################
# .. note::
#   We divide the covariance by four because it was computed using images with values in the range (-1, 1), whereas our images are in the range (0, 1). Therefore, the covariance is four times larger than expected.

###############################################################################
# Reconstruction
dcnet = dcnet.to(device)
with torch.no_grad():
    x_dc = dcnet.reconstruct(y)

###############################################################################
# We display the second image in the batch
imagesc(x_dc[1, 0, :, :].cpu(), "denoised completion")

######################################################################
# .. note::
#   In this tutorial, the covariance matrix used to define subsampling is also used as prior for reconstruction.

# %%
# (Learned) Denoising in the image domain
# =========================================


###############################################################################
# We download the parameters of a (spyrit 2.4) UNet denoiser
from spyrit.misc.load_data import download_girder

url = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
model_folder = "./model/"
dataID = "67221559f03a54733161e960"  # unique ID of the file
model_cnn_path = download_girder(url, dataID, model_folder)

###############################################################################
# The UNet should be placed in an ordered dictionary and passed to a
# :class:`nn.Sequential`.
# SPyRiT 2.4 trains neural networks for images with values in the
# range (-1, 1), while SPyRiT 3 assumes images with values in the range (0, 1).
# This can be compensated for using :class:`spyrit.core.prep.Rerange`.
from spyrit.core.prep import Rerange
from typing import OrderedDict
from spyrit.core.nnet import Unet
from spyrit.core.train import load_net

rerange = Rerange((0, 1), (-1, 1))
denoiser = OrderedDict(
    {"rerange": rerange, "denoi": Unet(), "rerange_inv": rerange.inverse()}
)
denoiser = torch.nn.Sequential(denoiser)
load_net(model_cnn_path, denoiser, device, False)

######################################################################
# To implement denoising in the image domain, we pass the :class:`spyrit.core.nnet.Unet` denoiser to a :class:`spyrit.core.recon.DCNet`.
dcnet = DCNet(meas_op, prep_op, Cov, denoiser, device=device)
dcnet = dcnet.to(device)  # Use GPU, if available

######################################################################
# We reconstruct the image
with torch.no_grad():
    x_dcnet = dcnet.reconstruct(y)

###############################################################################
# We display the second image in the batch
# sphinx_gallery_thumbnail_number = 4
im = imagesc(x_dcnet[1, 0, :, :].cpu(), "denoised completion")

# %%
# Results
# =========================================
import matplotlib.pyplot as plt
from spyrit.misc.disp import add_colorbar, noaxis

i_im = 1

f, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot the ground-truth image
im1 = axs[0, 0].imshow(x[i_im, 0, :, :], cmap="gray")
axs[0, 0].set_title("Ground-truth image", fontsize=16)
noaxis(axs[0, 0])
add_colorbar(im1, "bottom")

# Plot the pseudo inverse solution
im2 = axs[0, 1].imshow(x_pinv.cpu()[i_im, 0, :, :], cmap="gray")
axs[0, 1].set_title("Pseudoinverse", fontsize=16)
noaxis(axs[0, 1])
add_colorbar(im2, "bottom")

# Plot the solution obtained from denoised completion
im3 = axs[1, 0].imshow(x_dc.cpu()[i_im, 0, :, :], cmap="gray")
axs[1, 0].set_title("Denoised completion", fontsize=16)
noaxis(axs[1, 0])
add_colorbar(im3, "bottom")

# Plot the solution obtained from denoised completion with UNet denoising
im4 = axs[1, 1].imshow(x_dcnet.cpu()[i_im, 0, :, :], cmap="gray")
axs[1, 1].set_title("Denoised completion + UNet", fontsize=16)
noaxis(axs[1, 1])
add_colorbar(im4, "bottom")

######################################################################:
# While the pseudo inverse reconstrcution is pixelized, the solution obtained by denoised completion is smoother. DCNet with UNet provides the best reconstruction.

######################################################################
# .. note::
#   We refer to `spyrit-examples <https://github.com/openspyrit/spyrit-examples/blob/master/2025_spyrit_v3/figure_3.py>`_ for a comparison of several methods (e.g., pinvNet, DCNet, DRUNet).
