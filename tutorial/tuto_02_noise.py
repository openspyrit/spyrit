r"""
02. Noise operators
===================================================
.. _tuto_noise:

This tutorial shows how to use noise operators using the :mod:`spyrit.core.noise` submodule.

.. image:: ../fig/tuto2.png
   :width: 600
   :align: center
   :alt: Reconstruction architecture sketch

|
"""

# %%
# Load a batch of images
# -----------------------------------------------------------------------------

###############################################################################
# We load a batch of images from the `/images/` folder. Using the
# :func:`transform_gray_norm` function with the :attr:`normalize=False`
# argument returns images with values in (0,1).
import os

import torch
import torchvision
import matplotlib.pyplot as plt

from spyrit.misc.disp import imagesc
from spyrit.misc.statistics import transform_gray_norm

spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, "images/")

# Grayscale images of size 64 x 64, no normalization to keep values in (0,1)
transform = transform_gray_norm(img_size=64, normalize=False)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=7)

x, _ = next(iter(dataloader))
print(f"Shape of input images: {x.shape}")


###############################################################################
# We select the first image in the batch and plot it.

i_plot = 1
imagesc(x[i_plot, 0, :, :], r"$x$ in (0, 1)")


# %%
# Gaussian noise
# -----------------------------------------------------------------------------

###############################################################################
# We consider additive Gaussiane noise,
#
# .. math::
#       y \sim z + \mathcal{N}(0,\sigma^2),
#
# where :math:`\mathcal{N}(\mu, \sigma^2)` is a Gaussian distribution with mean :math:`\mu` and variance :math:`\sigma^2`, and :math:`z` is the noiseless image. The larger :math:`\sigma`, the lower the signal-to-noise ratio.

###############################################################################
# To add 10% Gaussian noise, we instantiate a :class:`spyrit.core.noise`
# operator with :attr:`sigma=0.1`.

from spyrit.core.noise import Gaussian

noise_op = Gaussian(sigma=0.1)
x_noisy = noise_op(x)

imagesc(x_noisy[1, 0, :, :], r"10% Gaussian noise")
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# To add 2% Gaussian noise, we update the class attribute :attr:`sigma`.

noise_op.sigma = 0.02
x_noisy = noise_op(x)

imagesc(x_noisy[1, 0, :, :], r"2% Gaussian noise")

# %%
# Poisson noise
# -----------------------------------------------------------------------------

###############################################################################
# We now consider Poisson noise,
#
# .. math::
#       y \sim \mathcal{P}(\alpha z), \quad z \ge 0,
#
# where :math:`\alpha \ge 0` is a scalar value that represents the maximum
# image intensity (in photons). The larger :math:`\alpha`, the higher the signal-to-noise ratio.

###############################################################################
# We consider the :class:`spyrit.core.noise.Poisson` class and set :math:`\alpha`
# to 100 photons (which corresponds to the Poisson parameter).

from spyrit.core.noise import Poisson
from spyrit.misc.disp import add_colorbar, noaxis

alpha = 100  # number of photons
noise_op = Poisson(alpha)

###############################################################################
# We simulate two noisy versions of the same images

y1 = noise_op(x)  # first sample
y2 = noise_op(x)  # another sample

###############################################################################
# We now consider the case :math:`\alpha = 1000` photons.

noise_op.alpha = 1000
y3 = noise_op(x)  # noisy measurement vector

###############################################################################
# We finally plot the noisy images

# plot
f, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].set_title("100 photons")
im = axs[0].imshow(y1[1, 0].reshape(64, 64), cmap="gray")
add_colorbar(im, "bottom")

axs[1].set_title("100 photons")
im = axs[1].imshow(y2[1, 0].reshape(64, 64), cmap="gray")
add_colorbar(im, "bottom")

axs[2].set_title("1000 photons")
im = axs[2].imshow(
    y3[
        1,
        0,
    ].reshape(64, 64),
    cmap="gray",
)
add_colorbar(im, "bottom")

noaxis(axs)

###############################################################################
# As expected the signal-to-noise ratio of the measurement vector is higher for
# 1,000 photons than for 100 photons
#
# .. note::
#   Not only the signal-to-noise, but also the scale of the measurements
#   depends on :math:`\alpha`, which motivates the introduction of the
#   preprocessing operator.
