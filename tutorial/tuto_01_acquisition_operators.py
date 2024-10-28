r"""
01. Acquisition operators
==========================
.. _tuto_acquisition_operators:

This tutorial shows how to simulate measurements using the :mod:`spyrit.core`
submodule. The simulation is based on three modules:

1. **Measurement operators** compute linear measurements :math:`y = Hx` from
   images :math:`x`, where :math:`H` is a linear operator (matrix) and :math:`x`
   is an image (see :mod:`spyrit.core.meas`)

2. **Noise operator** corrupts measurements :math:`y` with noise (see :mod:`spyrit.core.noise`)

3. **Preprocessing operators** are typically used to process the noisy
   measurements prior to reconstruction (see :mod:`spyrit.core.prep`)

.. image:: ../fig/tuto1.png
   :width: 600
   :align: center
   :alt: Measurement, noise, and preprocessing sketches

These tutorials load image samples from `/images/`.

Please note that as of v.2.4.0, the inputs to the measurement operators are
no longer vectorized images, but rather image tensors with shape :math:`(*, H, W)`,
where :math:`*` represents any number of additional dimensions, e.g. batch size
and number of channels.
"""

# %%
# Load a batch of images
# -----------------------------------------------------------------------------

###############################################################################
# Images :math:`x` for training neural networks expect values in [-1,1]. The images are normalized
# using the :func:`transform_gray_norm` function. Spyrit can handle images
# with the shape :math:`(h, w)` or :math:`(*, h, w)`, where :math:`*` represents
# any number of additional dimensions, e.g. batch size and number of channels.
# In this case, we load a batch of black and white images of size :math:`64 \times 64`,
# and select one image for the tutorial. This results in a tensor of shape :math:`(1, 1, 64, 64)`.

import os

import torch
import torchvision
import matplotlib.pyplot as plt

from spyrit.misc.disp import imagesc
from spyrit.misc.statistics import transform_gray_norm

# sphinx_gallery_thumbnail_path = 'fig/tuto1.png'

h = 64  # image size hxh
i = 1  # Image index (modify to change the image)
spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, "images/")


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
print(f"Shape of selected image: {x.shape}")
b, c, h, w = x.shape

# plot
imagesc(x[0, 0, :, :], r"$x$ in [-1, 1]")

# %%
# The measurement and noise operators
# -----------------------------------------------------------------------------

###############################################################################
# Noise operators are defined in the :mod:`~spyrit.core.noise` module. A noise
# operator computes the following three steps sequentially:
#
# 1. Normalization of the image :math:`x` with values in [-1,1] to get an image
# :math:`\tilde{x}=\frac{x+1}{2}` in [0,1], as it is required for measurement simulation
#
# 2. Application of the measurement model, i.e., computation of :math:`H\tilde{x}`
#
# 3. Application of the noise model
#
# .. math::
#       y \sim \texttt{Noise}(H\tilde{x}) = \texttt{Noise}\left(\frac{H(x+1)}{2}\right),
#
# The normalization is usefull when considering distributions such
# as the Poisson distribution that are defined on positive values.
#
# .. note::
#   The noise operator is constructed from a measurement operator (see the
#   :mod:`~spyrit.core.meas` submodule) in order to compute the measurements
#   :math:`H\tilde{x}`, as given by step #2.


# %%
# A simple example: identity measurement matrix and no noise
# -----------------------------------------------------------------------------

###############################################################################
# .. math::
#       y = \tilde{x}

###############################################################################
# We start with a simple example where the measurement matrix :math:`H` is
# the identity, which can be handled  by the more general
# :class:`spyrit.core.meas.Linear` class. We consider the noiseless case handled
# by the :class:`spyrit.core.noise.NoNoise` class.
#
# Usually, the measurement tensor is in another space than the image tensor (e.g. Fourier space or
# Hadamard space), but using the identity matrix results in the measurement
# vector being (identical and) in the same space as the image tensor. As measurements
# are always vectorized, the measurement vector is a vectorized image.

from spyrit.core.meas import Linear
from spyrit.core.noise import NoNoise

meas_op = Linear(torch.eye(h * h))
noise_op = NoNoise(meas_op)

###############################################################################
# We simulate the measurement vector :math:`y` that we want to visualise as an image.
# Note that the measurement vector :math:`y` lost a dimension compared to the image :math:`x`,
# because the measurement operator acts on the last 2 dimensions of the image tensor.

y_eye = noise_op(x)  # noisy measurement vector
print(f"Shape of simulated measurements y: {y_eye.shape}")

# plot
imagesc(y_eye[0, 0, :].reshape(h, h), r"$\tilde{x}$ in [0, 1]")

###############################################################################
# .. note::
#   Note that the image is identical to the original one, except it has been
#   normalized in [0,1].

# %%
# Same example with Poisson noise
# -----------------------------------------------------------------------------

###############################################################################
# We now consider Poisson noise, i.e., a noisy measurement vector given by
#
# .. math::
#       y \sim \mathcal{P}(\alpha H \tilde{x}),
#
# where :math:`\alpha` is a scalar value that represents the maximum image intensity
# (in photons). The larger :math:`\alpha`, the higher the signal-to-noise ratio.


###############################################################################
# We consider the :class:`spyrit.core.noise.Poisson` class and set :math:`\alpha`
# to 100 photons (which corresponds to the Poisson parameter).

from spyrit.core.noise import Poisson
from spyrit.misc.disp import add_colorbar, noaxis

alpha = 100  # number of photons
noise_op = Poisson(meas_op, alpha)

###############################################################################
# We simulate two noisy measurement vectors

y1 = noise_op(x)  # a noisy measurement vector
y2 = noise_op(x)  # another noisy measurement vector

###############################################################################
# We now consider the case :math:`\alpha = 1000` photons.

noise_op.alpha = 1000
y3 = noise_op(x)  # noisy measurement vector

###############################################################################
# We finally plot the measurement vectors as images

# plot
f, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].set_title("100 photons")
im = axs[0].imshow(y1[0, 0, :].reshape(h, h), cmap="gray")
add_colorbar(im, "bottom")

axs[1].set_title("100 photons")
im = axs[1].imshow(y2[0, 0, :].reshape(h, h), cmap="gray")
add_colorbar(im, "bottom")

axs[2].set_title("1000 photons")
im = axs[2].imshow(y3[0, 0, :].reshape(h, h), cmap="gray")
add_colorbar(im, "bottom")

noaxis(axs)
plt.show()

###############################################################################
# As expected the signal-to-noise ratio of the measurement vector is higher for
# 1,000 photons than for 100 photons
#
# .. note::
#   Not only the signal-to-noise, but also the scale of the measurements
#   depends on :math:`\alpha`, which motivates the introduction of the
#   preprocessing operator.

# %%
# The preprocessing operator
# -----------------------------------------------------------------------------

###############################################################################
# Preprocessing operators are defined in the :mod:`spyrit.core.prep` module.
# A preprocessing operator applies to the noisy measurements
#
# .. math::
#       m = \texttt{Prep}(y),
#
# For instance, a preprocessing operator can be used to compensate for the
# scaling factors that appear in the measurement or noise operators. In this
# case, a preprocessing operator is closely linked to its measurement and/or
# noise operator counterpart. While scaling factors are required to simulate
# realistic measurements, they are not required for reconstruction.

# %%
# Preprocessing measurements corrupted by Poisson noise
# -----------------------------------------------------------------------------

###############################################################################
# We consider the :class:`spyrit.core.prep.DirectPoisson` class that intends
# to "undo" the :class:`spyrit.core.noise.Poisson` class by compensating for:
#
# * the scaling that appears when computing Poisson-corrupted measurements
#
# * the affine transformation to get images in [0,1] from images in [-1,1]
#
# For this, it computes
#
# .. math::
#       m = \frac{2}{\alpha} y - H1
#

###############################################################################
# We consider the :class:`spyrit.core.prep.DirectPoisson` class and set :math:`\alpha`
# to 100 photons.

from spyrit.core.prep import DirectPoisson

alpha = 100  # number of photons
prep_op = DirectPoisson(alpha, meas_op)

###############################################################################
# We preprocess the first two noisy measurement vectors

m1 = prep_op(y1)
m2 = prep_op(y2)

###############################################################################
# We now consider the case :math:`\alpha = 1000` photons to preprocess the third
# measurement vector

prep_op.alpha = 1000
m3 = prep_op(y3)

###############################################################################
# We finally plot the preprocessed measurement vectors as images

# plot
f, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].set_title("100 photons")
im = axs[0].imshow(m1[0, 0, :].reshape(h, h), cmap="gray")
add_colorbar(im, "bottom")

axs[1].set_title("100 photons")
im = axs[1].imshow(m2[0, 0, :].reshape(h, h), cmap="gray")
add_colorbar(im, "bottom")

axs[2].set_title("1000 photons")
im = axs[2].imshow(m3[0, 0, :].reshape(h, h), cmap="gray")
add_colorbar(im, "bottom")

noaxis(axs)
plt.show()

###############################################################################
#
# .. note::
#   The preprocessed measurements still have different the signal-to-noise ratios
#   depending on :math:`\alpha`; however, they (approximately) all lie within
#   the same range (here, [-1, 1]).

###############################################################################
# We show again one of the preprocessed measurement vectors (tutorial thumbnail purpose)

# Plot
imagesc(m2[0, 0, :].reshape(h, h), "100 photons", title_fontsize=20)
