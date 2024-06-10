r"""
======================================================================
08. Learned proximal gradient descent (LPGD) for split measurements
======================================================================
.. _tuto_lpgd_split_measurements:

This tutorial shows how to perform image reconstruction with unrolled Learned Proximal Gradient
Descent (LPGD) for split measurements.

.. figure:: ../fig/lpgd.png
    :width: 600
    :align: center
    :alt: Sketch of the unrolled Learned Proximal Gradient Descent

"""

###############################################################################
# LPGD is a unrolled method, which can be explained as a recurrent network where
# each block corresponds to un unrolled iteration of the proximal gradient descent.
# At each iteration, the network performs a gradient step and a denoising step.
#
# The updated rule for the LPGD network is given by:
#
# .. math::
#     x^{(k+1)} = \mathcal{G}_{\theta}(x^{(k)} - \gamma H^T(H(x^{(k)}-m))).
#
# where :math:`x^{(k)}` is the image estimate at iteration :math:`k`,
# :math:`H` is the forward operator, :math:`\gamma` is the step size,
# and :math:`\mathcal{G}_{\theta}` is a denoising network with
# learnable parameters :math:`\theta`.

# sphinx_gallery_thumbnail_path = 'fig/lpgd.png'

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

from spyrit.misc.statistics import transform_gray_norm
import torchvision
import torch

h = 128  # image size hxh
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
# We consider noisy split measurements for a Hadamard operator and a simple
# rectangular subsampling” strategy
# (for more details, refer to :ref:`Acquisition - split measurements <tuto_acquisition_split_measurements>`).
#
#
# We define the measurement, noise and preprocessing operators and then
# simulate a measurement vector :math:`y` corrupted by Poisson noise. As in the previous tutorial,
# we simulate an accelerated acquisition by subsampling the measurement matrix
# by retaining only the first rows of a Hadamard matrix.

from spyrit.core.meas import HadamSplit
from spyrit.core.noise import Poisson
from spyrit.misc.sampling import meas2img2
from spyrit.misc.statistics import Cov2Var
from spyrit.core.prep import SplitPoisson

import math

# Measurement parameters
M = 4096  # Number of measurements (here, 1/4 of the pixels)
alpha = 10.0  # number of photons

# Sampling: rectangular matrix
Ord_rec = np.ones((h, h))
n_sub = math.ceil(M**0.5)
Ord_rec[:, n_sub:] = 0
Ord_rec[n_sub:, :] = 0

# Measurement and noise operators
meas_op = HadamSplit(M, h, Ord_rec)
noise_op = Poisson(meas_op, alpha)
prep_op = SplitPoisson(alpha, meas_op)

# Vectorize image
x = x.view(b * c, h * w)
print(f"Shape of vectorized image: {x.shape}")

# Measurements
y = noise_op(x)  # a noisy measurement vector
m = prep_op(y)  # preprocessed measurement vector

m_plot = m.detach().numpy()
m_plot = meas2img2(m_plot.T, Ord_rec)
imagesc(m_plot, r"Measurements $m$")