r"""
02. Pseudoinverse solution from linear measurements
===================================================
.. _tuto_pseudoinverse_linear:

This tutorial shows how to simulate measurements and perform image reconstruction.
The measurement operator is chosen as a Hadamard matrix with positive coefficients.
Note that this matrix can be replaced by any desired matrix.

These tutorials load image samples from `/images/`.
"""


# %%
# Load a batch of images
# -----------------------------------------------------------------------------

###############################################################################
# Images :math:`x` for training expect values in [-1,1]. The images are normalized
# using the :func:`transform_gray_norm` function.

import os

import torch
import torchvision
import numpy as np

from spyrit.misc.disp import imagesc
from spyrit.misc.statistics import transform_gray_norm

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
b, c, h, w = x.shape

# plot
x_plot = x.view(-1, h, h).cpu().numpy()
imagesc(x_plot[0, :, :], r"$x$ in [-1, 1]")

# %%
# Define a measurement operator
# -----------------------------------------------------------------------------
# .. _hadamard_positive:

###############################################################################
# We consider the case where the measurement matrix is the positive
# component of a Hadamard matrix, which is often used in single-pixel imaging.
# First, we compute a full Hadamard matrix that computes the 2D transform of an
# image of size :attr:`h` and takes its positive part.

from spyrit.misc.walsh_hadamard import walsh2_matrix

F = walsh2_matrix(h)
F = np.where(F > 0, F, 0)

###############################################################################
# .. _low_frequency:
#
# Next, we subsample the rows of the measurement matrix to simulate an
# accelerated acquisition. For this, we use the
# :func:`spyrit.misc.sampling.Permutation_Matrix` function
# that returns a :attr:`h*h`-by-:attr:`h*h` permutation matrix from a
# :attr:`h`-by-:attr:`h` sampling maps that indicates the location of the most
# relevant coefficients in the transformed domain.
#
# To keep the low-frequency Hadamard coefficients, we choose a sampling map
# with ones in the top left corner and zeros elsewhere.

import math

und = 4  # undersampling factor
M = h**2 // und  # number of measurements (undersampling factor = 4)

Sampling_map = np.ones((h, h))
M_xy = math.ceil(M**0.5)
Sampling_map[:, M_xy:] = 0
Sampling_map[M_xy:, :] = 0

imagesc(Sampling_map, "low-frequency sampling map")

###############################################################################
# After permutation of the full Hadamard matrix, we keep only its first
# :attr:`M` rows

from spyrit.misc.sampling import Permutation_Matrix

Perm = Permutation_Matrix(Sampling_map)
F = Perm @ F
H = F[:M, :]

print(f"Shape of the measurement matrix: {H.shape}")

###############################################################################
# Then, we instantiate a :class:`spyrit.core.meas.Linear` measurement operator

from spyrit.core.meas import Linear

meas_op = Linear(H, pinv=True)

# %%
# Noiseless case
# -----------------------------------------------------------------------------

###############################################################################
# In the noiseless case, we consider the :class:`spyrit.core.noise.NoNoise` noise
# operator

from spyrit.core.noise import NoNoise

noise = NoNoise(meas_op)

# Simulate measurements
y = noise(x.view(b * c, h * w))
print(f"Shape of raw measurements: {y.shape}")

###############################################################################
# To display the subsampled measurement vector as an image in the transformed
# domain, we use the :func:`spyrit.misc.sampling.meas2img` function

# plot
from spyrit.misc.sampling import meas2img

y_plot = y.detach().numpy().squeeze()
y_plot = meas2img(y_plot, Sampling_map)
print(f"Shape of the raw measurement image: {y_plot.shape}")

imagesc(y_plot, "Raw measurements (no noise)")


###############################################################################
# We now compute and plot the preprocessed measurements corresponding to an
# image in [-1,1]. For details in the preprocessing, see :ref:`Tutorial 1 <sphx_glr_gallery_tuto_01_acquisition_operators.py>`.
#
# .. note::
#
#       Using :class:`spyrit.core.prep.DirectPoisson` with :math:`\alpha = 1`
#       allows to compensate for the image normalisation achieved by
#       :class:`spyrit.core.noise.NoNoise`.

from spyrit.core.prep import DirectPoisson

prep = DirectPoisson(1.0, meas_op)  # "Undo" the NoNoise operator

m = prep(y)
print(f"Shape of the preprocessed measurements: {m.shape}")

# plot
m_plot = m.detach().numpy().squeeze()
m_plot = meas2img(m_plot, Sampling_map)
print(f"Shape of the preprocessed measurement image: {m_plot.shape}")

imagesc(m_plot, "Preprocessed measurements (no noise)")

# %%
# Pseudo inverse
# -----------------------------------------------------------------------------

###############################################################################
# We can use the :class:`spyrit.core.recon.PseudoInverse` class to perform the
# pseudo inverse reconstruction from the measurements :attr:`y`

from spyrit.core.recon import PseudoInverse

# Pseudo-inverse reconstruction operator
recon_op = PseudoInverse()

# Reconstruction
x_rec = recon_op(y, meas_op)

# plot
# Choose this plot to be the thumbnail
# sphinx_gallery_thumbnail_number = 5
x_plot = x_rec.squeeze().view(h, h).cpu().numpy()
imagesc(x_plot, "Pseudoinverse reconstruction (no noise)", title_fontsize=20)

# %%
# PinvNet Network
# -----------------------------------------------------------------------------

###############################################################################
# Alternatively, we can consider the :class:`spyrit.core.recon.PinvNet` class that reconstructs an
# image by computing the pseudoinverse solution, which is fed to a neural
# networker denoiser. To compute the pseudoinverse solution only, the denoiser
# can be set to the identity operator

###############################################################################
# .. image:: ../../docs/source/fig/pinvnet.png
#    :width: 400
#    :align: center
#    :alt: Sketch of the PinvNet architecture

from spyrit.core.recon import PinvNet

pinv_net = PinvNet(noise, prep, denoi=torch.nn.Identity())

###############################################################################
# or equivalently
pinv_net = PinvNet(noise, prep)

###############################################################################
# Then, we reconstruct the image from the measurement vector :attr:`y` using the
# :func:`~spyrit.core.recon.PinvNet.reconstruct` method

x_rec = pinv_net.reconstruct(y)

# plot
x_plot = x_rec.squeeze().cpu().numpy()
imagesc(x_plot, "PinvNet reconstruction (no noise)", title_fontsize=20)

###############################################################################
# Alternatively, the measurement vector can be simulated using the
# :func:`~spyrit.core.recon.PinvNet.acquire` method

y = pinv_net.acquire(x)
x_rec = pinv_net.reconstruct(y)

# plot
x_plot = x_rec.squeeze().cpu().numpy()
imagesc(x_plot, "Another pseudoinverse reconstruction (no noise)")

###############################################################################
# Note that the full module :attr:`pinv_net` both simulates noisy measurements
# and reconstruct them

x_rec = pinv_net(x)
print(f"Ground-truth image x: {x.shape}")
print(f"Reconstructed x_rec: {x_rec.shape}")

# plot
x_plot = x_rec.squeeze().cpu().numpy()
imagesc(x_plot, "One more pseudoinverse reconstruction (no noise)")

# %%
# Poisson-corrupted measurement
# -----------------------------------------------------------------------------

###############################################################################
# Here, we consider the :class:`spyrit.core.noise.Poisson` class
# together with a :class:`spyrit.core.prep.DirectPoisson`
# preprocessing operator (see :ref:`Tutorial 1 <sphx_glr_gallery_tuto_01_acquisition_operators.py>`).

alpha = 10  # maximum number of photons in the image

from spyrit.core.noise import Poisson
from spyrit.misc.disp import imagecomp

noise = Poisson(meas_op, alpha)
prep = DirectPoisson(alpha, meas_op)  # To undo the "Poisson" operator
pinv_net = PinvNet(noise, prep)

x_rec_1 = pinv_net(x)
x_rec_2 = pinv_net(x)
print(f"Ground-truth image x: {x.shape}")
print(f"Reconstructed x_rec: {x_rec.shape}")

# plot
x_plot_1 = x_rec_1.squeeze().cpu().numpy()
x_plot_1[:2, :2] = 0.0  # hide the top left "crazy pixel" that collects noise
x_plot_2 = x_rec_2.squeeze().cpu().numpy()
x_plot_2[:2, :2] = 0.0  # hide the top left "crazy pixel" that collects noise
imagecomp(x_plot_1, x_plot_2, "Pseudoinverse reconstruction", "Noise #1", "Noise #2")

###############################################################################
# As shown in the next tutorial, a denoising neural network can be trained to
# postprocess the pseudo inverse solution.