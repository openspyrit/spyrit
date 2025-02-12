r"""
01. Acquisition operators (basic)
====================================================
.. _tuto_acquisition_operators:

This tutorial shows how to simulate measurements using the :mod:`spyrit.core.meas` submodule.


.. image:: ../fig/tuto1.png
   :width: 600
   :align: center
   :alt: Reconstruction architecture sketch

|

All simulations are based on :class:`spyrit.core.meas.Linear` base class that simulates linear measurements

.. math::
    m = Hx,

where :math:`H\in\mathbb{R}^{M\times N}` is the acquisition matrix, :math:`x \in \mathbb{R}^N` is the signal of interest, :math:`M` is the number of measurements, and :math:`N` is the dimension of the signal.

.. important::
    The vector :math:`x \in \mathbb{R}^N` represents a multi-dimensional array (e.g, an image :math:`X \in \mathbb{R}^{N_1 \times N_2}` with :math:`N = N_1 \times N_2`). Both variables are related through vectorization , i.e., :math:`x = \texttt(vec)(X)`.

"""

# %%
# 1D Measurements 
# -----------------------------------------------------------------------------

###############################################################################
# We instantiate a measurement operator from a matrix of shape (10, 15).
import torch
from spyrit.core.meas import Linear

H = torch.randn(10, 15)
meas_op = Linear(H)

###############################################################################
# We consider 3 signals of length 15
x = torch.randn(3, 15)

###############################################################################
# We apply the operator to the batch of images, which produces 3 measurements 
# of length 10
y = meas_op(x)
print(y.shape)

###############################################################################
# We now plot the matrix-vector products

from spyrit.misc.disp import add_colorbar, noaxis
import matplotlib.pyplot as plt

f, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].set_title("Forward matrix H")
im = axs[0].imshow(H, cmap="gray")
add_colorbar(im, "bottom")

axs[1].set_title("Signals x")
im = axs[1].imshow(x.T, cmap="gray")
add_colorbar(im, "bottom")

axs[2].set_title("Measurements m")
im = axs[2].imshow(y.T, cmap="gray")
add_colorbar(im, "bottom")

noaxis(axs)

# %%
# 2D Measurements 
# -----------------------------------------------------------------------------

###############################################################################
# We load a batch of images from the :attr:`/images/` folder. Using the 
# :func:`transform_gray_norm` function with the :attr:`normalize=False` 
# argument returns images with values in (0,1).
import os
import torchvision
from spyrit.misc.statistics import transform_gray_norm

spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, "images/")

# Grayscale images of size (32, 32), no normalization to keep values in (0,1)
transform = transform_gray_norm(img_size=32, normalize=False)

# Create dataset and loader (expects class folder :attr:'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=7)

x, _ = next(iter(dataloader))

###############################################################################
# We crop the batch to get image of shape (15, 32) 
x = x[:,:,:15,:]
print(f"Shape of input images: {x.shape}")

###############################################################################
# We plot the second image
from spyrit.misc.disp import imagesc
imagesc(x[1, 0, :, :], r"image $X$")


###############################################################################
# We instantiate a measurement operator from a random matrix with shape (10, 15*32). To indicate that the operator works in 2D, we use the :attr:`meas_shape` argument.
H = torch.randn(10, 15*32)
meas_op = Linear(H, meas_shape=(15, 32))

###############################################################################
# We apply the operator to the batch of images, which produces a batch of measurement 
# vectors of length 10
y = meas_op(x)
print(y.shape)