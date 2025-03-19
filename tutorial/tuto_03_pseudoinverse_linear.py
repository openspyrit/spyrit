r"""
03. Pseudoinverse solution from linear measurements
===================================================
.. _tuto_pseudoinverse_linear:

This tutorial shows how to simulate measurements and perform image reconstruction using the :class:`spyrit.core.inverse.PseudoInverse` class of the :mod:`spyrit.core.inverse` submodule.

.. image:: ../fig/tuto3_pinv.png
   :width: 600
   :align: center
   :alt: Reconstruction architecture sketch

|
"""

# %%
# Loads images
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

# Grayscale images of size 32 x 32, no normalization to keep values in (0,1)
transform = transform_gray_norm(img_size=64, normalize=False)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=7)

x, _ = next(iter(dataloader))
print(f"Ground-truth images: {x.shape}")


# %%
# Linear measurements without noise
# -----------------------------------------------------------------------------

###############################################################################
# We consider a Hadamard matrix in "2D". The matrix has a shape of (64*64, 64*64)and values in {-1, 1}.
from spyrit.core.torch import walsh_matrix_2d

H = walsh_matrix_2d(64)

print(f"Acquisition matrix: {H.shape}", end=" ")
print(rf"with values in {{{H.min()}, {H.max()}}}")

###############################################################################
# We instantiate a :class:`spyrit.core.meas.Linear` operator. To indicate that the operator works in 2D, on images with shape (64, 64), we use the :attr:`meas_shape` argument.
from spyrit.core.meas import Linear

meas_op = Linear(H, (64, 64))

###############################################################################
# We simulate the measurement vectors, which have a shape of (7, 1, 4096).
y = meas_op(x)

print(f"Measurement vectors: {y.shape}")

###############################################################################
# We now compute the pseudo inverse solutions, which have a shape of (7, 1, 64, 64).
from spyrit.core.inverse import PseudoInverse

pinv = PseudoInverse(meas_op)
x_rec = pinv(y)

print(f"Reconstructed images: {x_rec.shape}")

###############################################################################
# We plot the reconstruction of the second image in the batch
from spyrit.misc.disp import imagesc, add_colorbar

imagesc(x_rec[1, 0])
# sphinx_gallery_thumbnail_number = 1

###############################################################################
# .. note::
#   The measurement operator is chosen as a Hadamard matrix with positive but this matrix can be replaced by any other matrix.

# %%
# LinearSplit measurements with Gaussian noise
# -----------------------------------------------------------------------------

###############################################################################
# We consider a linear operator where the positive and negative components are split, i.e. acquired separately. To do so, we instantiate a :class:`spyrit.core.meas.LinearSplit` operator.
from spyrit.core.meas import LinearSplit

meas_op = LinearSplit(H, (64, 64))

###############################################################################
# We consider additive Gaussian noise with standard deviation 2.
from spyrit.core.noise import Gaussian

meas_op.noise_model = Gaussian(2)

###############################################################################
# We simulate the measurement vectors, which have shape (7, 1, 8192).
y = meas_op(x)

print(f"Measurement vectors: {y.shape}")

###############################################################################
# We preprocess measurement vectors by computing the difference of the positive and negative components of the measurement vectors. To do so, we use the :class:`spyrit.core.prep.Unsplit` class. The preprocess measurements have a shape of (7, 1, 4096).

from spyrit.core.prep import Unsplit

prep = Unsplit()
m = prep(y)

print(f"Preprocessed measurement vectors: {m.shape}")

###############################################################################
# We now compute the pseudo inverse solutions, which have a shape of (7, 1, 64, 64).
from spyrit.core.inverse import PseudoInverse

pinv = PseudoInverse(meas_op)
x_rec = pinv(m)

print(f"Reconstructed images: {x_rec.shape}")

###############################################################################
# We plot the reconstruction
from spyrit.misc.disp import imagesc, add_colorbar

imagesc(x_rec[1, 0])

# %%
# HadamSplit2d with x4 subsampling with Poisson noise
# -----------------------------------------------------------------------------

###############################################################################
# We consider the acquisition of the 2D Hadamard transform of an image, where the positive and negative components of acquisition matrix are acquired separately. To do so, we use the dedicated :class:`spyrit.core.meas.HadamSplit2d` operator.  It also allows for subsampling the rows the Hadamard matrix, using a sampling map.

from spyrit.core.meas import HadamSplit2d

# Sampling map with ones in the top left corner and zeros elsewhere (low-frequency subsampling)
sampling_map = torch.ones((64, 64))
sampling_map[:, 64 // 2 :] = 0
sampling_map[64 // 2 :, :] = 0

# Linear operator with HadamSplit2d
meas_op = HadamSplit2d(64, 64**2 // 4, order=sampling_map, reshape_output=True)

###############################################################################
# We consider additive Poisson noise with an intensity of 100 photons.
from spyrit.core.noise import Poisson

meas_op.noise_model = Poisson(100)


###############################################################################
# We simulate the measurement vectors, which have a shape of (7, 1, 2048)

###############################################################################
# .. note::
#   The :class:`spyrit.core.noise.Poisson` class noise assumes that the images are in the range [0, 1]
y = meas_op(x)

print(rf"Reference images with values in {{{x.min()}, {x.max()}}}")
print(f"Measurement vectors: {y.shape}")

###############################################################################
# We preprocess measurement vectors by i) computing the difference of the positive and negative components, and ii) normalizing the intensity. To do so, we use the :class:`spyrit.core.prep.UnsplitRescale` class. The preprocessed measurements have a shape of (7, 1, 1024).

from spyrit.core.prep import UnsplitRescale

prep = UnsplitRescale(100)

m = prep(y)  # (y+ - y-)/alpha
print(f"Preprocessed measurement vectors: {m.shape}")

###############################################################################
# We compute the pseudo inverse solution, which has a shape of (7, 1, 64, 64).

x_rec = meas_op.fast_pinv(m)

print(f"Reconstructed images: {x_rec.shape}")

###############################################################################
# .. note::
#   There is no need to use the :class:`spyrit.core.inverse.PseudoInverse` class here, as the :class:`spyrit.core.meas.HadamSplit2d` class includes a method that returns the pseudo inverse solution.

###############################################################################
# We plot the reconstruction
from spyrit.misc.disp import imagesc, add_colorbar

imagesc(x_rec[1, 0])
