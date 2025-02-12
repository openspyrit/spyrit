r"""
02. Pseudoinverse solution from linear measurements
===================================================
.. _tuto_pseudoinverse_linear:

This tutorial shows how to simulate measurements and perform image reconstruction.
The measurement operator is chosen as a Hadamard matrix with positive coefficients.
Note that this matrix can be replaced by any desired matrix.

.. image:: ../fig/tuto2.png
   :width: 600
   :align: center
   :alt: Reconstruction architecture sketch

These tutorials load image samples from `/images/`.
"""

# %%
# Simulate measurements
# -----------------------------------------------------------------------------

###############################################################################
# We load a batch of images from the `/images/` folder. Using the 
# :func:`transform_gray_norm` function with the :attr:`normalize=False` 
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
# Linear without noise
# -----------------------------------------------------------------------------

###############################################################################
# Hadamard matrix in "2D" with shape (64*64, 64*64) with values in {-1, 1}
from spyrit.core.torch import walsh_matrix_2d

H = walsh_matrix_2d(64)

print(f"Acquisition matrix: {H.shape}", end=' ')
print(fr"with values in {{{H.min()}, {H.max()}}}")

###############################################################################
# Linear operator, working on images with shape (64, 64). To indicate that the operator works in 2D, we use the :attr:`meas_shape` argument.
from spyrit.core.meas import Linear

meas_op = Linear(H, (64,64))

###############################################################################
# We simulate the measurement vectors with shape (7, 1, 4096)
y = meas_op(x)

print(f"Measurement vectors: {y.shape}")

###############################################################################
# We now compute the pseudo inverse solution with shape (7, 1, 64, 64)
from spyrit.core.inverse import PseudoInverse

pinv = PseudoInverse(meas_op)
x_rec = pinv(y)

print(f"Reconstructed images: {x_rec.shape}")

###############################################################################
# We plot the reconstruction
from spyrit.misc.disp import imagesc, add_colorbar

imagesc(x_rec[1,0])

# %%
# LinearSplit and Gaussian noise
# -----------------------------------------------------------------------------

###############################################################################
# Linear operator where the positive and negative components are split, i.e. acquired separately.
from spyrit.core.meas import LinearSplit

meas_op = LinearSplit(H, (64,64))

###############################################################################
# We consider additive Gaussian noise with standard deviation 2.
from spyrit.core.noise import Gaussian

meas_op.noise_model = Gaussian(2)

###############################################################################
# We simulate the measurement vectors with shape (7, 1, 8192)
y = meas_op(x)

print(f"Measurement vectors: {y.shape}")

###############################################################################
# We preprocess measurement vectors with shape (7, 1, 4096) by computing the difference of the positive and negative components :math:`m = y_+ - y_-`. To do so, we use the :class:`spyrit.core.prep.Unsplit` class.

from spyrit.core.prep import Unsplit

prep = Unsplit()
m = prep(y)

print(f"Preprocessed measurement vectors: {m.shape}")

###############################################################################
# We now compute the pseudo inverse solution with shape (7, 1, 64, 64)
from spyrit.core.inverse import PseudoInverse

pinv = PseudoInverse(meas_op)
x_rec = pinv(m)

print(f"Reconstructed images: {x_rec.shape}")

###############################################################################
# We plot the reconstruction
from spyrit.misc.disp import imagesc, add_colorbar

imagesc(x_rec[1,0])

#%% 
# HadamSplit2d with x4 subsampling x4 and Poisson noise
# -----------------------------------------------------------------------------

###############################################################################
# Hadamard transform in 2D using the dedicated :class:`spyrit.core.prep.HadamSplit2d` operator. The operator is used to simulate the acquisition of the positive and negative components of the Hadamard transform of an image. It allows for subsampling the rows the Hadamard matrix using a sampling map.

from spyrit.core.meas import HadamSplit2d

# Sampling map with ones in the top left corner and zeros elsewhere (low-frequency subsampling)
sampling_map = torch.ones((64, 64))
sampling_map[:, 64 // 2 :] = 0
sampling_map[64 // 2 :, :] = 0

# Linear operator with HadamSplit2d
meas_op = HadamSplit2d(64, 64**2//4, order=sampling_map, reshape_output=True)

###############################################################################
# We consider additive Poisson noise with an intensity of 100 photons.
from spyrit.core.noise import Poisson

meas_op.noise_model = Poisson(100)


###############################################################################
# We simulate the measurement vectors with shape (7, 1, 8192)

# .. note::
#       The :class:`spyrit.core.prep.Poisson` class noise assumes that the images are in the range [0, 1] 
y = meas_op(x)

print(fr"Reference images with values in {{{x.min()}, {x.max()}}}")
print(f"Measurement vectors: {y.shape}")

###############################################################################
# We preprocess measurement vectors with shape (7, 1, 1024) by i)computing the difference of the positive and negative components, and ii)normalizing the intensity :math:`m = (y_+ - y_-)/\alpha`. To do so, we use the :class:`spyrit.core.prep.UnsplitRescale` class.

from spyrit.core.prep import UnsplitRescale
prep = UnsplitRescale(100)   

m = prep(y) # (y+ - y-)/alpha 
print(f"Preprocessed measurement vectors: {m.shape}")

###############################################################################
# We compute the pseudo inverse solution with shape (7, 1, 64, 64). There is no need to use the :class:`spyrit.core.inverse.PseudoInverse` class as the measurement operator has a pseudo-inverse already computed.

x_rec = meas_op.fast_pinv(m)  

print(f"Reconstructed images: {x_rec.shape}")

###############################################################################
# We plot the reconstruction
from spyrit.misc.disp import imagesc, add_colorbar

imagesc(x_rec[1,0])


    
if False:
    ###############################################################################
    # There are two ways to perform the pseudo inverse reconstruction from the
    # measurements :attr:`y`. The first consists of explicitly computing the
    # pseudo inverse of the measurement matrix :attr:`H` and applying it to the
    # measurements. The second computes a least-squares solution using :func:`torch.linalg.lstsq`
    # to compute the pseudo inverse solution.
    # The choice is made automatically: if the measurement operator has a pseudo-inverse
    # already computed, it is used; otherwise, the least-squares solution is used.
    #
    # .. note::
    #  Generally, the second method is preferred because it is faster and more
    #  numerically stable. However, if you will use the pseudo inverse multiple
    #  times, it becomes more efficient to compute it explicitly.
    #
    # First way: explicit computation of the pseudo inverse
    # We can use the :class:`spyrit.core.recon.PseudoInverse` class to perform the
    # pseudo inverse reconstruction from the measurements :attr:`y`.

    from spyrit.core.recon import PseudoInverse

    # Pseudo-inverse reconstruction operator
    recon_op = PseudoInverse()

    # Reconstruction
    x_rec1 = recon_op(m, meas_op)  # equivalent to: meas_op.pinv(y)
    print("Shape of the explicit pseudo-inverse reconstructed image:", x_rec1.shape)

    ###############################################################################
    # Second way: calling pinv method from the Linear operator
    # The code is very similar to the previous case, but we need to make sure the
    # measurement operator has no pseudo-inverse computed. We can also specify
    # regularization parameters for the least-squares solution when calling
    # `recon_op`. In our case, the pseudo-inverse was computed at initialization
    # of the meas_op object.

    print(f"Pseudo-inverse computed: {hasattr(meas_op, 'H_pinv')}")
    temp = meas_op.H_pinv  # save the pseudo-inverse
    del meas_op.H_pinv  # delete the pseudo-inverse
    print(f"Pseudo-inverse computed: {hasattr(meas_op, 'H_pinv')}")

    # Reconstruction
    x_rec2 = recon_op(m, meas_op, reg="rcond", eta=1e-6)
    print("Shape of the least-squares reconstructed image:", x_rec2.shape)

    # restore the pseudo-inverse
    meas_op.H_pinv = temp

    ##############################################################################
    # .. note::
    #   This choice is also offered for dynamic measurement operators which are
    #   explained in :ref:`Tutorial 9 <sphx_glr_gallery_tuto_09_dynamic.py>`.

    # plot side by side
    import matplotlib.pyplot as plt
    from spyrit.misc.disp import add_colorbar

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    im1 = ax1.imshow(x_rec1[0, 0, :, :], cmap="gray")
    ax1.set_title("Explicit pseudo-inverse reconstruction")
    add_colorbar(im1, "right", size="20%")

    im2 = ax2.imshow(x_rec2[0, 0, :, :], cmap="gray")
    ax2.set_title("Least-squares pseudo-inverse reconstruction")
    add_colorbar(im2, "right", size="20%")

