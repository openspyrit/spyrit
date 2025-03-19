r"""
01.b. Acquisition operators (splitting)
====================================================
.. _tuto_acquisition_operators_splitting:

This tutorial shows how to simulate linear measurements by splitting an acquisition matrix :math:`H\in \mathbb{R}^{M\times N}` that contains negative values.  It based on the :class:`spyrit.core.meas.LinearSplit` class of the :mod:`spyrit.core.meas` submodule.


.. image:: ../fig/tuto1.png
   :width: 600
   :align: center
   :alt: Reconstruction architecture sketch

|

In practice, only positive values can be implemented using a digital micromirror device (DMD). Therefore, we acquire

.. math::
    y =Ax,

where :math:`A \colon\, \mathbb{R}_+^{2M\times N}` is the acquisition matrix that contains positive DMD patterns, :math:`x \in \mathbb{R}^N` is the signal of interest, :math:`2M` is the number of DMD patterns, and :math:`N` is the dimension of the signal.

.. important::
    The vector :math:`x \in \mathbb{R}^N` represents a multi-dimensional array (e.g, an image :math:`X \in \mathbb{R}^{N_1 \times N_2}` with :math:`N = N_1 \times N_2`). Both variables are related through vectorization , i.e., :math:`x = \texttt{vec}(X)`.

Given a matrix :math:`H` with negative values, we define the positive DMD patterns :math:`A` from the positive and negative components :math:`H`. In practice, the even rows of :math:`A` contain the positive components of :math:`H`, while odd rows of :math:`A` contain the negative components of :math:`H`.

    .. math::
        \begin{cases}
            A[0::2, :] = H_{+}, \text{ with } H_{+} = \max(0,H),\\
            A[1::2, :] = H_{-}, \text{ with } H_{-} = \max(0,-H).
        \end{cases}

"""

# %%
# Splitting in 1D
# -----------------------------------------------------------------------------

###############################################################################
# We instantiate a measurement operator from a matrix of shape (10, 15).
import torch
from spyrit.core.meas import LinearSplit

H = torch.randn(10, 15)
meas_op = LinearSplit(H)

###############################################################################
# We consider 3 signals of length 15.
x = torch.randn(3, 15)

###############################################################################
# We apply the operator to the batch of images, which produces 3 measurements
# of length 10*2 = 20.
y = meas_op(x)
print(y.shape)

###############################################################################
# .. note::
#   The number of measurements is twice the number of rows of the matrix H that contains negative values.

# %%
# Illustration
# -----------------------------------------------------------------------------

###############################################################################
# We plot the positive and negative components of H that are concatenated in the matrix A.

A = meas_op.A
H_pos = meas_op.A[::2, :]  # Even rows
H_neg = meas_op.A[1::2, :]  # Odd rows

from spyrit.misc.disp import add_colorbar, noaxis
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

ax1.set_title("Forward matrix A")
im = ax1.imshow(A, cmap="gray")
add_colorbar(im)

ax2.set_title("Forward matrix H_pos")
im = ax2.imshow(H_pos, cmap="gray")
add_colorbar(im)

ax3.set_title("Measurements H_neg")
im = ax3.imshow(H_neg, cmap="gray")
add_colorbar(im)

noaxis(ax1)
noaxis(ax2)
noaxis(ax3)
# sphinx_gallery_thumbnail_number = 1

###############################################################################
# We can verify numerically that H = H_pos - H_neg

H = meas_op.H
diff = torch.linalg.norm(H - (H_pos - H_neg))

print(f"|| H - (H_pos - H_neg) || = {diff}")

###############################################################################
# We now plot the matrix-vector products between A and x.

f, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].set_title("Forward matrix A")
im = axs[0].imshow(A, cmap="gray")
add_colorbar(im, "bottom")

axs[1].set_title("Signals x")
im = axs[1].imshow(x.T, cmap="gray")
add_colorbar(im, "bottom")

axs[2].set_title("Split measurements y")
im = axs[2].imshow(y.T, cmap="gray")
add_colorbar(im, "bottom")

noaxis(axs)

# %%
# Simulations with noise and using the matrix H
# --------------------------------------------------------------------

######################################################################
# The operators in the :mod:`spyrit.core.meas` submodule allow for simulating noisy measurements
#
# .. math::
#   y =\mathcal{N}\left(Ax\right),
#
# where :math:`\mathcal{N} \colon\, \mathbb{R}^{2M} \to \mathbb{R}^{2M}` represents a noise operator (e.g., Gaussian). By default, no noise is applied to the measurement, i.e., :math:`\mathcal{N}` is the identity. We can consider noise by setting the :attr:`noise_model` attribute of the :class:`spyrit.core.meas.LinearSplit` class.

#####################################################################
# For instance, we can consider additive Gaussian noise with standard deviation 2.

from spyrit.core.noise import Gaussian

meas_op.noise_model = Gaussian(2)

#####################################################################
# .. note::
#   To learn more about noise models, please refer to :ref:`tutorial 2 <sphx_glr_gallery_tuto_02_noise.py>`.

#####################################################################
# We simulate the noisy measurement vectors
y_noise = meas_op(x)

#####################################################################
# Noiseless measurements can be simulated using the :meth:`spyrit.core.LinearSplit.measure` method.
y_nonoise = meas_op.measure(x)

#####################################################################
# The :meth:`spyrit.core.LinearSplit.measure_H` method simulates noiseless measurements using the matrix H, i.e., :math:`m = Hx`.
m_nonoise = meas_op.measure_H(x)

#####################################################################
# We now plot the noisy and noiseless measurements
f, axs = plt.subplots(1, 3, figsize=(8, 5))
axs[0].set_title("Split measurements y \n with noise")
im = axs[0].imshow(y_noise.mT, cmap="gray")
add_colorbar(im)

axs[1].set_title("Split measurements y \n without noise")
im = axs[1].imshow(y_nonoise.mT, cmap="gray")
add_colorbar(im)

axs[2].set_title("Measurements m \n without noise")
im = axs[2].imshow(m_nonoise.mT, cmap="gray")
add_colorbar(im)

noaxis(axs)
