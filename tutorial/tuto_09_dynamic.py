r"""
09. Acquisition and reconstruction in dynamic conditions
========================================================
.. _tuto_dynamic:

This tutorial explains how to simulate dynamic measurement and reconstruction
of a moving object. There are three steps in this process:

1. First, a still image is deformed to generate multiple frames. This step
simulates movement of the object. The module :mod:`spyrit.core.warp` is used
to warp images.

2. Second, the measurement is performed on the series of frames. The 'Dynamic'
classes from :mod:`spyrit.core.meas` are used.

3. Third, the reconstruction from pesudo-inverse matrices is used to reconstruct
the motion-compensated image.

This tutorial will present an example in which all three steps will be
explained in an example. To understand the specificities of the module
:mod:`spyrit.core.warp`, a more detailed explanation is included at the end
of the example.

.. image:: ../fig/tuto9.png
   :width: 600
   :align: center
   :alt: Dynamic measurement and reconstruction steps
"""

# %%
# 1. Example: warping an image to generate a motion picture
# *****************************************************************************

###############################################################################
# This tutorial loads example images from the relative folder `/images/`.

# %%
# 1.a Load an image from a batch of images
# -----------------------------------------------------------------------------
# This part is identical to other tutorials. We consider an image of size
# 32x32 pixels. 

import os

import math
import torch
import torchvision

from spyrit.misc.disp import imagesc
from spyrit.misc.statistics import transform_gray_norm

# sphinx_gallery_thumbnail_path = 'fig/tuto9.png'

img_size = 32  # full image side's size in pixels
meas_size = 32  # measurement pattern side's size in pixels (Hadamard matrix)
img_shape = (img_size, img_size)
meas_shape = (meas_size, meas_size)
i = 1  # Image index (modify to change the image)
spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, "images/")


# Create a transform for natural images to normalized grayscale image tensors
transform = transform_gray_norm(img_size=img_size)

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
x_plot = x.view(img_shape).cpu()
imagesc(x_plot, r"Original image $x$ in [-1, 1]")

# %%
# 1.b Define an affine transformation
# -----------------------------------------------------------------------------
# Here we will define an affine transformation using a matrix and the class
# :class:`spyrit.core.warp.AffineDeformationField`.
#
# This class takes 3 arguments:
# a function :math:`f(t) = Mat`, where :math:`t` represents the time
# at which :math:`f` is evaluated, and :math:`Mat` is the `(3,3)` matrix that
# represents the affine transformation. For more details about these matrices,
# check out `this wiki page <https://en.wikipedia.org/wiki/Affine_transformation#Image_transformation>`_.
# The two other arguments are a list of times :math:`(t_0, ... , t_n)` at
# which to evaluate the fuction :math:`f`, and the image size (used to
# determine the grid size) :math:`(height, width)`.
#
# Let's first see th construction of the function :math:`f`.

from spyrit.core.warp import AffineDeformationField

# we want to define a deformation similar to that see in [ref to Thomas].

a = 0.2  # amplitude
omega = math.pi  # angular speed

def s(t):
    return 1 + a * math.sin(t * omega)  # base function for f

def f(t):
    return torch.tensor(
        [
            [1 / s(t), 0, 0],
            [0, s(t), 0],
            [0, 0, 0],
        ],
        dtype=torch.float64,
    )

###############################################################################
# .. note::
#       It is recommended when building the function :math:`f` to have its
#       output dtype set to `float64`, so that computations are as accurate
#       as possible. This is especially true for large images.
#
# Next, we will create the time vector and define the image shape.
#
# The measurement size (the size of the Hadamard patterns applied to the image)
# determines the number of measurements - if there is no subsampling. The
# number of patterns must match the number of frames of the motion picture. It
# is for this reason that the number of frames is set to the square of the
# measurement size.

time_vector = torch.linspace(0, 10, (meas_size**2) *2)  # *2 because of the splitting

aff_field = AffineDeformationField(f, time_vector, img_shape)

# %%
# 1.c Warp the image
# -----------------------------------------------------------------------------
# Now that the field is defined, we can warp the image. Spyrit works mostly
# with vectorized images, and warping images is no exception. Currently, the
# classes :class:`spyrit.core.warp.AffineDeformationField` and
# :class:`spyrit.core.warp.DeformationField` can only warp a single image at a
# time.

import matplotlib.pyplot as plt
from spyrit.misc.disp import add_colorbar

# Reshape the image from (b,c,h,w) to (c, h*w)
x = x.view(c, h * w)

x_motion = aff_field(x, 0, (meas_size**2) *2)
c, n_frames, n_pixels = x_motion.shape

# show random frames
frames = [100, 300]

plot, axes = plt.subplots(1, len(frames), figsize=(10, 5))

for i, f in enumerate(frames):
    im = axes[i].imshow(x_motion[0, f, :].view(img_shape).cpu().numpy(), cmap="gray")
    axes[i].set_title(f"Frame {f}")
    add_colorbar(im, "right", size="20%")
plot.tight_layout()
plt.show()


# %%
# 2. Example: measuring the moving object
# *****************************************************************************
# In this section, we will simulate the measurement of the moving object. We
# will use a full Hadamard matrix (no subsampling) to measure the object. The
# best option when using a Hadamard matrix is to use the class
# :class:`spyrit.core.meas.DynamicHadamSplit`. For the moment, no noise can be
# applied to the dynamic measurement operators. This will be available in a
# future release. As a consequence, the preprocessing operators are also
# unavailable.

# %%
# 2.a Define the measurement operator
# -----------------------------------------------------------------------------
# The class :class:`spyrit.core.meas.DynamicHadamSplit` is the mirror class of
# :class:`spyrit.core.meas.HadamardSplit`. The difference is that the dynamic
# will apply a different measurement pattern to each frame. There must
# therefore be as many patterns as there are frames.
#
# If you have too many frames in your motion picture, you may want to use only
# the first frames of the motion picture. If you have too many patterns, you
# may want to set the number of patterns to the number of frames using the
# parameter `M`.

from spyrit.core.meas import DynamicHadamSplit

meas_op = DynamicHadamSplit(M=meas_size**2, h=meas_size, Ord=None, img_shape=img_shape)

# show the measurement matrix H
print("Shape of the measurement matrix H:", meas_op.H_static.shape)
# as we are using split measurements, it is the matrix P that is effectively
# used when computing the measurements
print("Shape of the measurement matrix P:", meas_op.P.shape)

# %%
# 2.b Measure the moving object
# -----------------------------------------------------------------------------
# Now that the measurement operator is defined, we can measure the moving
# object. As with the static case, this is done by using the implicit forward
# method.

# measure the moving object
y = meas_op(x_motion)

# show the measurement vector
imagesc(y.view((meas_size * 2, meas_size)).cpu().numpy(), "Measurement vector")


# %%
# 3. Example: reconstructing the motion-compensated image
# *****************************************************************************
# In this section, we will reconstruct the motion-compensated image from the measurements.
# This is done by combining the information contained in the measurement
# patterns and in the deformation field. This theoretical work has been
# explained in [1]_ and [2]_. The reconstruction follows the physical
# discretization of the problem, thus avoiding to warp the Hadamard patterns.
# The class :class:`spyrit.core.meas.DynamicHadamSplit` (and the other dynamic classes)
# handle the dynamic reconstruction through various methods.

# %%
# 3.a Compute the dynamic measurement matrix
# -----------------------------------------------------------------------------
# The dynamic measurement matrix :math:`H_{dyn}` is defined as the measurement
# matrix that would give the same measurement vector :math:`y` as the one
# computed before when applied to a still image :math:`x_{ref}`:
#
# .. math::
#     y = H_{dyn} x_{ref}
# 
# Or, following the notations from [2]_, :math:`m = H_{dyn} f_{ref}`. 
# 
# To build the # dynamic measurement matrix, we need the measurement patterns and the
# deformation field. In this case, the deformation field is known, but in some
# cases it might have to be estimated.
#
# The dynamic measurement matrix `H_dyn` is built using the measurement
# operator itself.

# compute the dynamic measurement matrix
print("H_dyn computed:", hasattr(meas_op, "H_dyn"))
meas_op.build_H_dyn(aff_field, mode="bilinear")
print("H_dyn computed:", hasattr(meas_op, "H_dyn"))

###############################################################################
# .. important::
#   Because :math:`P` is the actual matrix used for measuring, the attribute
#   :attr:`H_dyn` is computed using the matrix :math:`P`. This can be seen in
#   their shapes, which are the transpose of each other.

# recommended way
print("H_dyn shape:", meas_op.H_dyn.shape)
print("P shape:", meas_op.P.shape)
# NOT recommended, can cause confusions
print("H shape:", meas_op.H.shape)
print("H_dyn is same as H:", (meas_op.H == meas_op.H_dyn).all())

# show P and H_dyn side by side

plot, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

im1 = ax1.imshow(meas_op.P.cpu().numpy(), vmin=0, vmax=1.5, cmap="gray")
ax1.set_title("Measurement matrix P")
add_colorbar(im1, "right", size="20%")

im2 = ax2.imshow(meas_op.H_dyn.cpu().numpy(), vmin=0, vmax=1.5, cmap="gray")
ax2.set_title("Dynamic measurement matrix H_dyn")
add_colorbar(im2, "right", size="20%")

plot.tight_layout()
plt.show()

###############################################################################
# This method adds to the measurement operator a new attribute named
# :attr:`H_dyn`. It can also be accessed using the attribute name :attr:`H` for
# compatibility reasons, although it is NOT recommended.

# %%
# 3.b Reconstruct the motion-compensated image
# -----------------------------------------------------------------------------
# Now that the dynamic measurement matrix has been computed, we can reconstruct
# the motion-compensated image. To do this, we can first compute the pseudo-inverse
# of our dynamic measurement matrix:

# compute the pseudo-inverse using the requested regularizers
print("H_dyn_pinv computed:", hasattr(meas_op, "H_dyn_pinv"))
meas_op.build_H_dyn_pinv(reg="L2", eta=1e-6)
print("H_dyn_pinv computed:", hasattr(meas_op, "H_dyn_pinv"))

# recommended way
print("H_dyn_pinv shape:", meas_op.H_dyn_pinv.shape)
# NOT recommended, can cause confusions
print("H_pinv shape", meas_op.H_pinv.shape)
print("H_dyn_pinv is same as H_pinv:", (meas_op.H_dyn_pinv == meas_op.H_pinv).all())

###############################################################################
# This creates a new attribute :attr:`H_dyn_pinv` in which is stored the
# pseudo-inverse of :attr:`H_dyn`. As before, the same tensor can be accessed
# through the attribute :attr:`H_pinv` for compatibility reasons, although it
# is not recommended.
#
# Once the pseudo-inverse has been computed, we can simply call the method
# :meth:`pinv` associated with some measurements to reconstruct the motion-compensated
# image. As with the static case, this can also be done through the class
# :class:`spyrit.core.recon.PseudoInverse`

# using self.pinv directly
x_hat1 = meas_op.pinv(y)
print("x_hat1 shape:", x_hat1.shape)

# using a PseudoInverse instance, no difference
from spyrit.core.recon import PseudoInverse
recon_op = PseudoInverse()
x_hat2 = recon_op(y, meas_op)

print("x_hat1 and x_hat2 are equal:", (x_hat1 == x_hat2).all())

# show the motion-compensated image and the difference with the original image

plot, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

im1 = ax1.imshow(x_hat1.view(img_shape), cmap="gray")
ax1.set_title("Motion-compensated image,\nusing pinv")
add_colorbar(im1, "right", size="20%")

im2 = ax2.imshow(x_plot.view(img_shape) - x_hat1.view(img_shape), cmap="gray")
ax2.set_title("Difference between original\nand motion-compensated image")
add_colorbar(im2, "right", size="20%")

# plot.tight_layout()
plt.show()

# imagesc(x_hat1.view(img_shape), "Motion-compensated image, using pinv")
# imagesc(x_plot.view(img_shape) - x_hat1.view(img_shape),
#     "Difference between original\nand motion-compensated image",
# )

###############################################################################
# .. important::
#   As with static reconstruction, it is possible to reconstruct the
#   motion-compensated image without having to compute the pseudo-inverse 
#   explicitly. Calling the method :meth:`pinv` while the attribute
#   :attr:`H_dyn_pinv` is not defined will result in using the least-squares
#   function provided in torch: :func:`torch.linalg.lstsq`. 


# %%
# 4. Warping detailed explanation
# *****************************************************************************
# This tutorial uses the class :class:`spyrit.core.warp.AffineDeformationField`
# to simulate the movement of a still image. This class is a subclass of
# :class:`spyrit.core.warp.DeformationField`, which can be used to deform an
# image in a more general manner. This is particularly useful for experimental
# setups where the deformation is estimated from real measurements.
#
# Here, we provide an example of how to use the class
# :class:`spyrit.core.warp.DeformationField`. The class takes one argument:
# the deformation field itself of shape :math:`(n_frames,h,w,2)`, where
# :math:`n_frames` is the number of frames, and :math:`h` and :math:`w` are the
# height and width of the image. The last dimension represents the 2D
# pixel from where to interpolate the new pixel value at the coordinate
# :math:`(h,w)`.
#
# We will first use an instance of :class:`spyrit.core.warp.AffineDeformationField`
# to create the deformation field. Then, a separate instance of
# :class:`spyrit.core.warp.DeformationField` will be created using the
# deformation field from the affine deformation field.

from spyrit.core.warp import DeformationField

# define a rotation function
omega = 2 * math.pi  # angular velocity

def rot(t):
    ans = torch.tensor([
            [math.cos(t * omega), -math.sin(t * omega), 0],
            [math.sin(t * omega), math.cos(t * omega), 0],
            [0, 0, 1],
        ], dtype=torch.float64)  # it is recommended to use float64
    return ans

# create a time vector of length 100 (change this to fit your needs)
t0 = 0
t1 = 10
n_frames = 100
time_vector = torch.linspace(t0, t1, n_frames)
img_shape = (50, 50)  # image shape

# create the affine deformation field
aff_field2 = AffineDeformationField(rot, time_vector, img_shape)

###############################################################################
# Now that the affine deformation field is created, we can access the
# deformation field through the attribute :attr:`field`. Its value can then be
# used to create a new instance of :class:`spyrit.core.warp.DeformationField`.

# get the deformation field
field = aff_field2.field
print("field shape:", field.shape)

# create an instance of DeformationField
def_field = DeformationField(field)

# def_field and aff_field2 are the same
print("def_field and aff_field2 are the same:", (def_field == aff_field2))

###############################################################################
# .. rubric:: References for dynamic reconstruction
#
# .. [1] Thomas Maitre, Elie Bretin, L. Mahieu-Williame, Michaël Sdika, Nicolas Ducros. Hybrid single-pixel camera for dynamic hyperspectral imaging. 2023. hal-04310110

# .. [2] (MICCAI 2024 paper #883) Thomas Maitre, Elie Bretin, Romain Phan, Nicolas Ducros, Michaël Sdika. Dynamic Single-Pixel Imaging on an Extended Field of View without Warping the Patterns. 2024. hal-04533981