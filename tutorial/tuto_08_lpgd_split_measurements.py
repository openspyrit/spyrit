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