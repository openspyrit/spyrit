.. spyrit documentation master file, created by
   sphinx-quickstart on Fri Mar 12 11:04:59 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPyRiT: a single-pixel image reconstruction toolkit based on PyTorch
#####################################

SPyRiT is a `PyTorch <https://pytorch.org/>`_-based deep image reconstruction
package primarily designed for single-pixel imaging. SPyRiT is modular and may be useful for other linear inverse problems.


Single-pixel imaging
==================================

Single-pixel imaging aims to recover an unknown image :math:`x\in\Re^N` from a few noisy observations 

.. math::
    m \approx Hx,

where :math:`H\colon  \Re^{M\times N}` is a linear measurement operator, :math:`M` is the number of measurements and :math:`N` is the number of pixels in the image.

Simulation of the measurements
-----------------------------------
In practice, measurements are obtained by uploading a set of light patterns onto a spatial light modulator (e.g., a digital micromirror device (DMD), see \Fig{fig:principle}). Therefore, only positive patterns can be implemented. We model the actual acquisition process as 

.. math::
    y = \mathcal{N}(Ax), \label{eq:acquisition}

where :math:`\mathcal{N} \colon \Re^J \to \Re^J` represents a noise operator (e.g., Poisson or Poisson-Gaussian), :math:`A \in \Re_+^{J\times N}` is the actual acquisition operator that models the (positive) DMD patterns, and :math:`J` is the number of DMD patterns. 

Handling non negativity with pre-processing
-----------------------------------
We may preprocess the measurements before reconstruction to transform the actual model \Eq{eq:acquisition} into the target model \Eq{eq:acq-process}

.. math::
    m = By \approx Hx, \label{eq:prep}

where :math:`B\colon\Re^{J}\to \Re^{M}` is the preprocessing operator chosen such that :math:`BA=H$. Note that the noise of the preprocessed measurements :math:`m=By` is not the same as that of the actual measurements :math:`y$. 

Data-driven image reconstruction
-----------------------------------
Data-driven methods based on deep learning aim to find an estimate :math:`x^*\in \Re^N` of the unknown image :math:`x` from the preprocessed measurements :math:`By` (see \Eq{eq:prep}), using a reconstruction operator :math:`\mathcal{R}_{\theta^*} \colon \Re^M \to \Re^N$

.. math::
    \mathcal{R}_{\theta^*}(m) = x^* \approx x, 

where :math:`\theta^*` represents the parameters learned during a training procedure. 

Learning phase
-----------------------------------
In the case of supervised learning, it is assumed that a training dataset :math:`\{x^{(i)},y^{(i)}\}_{1 \le i \le I}` of :math:`I` pairs of ground truth images in :math:`\Re^N` and measurements in :math:`\Re^M` is available}. :math:`\theta^* :math:` is then obtained by solving 

.. math::
    \minimize{\theta}{\sum_{i =1}^I \mathcal{L}\left(x^{(i)},\mathcal{R}_\theta(By^{(i)})\right)},
    \label{eq:train}

where :math:`\mathcal{L}` is the training loss (e.g., squared error). In the case where only ground truth images :math:`\{x^{(i)}\}_{1 \le i \le I}` are available, the associated measurements are simulated as :math:`y^{(i)} = \op{N}(Ax^{(i)})$, :math:`1 \le i \le I$.


Reconstruction operator
-----------------------------------
A simple yet efficient method consists in correcting a traditional (e.g. linear) reconstruction by a data-driven nonlinear step 

.. math::
    \label{eq:recon_direct}
    \mathcal{R}_\theta = \mathcal{G}_\theta \circ \mathcal{R},  
    
where :math:`\mathcal{R}\colon\Re^{M}\to\Re^N` is a traditional hand-crafted (e.g., regularized) reconstruction operator and :math:`\mathcal{G}_\theta\colon\Re^{N}\to\Re^N` is a nonlinear neural network that acts in the image domain. 

Algorithm unfolding consists in defining :math:`\mathcal{R}_\theta` from an iterative scheme

.. math::
    \mathcal{R}_\theta = \mathcal{R}_{\theta_K} \circ \hdots \circ \mathcal{R}_{\theta_1},
    \label{eq:recon_iterative}

where :math:`\mathcal{R}_{\theta_k}` can be interpreted as the computation of the :math:`k$-th iteration of the iterative scheme and :math:`\theta = \bigcup_{k} \theta_k$.


SPyRiT's full network
==================================

.. image:: fig/full.png
   :width: 800
   :align: center

SPyRiT's full network

SPyRiT allows to simulate measurements and perform image reconstruction using
a full network. A full network is built using a measurement operator
:math:`\mathcal{P}`, a noise operator :math:`\mathcal{N}`, a preprocessing
operator :math:`\mathcal{B}`, a reconstruction operator :math:`\mathcal{R}`,
and a learnable neural network :math:`\mathcal{G}_{\theta}`. All operators
inherit from PyTorch's :class:`torch.nn.Module` class (`see here <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_),
which allows them to be easily combined into a full network.

Simulation of the measurements
-----------------------------------

We implement the acquisition as

      :math:`y = (\mathcal{N} \circ \mathcal{P})(x),`

where :math:`\mathcal{P}` is a linear operator that models the light patterns,
:math:`\mathcal{N}` is a noise operator, and :math:`\circ` denotes the composition.


Image reconstruction
-----------------------------------

Learning-based reconstruction approaches estimate the unknown image as
:math:`x^* = \mathcal{I}_\theta(y)`, where :math:`\theta`
represents the learnable parameters of the inversion model :math:`\mathcal{I}_\theta`.

A typical inversion operator :math:`\mathcal{I}_\theta` can be written as

      :math:`\mathcal{I}_\theta = \mathcal{G}_\theta \circ \mathcal{R} \circ \mathcal{B},`

where :math:`\mathcal{B}` is a preprocessing operator, :math:`\mathcal{R}` is
a linear reconstruction operator, and :math:`\mathcal{G}_\theta` is
a trainable neural network or any available image-domain denoiser.


Learning phase
-----------------------------------

In the case of supervised learning, the training phase solves

      :math:`\min_{\theta}{\sum_i \mathcal{L}\left(x^{(i)},\mathcal{I}_\theta(y^{(i)})\right)},`

where :math:`\mathcal{L}` is the training loss, and :math:`\{x^{(i)},y^{(i)}\}_i` is a set of training pairs.

By introducing the full network :math:`F_{\theta}(x) = (\mathcal{G}_\theta \circ \mathcal{R} \circ \mathcal{B} \circ \mathcal{N} \circ \mathcal{P})(x)`, the training phase relies on a database containing images only

      :math:`\min_{\theta}{\sum_i \mathcal{L}\left(x^{(i)},\mathcal{F}_\theta(x^{(i)})\right)}.`

The full network allows noisy data to be simulated on the fly, providing data augmentation while avoiding storing the measurements.

Getting started
==================================

Installation
-----------------------------------
The SPyRiT package is available for Linux, MacOs and Windows::

   pip install spyrit

Advanced installation guidelines are available on `GitHub <https://github.com/openspyrit/spyrit>`_.
Check out our `available tutorials <gallery/index.html>`_ to get started with SPyRiT.


Package structure
-----------------------------------

The main functionalities of SPyRiT are implemented in the subpackage
:class:`spyrit.core` , which contains 8 submodules:

1. **Measurement operators** (:mod:`spyrit.core.meas`) compute linear measurements :math:`\bar{y} = \mathcal{P}x`.

2. **Noise operators** (:mod:`spyrit.core.noise`) corrupt measurements :math:`y=\mathcal{N}(\bar{y})` with noise.

3. **Preprocessing operators** (:mod:`spyrit.core.prep`) are used to process noisy measurements, :math:`m=\mathcal{B}(y)` , before reconstruction. They typically compensate for the image normalization previously performed.

4. **Reconstruction operators** (:mod:`spyrit.core.recon`) comprise both standard linear reconstruction operators :math:`\mathcal{R}` and full network definitions :math:`\mathcal{F}_\theta`.

5. **Neural networks** (:mod:`spyrit.core.nnet`) include well-known neural networks :math:`\mathcal{G_{\theta}}`, generally used as denoiser layers.

6. **Training** (:mod:`spyrit.core.train`) provide the functionalities for training reconstruction networks.

7. **Warping** (:mod:`spyrit.core.warp`) contains the warping operators that are used to simulate moving objects.

8. **Torch utilities** (:mod:`spyrit.core.torch`) contains utility functions for PyTorch that are used throughout the package.


In addition, the subpackage :class:`spyrit.misc` contains various utility functions for Numpy / PyTorch that can be used independently of the core functionalities.

.. autosummary::
   :toctree: _autosummary
   :template: spyrit-module-template.rst
   :recursive:
   :caption: Contents

   spyrit.core
   spyrit.misc
   spyrit.external


Cite us
==================================
When using SPyRiT in scientific publications, please cite the following paper:

   - G. Beneti-Martin, L Mahieu-Williame, T Baudier, N Ducros, "OpenSpyrit: an Ecosystem for Reproducible Single-Pixel Hyperspectral Imaging," Optics Express, Vol. 31, No. 10, (2023). `DOI <https://doi.org/10.1364/OE.483937>`_.

When using SPyRiT specifically for the denoised completion network, please cite the following paper:

   - A Lorente Mur, P Leclerc, F Peyrin, and N Ducros, "Single-pixel image reconstruction from experimental data using neural networks," Opt. Express 29, 17097-17110 (2021). `DOI <https://doi.org/10.1364/OE.424228>`_.


Join the project
==================================
Feel free to contact us by `e-mail <mailto:nicolas.ducros@creatis.insa-lyon.fr>`_ for any question. Active developers are currently `Nicolas Ducros <https://www.creatis.insa-lyon.fr/~ducros/WebPage/index.html>`_, Thomas Baudier, `Juan Abascal <https://juanabascal78.wixsite.com/juan-abascal-webpage>`_ and Romain Phan.  Direct contributions via pull requests (PRs) are welcome.

The full list of contributors can be found `here <https://github.com/openspyrit/spyrit/blob/master/README.md#contributors-alphabetical-order>`_.


.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   gallery/index
