.. spyrit documentation master file, created by
   sphinx-quickstart on Fri Mar 12 11:04:59 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPyRiT's documentation
==================================

SPyRiT is a `PyTorch <https://pytorch.org/>`_-based package for deep image
reconstruction. While it is primarily designed for single-pixel image
reconstruction, it can solve other linear reconstruction problems.

SPyRiT allows to simulate measurements and perform image reconstruction using 
a full-network structure. It takes a normalized image as input and performs 
data simulation and image reconstruction in a single forward pass or in separate steps. 
A full-network generally comprises a measurement operator, a noise operator,
a preprocessing operator, a reconstruction operator, and a learnable neural network. 
All operators inherit from PyTorch `nn.Module` class, which allows to easily  
combine them into a full-network. 

.. image:: fig/principle.png
   :width: 700
   :align: center


Installation
==================================
The spyrit package is available for Linux, MacOs and Windows::

   pip install spyrit

Advanced installation guidelines are available on `GitHub <https://github.com/openspyrit/spyrit>`_.


Single-pixel imaging
==================================

Modelling of the measurements 
-----------------------------------

**Single-pixel imaging** aims to recover an image :math:`x\in\Re^N` from a few noisy scalar products :math:`y\in\Re^M`, where :math:`M\ll N`. We model the acquisition as

      :math:`y = (\mathcal{N} \circ \mathcal{P})(x),`

where :math:`\mathcal{P}` is a linear operator, :math:`\mathcal{N}` is a noise operator, and :math:`\circ` denotes the composition of operators. 

Image reconstruction
-----------------------------------

Learning-based reconstruction approaches estimate the unknown image as :math:`x^* = \mathcal{I}_\theta(y)`, 
where :math:`\mathcal{I}_\theta` represents the parameters that are learned during a training phase. 
In the case of supervised learning, **the training phase** solves 

      :math:`\min_{\theta}{\sum_i \mathcal{L}\left(x_i,\mathcal{I}_\theta(y_i)\right)},`

where :math:`\mathcal{L}` is the training loss between the true image :math:`x` and 
its estimation, and :math:`\{x_i,y_i\}_i` is a set of training pairs.

We consider the typical **reconstruction operator** :math:`\mathcal{I}_\theta` that can be written as follows:

      :math:`\mathcal{I}_\theta = \mathcal{G}_\theta \circ \mathcal{R} \circ \mathcal{B},`

where :math:`\mathcal{B}` is a preprocessing operator, :math:`\mathcal{R}` is a (standard) linear reconstruction operator, 
and :math:`\mathcal{G}_\theta` is a neural network that can be learnt during the training phase. 
Alternatively, :math:`\mathcal{R}` can be simply "plugged". In this case, its training is performed beforehand.

Introducing the **full network**, a forward pass can be written as follows:

      :math:`F_{\theta}(x) = (\mathcal{G}_\theta \circ \mathcal{R} \circ \mathcal{B} \circ \mathcal{N} \circ \mathcal{P})(x).`

The full network can be trained using a database that contains images only:

      :math:`\min_{\theta}{\sum_i \mathcal{L}\left(x_i,\mathcal{F}_\theta(x_i)\right)}.`

This pipeline allows to simulate noisy data on the fly, which provides data 
augmentation while avoiding storage of the measurements. 


Package structure 
-----------------------------------

The main functionalities of SPyRiT are implemented in the :class:`spyrit.core` subpackage, which contains six submodules:

1. **Measurement operators (meas)** compute linear measurements :math:`\mathcal{P}x` from
   images :math:`x`, where :math:`\mathcal{P}` is a linear operator (matrix) and :math:`x`
   is a vectorized image (see :mod:`spyrit.core.meas`).

2. **Noise operators (noise)** corrupt measurements :math:`y=(\mathcal{N}\circ\mathcal{P})(x)` with noise (see :mod:`spyrit.core.noise`).

3. **Preprocessing operators (prep)** are used to process noisy measurements, :math:`m=\mathcal{B}(y)` , 
   prior to reconstruction. They typically compensate for the image normalization previously performed (see :mod:`spyrit.core.prep`). 

4. **Reconstruction operators (recon)** comprise both standard linear reconstruction operators 
   :math:`\mathcal{R}` and full network definitions :math:`\mathcal{F}_\theta`, 
   which include both forward and reconstruction layers (see :mod:`spyrit.core.recon`).

5. **Neural networks (nnet)** include well-known neural networks :math:`\mathcal{G_{\theta}}`, generally used as denoiser layers (see :mod:`spyrit.core.nnet`).

6. **Training (train)** provide the functionalities for training reconstruction networks (see :mod:`spyrit.core.train`).

.. toctree::
   :maxdepth: 5
   :caption: Contents:

   api/modules
   gallery/index

.. Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

Cite us
==================================
When using SPyRiT in scientific publications, please cite the following paper:

   - G. Beneti-Martin, L Mahieu-Williame, T Baudier, N Ducros, "OpenSpyrit: an Ecosystem for Reproducible Single-Pixel Hyperspectral Imaging," Optics Express, Vol. 31, No. 10, (2023). `DOI <https://doi.org/10.1364/OE.483937>`_.

When using SPyRiT specifically for the denoised completion network, please cite the following paper:

   - A Lorente Mur, P Leclerc, F Peyrin, and N Ducros, "Single-pixel image reconstruction from experimental data using neural networks," Opt. Express 29, 17097-17110 (2021). `DOI <https://doi.org/10.1364/OE.424228>`_.

Join the project
==================================
Feel free to contact us by `e-mail <mailto:nicolas.ducros@creatis.insa-lyon.fr>` for any question. Active developers are currently `Nicolas Ducros <https://www.creatis.insa-lyon.fr/~ducros/WebPage/index.html>`_, Thomas Baudier  and `Juan Abascal <https://juanabascal78.wixsite.com/juan-abascal-webpage>`_.  Direct contributions via pull requests (PRs) are welcome.

The full list of contributors can be found `here <https://github.com/openspyrit/spyrit/blob/master/README.md#contributors-alphabetical-order>`_.
