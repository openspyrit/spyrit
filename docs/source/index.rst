.. spyrit documentation master file, created by
   sphinx-quickstart on Fri Mar 12 11:04:59 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPyRiT
#####################################################################

SPyRiT is a `PyTorch <https://pytorch.org/>`_-based image reconstruction
package designed for `single-pixel imaging <single_pixel.html>`_. SPyRiT has a `modular organisation <organisation.html>`_ and may be useful for other inverse problems.

Github repository: `openspyrit/spyrit <https://github.com/openspyrit/spyrit>`_


Installation
==================================

SPyRiT is available for Linux, MacOs and Windows::

   pip install spyrit

See `here <https://github.com/openspyrit/spyrit>`_ for advanced installation guidelines.


Getting started
==================================

Please check our `tutorials <gallery/index.html>`_ as well as the  `examples <.. _examples: https://github.com/openspyrit/spyrit-examples/tree/master/2025_spyrit_v3>`_ on GitHub.

Cite us
==================================

When using SPyRiT in scientific publications, please cite [v3]_ for SPyRiT v3, [v2]_ for SPyRiT v2, and [v1]_ for DC-Net.

.. [v3] JFJP Abascal, T Baudier, R Phan, A Repetti, N Ducros, "SPyRiT 3.0: an open source package for single-pixel imaging based on deep learning," Vol. 33, Issue 13, pp. 27988-28005 (2025). `DOI <https://doi.org/10.1364/OE.559227>`_.
.. [v2] G Beneti-Martin, L Mahieu-Williame, T Baudier, N Ducros, "OpenSpyrit: an Ecosystem for Reproducible Single-Pixel Hyperspectral Imaging," *Optics Express*, Vol. 31, Issue 10, (2023). `DOI <https://doi.org/10.1364/OE.483937>`_.
.. [v1] A Lorente Mur, P Leclerc, F Peyrin, and N Ducros, "Single-pixel image reconstruction from experimental data using neural networks," *Opt. Express*, Vol. 29, Issue 11, 17097-17110 (2021). `DOI <https://doi.org/10.1364/OE.424228>`_.


Join the project
==================================

The list of contributors can be found `here <https://github.com/openspyrit/spyrit/blob/master/README.md#contributors-alphabetical-order>`_. Feel free to contact us by `e-mail <mailto:nicolas.ducros@creatis.insa-lyon.fr>`_ for any question. Direct contributions via pull requests (PRs) are welcome.

.. toctree::
   :maxdepth: 2
   :hidden:

   single_pixel
   organisation


.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   gallery/index


Contents
========

.. autosummary::
   :toctree: _autosummary
   :template: spyrit-module-template.rst
   :recursive:
   :caption: Contents

   spyrit.core
   spyrit.misc
   spyrit.external
