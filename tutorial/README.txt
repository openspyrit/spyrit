Tutorials
=========

This series of tutorials should guide you through the use of the SPyRiT pipeline.

.. figure:: ../fig/direct_net.png
   :width: 600
   :align: center
   :alt: SPyRiT pipeline

|

Each tutorial focuses on a specific submodule of the full pipeline.

* :ref:`Tutorial 1 <sphx_glr_gallery_tuto_01_a_acquisition_operators.py>`.a introduces the basics of measurement operators.

* :ref:`Tutorial 1 <sphx_glr_gallery_tuto_01_b_splitting.py>`.b introduces the splitting of measurement operators.

* :ref:`Tutorial 1 <sphx_glr_gallery_tuto_01_c_HadamSplit2d.py>`.c introduces the 2d Hadamard transform with subsampling.

* :ref:`Tutorial 2 <sphx_glr_gallery_tuto_02_noise.py>` introduces the noise operators.

* :ref:`Tutorial 3 <sphx_glr_gallery_tuto_03_pseudoinverse_linear.py>` demonstrates pseudo-inverse reconstructions from Hadamard measurements.

* :ref:`Tutorial 4 <sphx_glr_gallery_tuto_04_pseudoinverse_cnn_linear.py>`.a introduces data-driven post-processing reconstruction.

* :ref:`Tutorial 4 <sphx_glr_gallery_tuto_04_b_train_pseudoinverse_cnn_linear.py>`.b trains the post-processing CNN used in :ref:`Tutorial 4 <sphx_glr_gallery_tuto_04_pseudoinverse_cnn_linear.py>`.a.

* :ref:`Tutorial 5 <sphx_glr_gallery_tuto_05_dcnet.py>` introduces the denoised completion network for the reconstruction of Poisson-corrupted subsampled measurements.

* :ref:`Tutorial 6 <sphx_glr_gallery_tuto_06_a_warp.py>`.a demonstrates how to create and apply deformation fields to simulate motion from static images.

* :ref:`Tutorial 6 <sphx_glr_gallery_tuto_06_b_dynamic.py>`.b shows how acquire measurements of a moving scene and how to reconstruct a clean image of the scene through motion-compensation.


.. note::

  The Python script (*.py*) or Jupyter notebook (*.ipynb*) corresponding to each tutorial can be downloaded at the bottom of the page. The images used in these files can be found on `GitHub`_.

The tutorials below will gradually be updated to be compatible with SPyRiT 3 (work in progress, in the meantime see SPyRiT `2.4.0`_).


* :ref:`Tutorial 6 <sphx_glr_gallery_tuto_06_dcnet_split_measurements.py>` uses a Denoised Completion Network with a trainable image denoiser to improve the results obtained in Tutorial 5

* :ref:`Tutorial 7 <sphx_glr_gallery_tuto_07_drunet_split_measurements.py>` shows how to perform image reconstruction using a pretrained plug-and-play denoising network.

* :ref:`Tutorial 8 <sphx_glr_gallery_tuto_08_lpgd_split_measurements.py>` shows how to perform image reconstruction using a learnt proximal gradient descent.

.. _GitHub: https://github.com/openspyrit/spyrit/tree/3895b5e61fb6d522cff5e8b32a36da89b807b081/tutorial/images/test

.. _2.4.0: https://spyrit.readthedocs.io/en/2.4.0/gallery/index.html
