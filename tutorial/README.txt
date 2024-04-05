Tutorials
=========

Here you can find a series of Tutorials that will guide you through the use
of Spyrit. It is recommended to follow them in order.

For each tutorial, please download the corresponding Python Script (*.py*) or
Jupyter notebook (*.ipynb*) file at the bottom of the page. The images used in
these tutorials can be found on `this page`_ of the Spyrit GitHub.

Below is a diagram of the entire image processing pipeline. Each tutorial
focuseson a specific part of the pipeline.

* :ref:`Tutorial 1 <sphx_glr_gallery_tuto_01_acquisition_operators.py>` focuses
on the measurement operators, with or without noise

* :ref:`Tutorial 2 <sphx_glr_gallery_tuto_02_pseudoinverse_linear.py>` explains
the pseudo-inverse reconstruction process from the (possibly noisy)
measurements

* :ref:`Tutorial 3 <sphx_glr_gallery_tuto_03_pseudoinverse_cnn_linear.py>` uses
a CNN to denoise the image if necessary

* :ref:`Tutorial 4 <sphx_glr_gallery_tuto_04_train_pseudoinverse_cnn_linear.py>`
is used to train the CNN introduced in Tutorial 3

* :ref:`Tutorial 5 <sphx_glr_gallery_tuto_05_acquisition_split_measurements.py>`
introduces a new type of measurement operator ('split') that simulates positive
and negative measurements

* :ref:`Tutorial 6 <sphx_glr_gallery_tuto_06_dcnet_split_measurements.py>` uses
a Denoised Completion Network with a trainable image denoiser to improve the
results obtained in Tutorial 5

* Explore :ref:`Bonus Tutorial <sphx_glr_gallery_tuto_bonus_advanced_methods_colab.py>`
if you want to go deeper into Spyrit's capabilities


.. image:: ../fig/full.png
   :width: 800
   :align: center
   :alt: Principle of the image processing pipeline

|
|


.. _this page: https://github.com/openspyrit/spyrit/tree/3895b5e61fb6d522cff5e8b32a36da89b807b081/tutorial/images/test
