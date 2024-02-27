Tutorials
=========

Here you can find a series of Tutorials that will guide you throughout the use
of Spyrit. It is recommended to run them in order. 

For each tutorial, please download the corresponding Python Script (*.py*) or
Jupyter notebook (*.ipynb*) file at the end of the page. You can find the
images used in these tutorials on `this page`_ of the Spyrit GitHub.

Here is a diagram of the whole image processing pipeline. Each tutorial focuses
on a specific part of the pipeline.

* :ref:`tuto_acquisition_operators` focuses on the measurement operators, with or without noise

* :ref:`tuto_pseudoinverse_linear` explains the pseudo-inverse reconstruction process from the (possibly noisy) measurements

* :ref:`tuto_pseudoinverse_cnn_linear` uses a CNN to de-noise the image if needed

* :ref:`tuto_train_pseudoinverse_cnn_linear` is used to train the CNN introduced in Tutorial 3

* :ref:`tuto_acquisition_split_measurements` introduces a new type of measurement operator ('Split') that simulates positive and negative measurements

* :ref:`tuto_dcnet_split_measurements` uses a Data Completion Network with a trainable image denoiser to improve the results obtained in Tutorial 5

* Explore :ref:`tuto_advanced_methods_colab` if you want to go deeper in understanding the capabilities of Spyrit


.. image:: ../fig/principle.png
   :width: 600
   :align: center
   :alt: Principle of the image processing pipeline



List of tutorials
-----------------



.. _this page: https://github.com/openspyrit/spyrit/tree/3895b5e61fb6d522cff5e8b32a36da89b807b081/tutorial/images/test