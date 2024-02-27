Tutorials
=========

Here you can find a series of Tutorials that will guide you throughout the use
of Spyrit. It is recommended to run them in order. 

For each tutorial, please download the corresponding Python Script (*.py*) or
Jupyter notebook (*.ipynb*) file at the end of the page. You can find the
images used in these tutorials on `this page`_ of the Spyrit GitHub.

Here is a diagram of the whole image processing pipeline. Each tutorial focuses
on a specific part of the pipeline.

.. image:: ../fig/principle.png
   :width: 600
   :align: center
   :alt: Principle of the image processing pipeline

* Tutorial 1 focuses on the measurement operators, with or without noise

* Tutorial 2 explains the pseudo-inverse reconstruction process from the
(possibly noisy) measurements

* Tutorial 3 uses a CNN to de-noise the image if needed

* Tutorial 4 is used to train the CNN introduced in Tutorial 3

* Tutorial 5 introduces a new type of measurement operator ('Split') that
simulates positive and negative measurements

* Tutorial 6 uses a Data Completion Network with a trainable image denoiser to
improve the results obtained in Tutorial 5

* Explore Bonus Tutorial if you want to go deeper in understanding the
capabilities of Spyrit



.. _this page: https://github.com/openspyrit/spyrit/tree/3895b5e61fb6d522cff5e8b32a36da89b807b081/tutorial/images/test