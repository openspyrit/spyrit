:orphan:

Tutorials
=========

Here you can find a series of Tutorials that will guide you throughout the use
of Spyrit. It is recommended to run them in order.

For each tutorial, please download the corresponding Python Script (*.py*) or
Jupyter notebook (*.ipynb*) file at the end of the page. You can find the
images used in these tutorials on `this page`_ of the Spyrit GitHub.

Below is a diagram of the whole image processing pipeline. Each tutorial focuses
on a specific part of the pipeline.

* :ref:`Tutorial 1 <sphx_glr_gallery_tuto_01_acquisition_operators.py>` focuses on the measurement operators, with or without noise

* :ref:`Tutorial 2 <sphx_glr_gallery_tuto_02_pseudoinverse_linear.py>` explains the pseudo-inverse reconstruction process from the (possibly noisy) measurements

* :ref:`Tutorial 3 <sphx_glr_gallery_tuto_03_pseudoinverse_cnn_linear.py>` uses a CNN to de-noise the image if needed

* :ref:`Tutorial 4 <sphx_glr_gallery_tuto_04_train_pseudoinverse_cnn_linear.py>` is used to train the CNN introduced in Tutorial 3

* :ref:`Tutorial 5 <sphx_glr_gallery_tuto_05_acquisition_split_measurements.py>` introduces a new type of measurement operator ('Split') that simulates positive and negative measurements

* :ref:`Tutorial 6 <sphx_glr_gallery_tuto_06_dcnet_split_measurements.py>` uses a Data Completion Network (DCNet) with a trainable image denoiser to improve the results obtained in Tutorial 5

* :ref:`Tutorial 7 <sphx_glr_gallery_tuto_dcdrunet_split_measurements.py>` uses DCNeta with a pretrained plug-and-play DR-UNet denoiser which allows to add the noise level as an input parameter

* Explore :ref:`Bonus Tutorial <sphx_glr_gallery_tuto_bonus_advanced_methods_colab.py>` if you want to go deeper in understanding the capabilities of Spyrit


.. image:: ../fig/principle.png
   :width: 600
   :align: center
   :alt: Principle of the image processing pipeline

|
|


.. _this page: https://github.com/openspyrit/spyrit/tree/3895b5e61fb6d522cff5e8b32a36da89b807b081/tutorial/images/test


.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial shows how to simulate measurements using the spyrit.core submodule, which is base...">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_tuto_01_acquisition_operators_thumb.png
    :alt:

  :ref:`sphx_glr_gallery_tuto_01_acquisition_operators.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">01. Acquisition operators</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial shows how to simulate measurements and perform image reconstruction. The measurem...">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_tuto_02_pseudoinverse_linear_thumb.png
    :alt:

  :ref:`sphx_glr_gallery_tuto_02_pseudoinverse_linear.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">02. Pseudoinverse solution from linear measurements</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial shows how to simulate measurements and perform image reconstruction using PinvNet...">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_tuto_03_pseudoinverse_cnn_linear_thumb.png
    :alt:

  :ref:`sphx_glr_gallery_tuto_03_pseudoinverse_cnn_linear.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">03. Pseudoinverse solution + CNN denoising</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial shows how to train PinvNet with a CNN denoiser for reconstruction of linear measu...">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_tuto_04_train_pseudoinverse_cnn_linear_thumb.png
    :alt:

  :ref:`sphx_glr_gallery_tuto_04_train_pseudoinverse_cnn_linear.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">04. Train pseudoinverse solution + CNN denoising</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial is a continuation of the Acquisition operators tutorial &lt;tuto_acquisition_operato...">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_tuto_05_acquisition_split_measurements_thumb.png
    :alt:

  :ref:`sphx_glr_gallery_tuto_05_acquisition_split_measurements.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">05. Acquisition operators (advanced) - Split measurements and subsampling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial shows how to perform image reconstruction using DCNet (data completion network) w...">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_tuto_06_dcnet_split_measurements_thumb.png
    :alt:

  :ref:`sphx_glr_gallery_tuto_06_dcnet_split_measurements.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">06. DCNet solution for split measurements</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We refer to spyrit-examples/tutorial for a list of tutorials that can be run directly in colab ...">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_tuto_bonus_advanced_methods_colab_thumb.png
    :alt:

  :ref:`sphx_glr_gallery_tuto_bonus_advanced_methods_colab.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bonus. Advanced methods - Colab</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:

   /gallery/tuto_01_acquisition_operators
   /gallery/tuto_02_pseudoinverse_linear
   /gallery/tuto_03_pseudoinverse_cnn_linear
   /gallery/tuto_04_train_pseudoinverse_cnn_linear
   /gallery/tuto_05_acquisition_split_measurements
   /gallery/tuto_06_dcnet_split_measurements
   /gallery/tuto_bonus_advanced_methods_colab



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
