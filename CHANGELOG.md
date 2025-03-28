# Changelog

### Notations

\- removals
/ changes
\+ additions

---

<details>
<summary>

## Changes to come in a future version

</summary>


</details>

---

<details><summary>

## v2.3.4
</summary>

### spyrit.core
* #### General changes
    * The input and output shapes have been standardized across operators. All still images (i.e. not videos) have shape `(*, h, w)`, where `*` is any batch dimension (e.g. batch size and number of channels), and `h` and `w` are the height and width of the image. All measurements have shape `(*, M)`, where `*` is the same batch dimension than the images they come from. Videos have shape `(*, t, c, h, w)` where `t` is the time dimension, representing the number of frames in the video, `c` is the number of channels. Dynamic measurements from videos will thus have shape `(*, c, M)`.
    * The overall use of gpu has been improved. Every class of the `core` module now has a method `self.device` that allows to track the device on which its parameters are.
* #### spyrit.core.meas
    * / The regularization value 'L1' has been changed to 'rcond'. The behavior is unchanged but the reconstruction did not correspond to L1 regularization.
    * / Fixed .pinv() output shape (it was transposed with some regularisation methods)
    * / Fixed some device errors when using cuda with .pinv()
    * / The measurement matrix H is now stored with the data type it is given to the constructor (it was previously converted to torch.float32 for memory reasons)
    * \+ added in the .pinv() method a diff parameter enabling differentiated reconstructions (subtracting negative patterns/measurements to the positive patterns/measurements), only available for dynamic operators.
    * / For HadamSplit, the pinv has been overwritten to use a fast Walsh-Hadamard transform, zero-padding the measurements if necessary (in the case of subsampling). The inverse() method has been deprecated and will be removed in a future release.
* #### spyrit.core.recon
    * \- The class core.recon.Denoise_layer is deprecated and will be removed in a future version
    * / The class TikhonovMeasurementPriorDiag no longer uses Denoise_layer and uses instead an internal method to handle the denoising.
* #### spyrit.core.train
    * / load_net() uses the weights_only=True parameter in the torch.load() function. Documentation updated.
* #### spyrit.core.warp
    * / The warping operation (forward method) now has to be performed on (b,c,h,w) input tensors, and returns (b, time, c, h, w) output tensors.
    * / The AffineDeformationField does not store anymore the field as an attribute, but is rather generated on the fly. This allows for more efficient memory management.
    * / In AffineDeformationField the image size can be changed.
    * \+ It is now possible to use biquintic (5th-order) warping. This uses scikit-image's (skimage) warp function, which relies on numpy arrays.

### Tutorials
* Tutorial 2 integrated the change from 'L1' to 'rcond'
* All Tutorials have been updated to include the above mentioned changes.

</details>

---

<details open><summary>

## v2.3.3
</summary>

### spyrit.core
* #### spyrit.core.meas
    * / The regularization value 'L1' has been changed to 'rcond'. The behavior is unchanged but the reconstruction did not correspond to L1 regularization.
* #### spyrit.core.recon
    * / The documentation for the class core.recon.Denoise_layer has been clarified.

### Tutorials

* Tutorial 2 integrated the change from 'L1' to 'rcond'

</details>

---

<details open><summary>

## v2.3.2
</summary>

### spyrit.core
* #### spyrit.core.meas
    * / The method forward_H has been optimized for the HadamSplit class
* #### spyrit.core.torch
    * \+ Added spyrit.core.torch.fwht that implements in Pytorch the fast Walsh-Hadamard tranform for natural and Walsh ordered tranforms.
    * \+ Added spyrit.core.torch.fwht_2d that implements in Pytorch the fast Walsh-Hadamard tranform in 2 dimensions for natural and Walsh ordered tranforms.

### spyrit.misc

* #### spyrit.misc.statistics
    * / The function spyrit.misc.statistics.Cov2Var has been sped up and now supports an output shape for non-square images
* #### spyrit.misc.walsh_hadamard
    * / The function spyrit.misc.walsh_hadamard.fwht has been significantly sped up, especially for sequency-ordered walsh-hadamard tranforms.
    * \- fwht_torch is now deprecated. Use spyrit.core.torch.fwht instead.
    * \- walsh_torch is now deprecated. Use spyrit.core.torch.fwht instead.
    * \- walsh2_torch is now deprecated. Use spyrit.core.torch.fwht_2d instead.
* #### spyrit.misc.load_data
    * \+ New function download_girder that downloads files identified by their hexadecimal ID from a url server

### Tutorials

* Tutorials 3, 4, 6, 7, 8 now download data from our own servers instead of using google drive and the gdown library. Dependency on gdown library will be fully removed in a future version.

</details>

---

<details open><summary>

## v2.3.1
</summary>

### spyrit.core

* #### spyrit.core.meas
    * \+ For static classes, self.set_H_pinv has been renamed to self.build_H_pinv to match with the dynamic classes.
    * \+ The dynamic classes now support bicubic dynamic reconstruction (spyrit.core.meas.DynamicLinear.build_h_dyn()). This uses cubic B-splines.
* #### spyrit.core.train
    * load_net() must take the full path, **with** the extension name (xyz.pth).

### Tutorials

* Tutorial 6 has been changed accordingly to the modification of spyrit.core.train.load_net().
* Tutorial 8 is now available.

</details>

---

<details open><summary>

## v2.3.0
</summary>

<details open><summary>

### spyrit.core
</summary>

* / no longer supports numpy.array as input, must use torch.tensor
* #### spyrit.core.meas
    * \- class LinearRowSplit (use LinearSplit instead)
    * \+ 3 dynamic classes: DynamicLinear, DynamicLinearSplit, DynamicHadamSplit that allow measurements over time
    * spyrit.core.meas.Linear
        * \- self.get_H() deprecated (use self.H)
        * \- self.H_adjoint (you might want to use self.H.T)
        * / constructor argument 'reg' renamed to 'rtol'
        * / self.H no longer refers to a torch.nn.Linear, but to a torch.tensor (not callable)
        * / self.H_pinv no longer refers to a torch.nn.Linear, but to a torch.tensor (not callable)
        * \+ self.__init__() has 'Ord' and 'meas_shape' optional arguments
        * \+ self.pinv() now supports lstsq image reconstruction if self.H_pinv is not defined
        * \+ self.set_H_pinv(), self.reindex() inherited from spyrit.misc.torch
        * \+ self.meas_shape, self.indices, self.Ord, self.H_static
    * spyrit.core.meas.LinearSplit
        * / [includes changes from Linear]
        * / self.P no longer refers to a torch.nn.Linear, but to a torch.tensor (not callable)
    * spyrit.core.meas.HadamSplit
        * / [includes changes from LinearSplit]
            * \- self.__init__() does not need 'meas_shape' argument, it is taken as (h,h)
        * \- self.Perm (use self.reindex() instead)
* #### spyrit.core.noise
    * spyrit.core.noise.NoNoise
        * \+ self.reindex() inherited from spyrit.core.meas.Linear.reindex()
* #### spyrit.core.prep
    * \- class SplitRowPoisson (was used with LinearRowSplit)
* #### spyrit.core.recon
    * spyrit.core.recon.PseudoInverse
        * / self.forward() now has **kwargs that are passed to meas_op.pinv(), useful for lstsq image reconstruction
* #### \+ spyrit.core.torch
    contains torch-specific functions that are commonly used in spyrit.core. Mirrors some spyrit.misc functions that are numpy-specific
* #### \+ spyrit.core.warp
    * \+ class AffineDeformationField
        warps an image using an affine transformation matrix
    * \+ class DeformationField
        warps an image using a deformation field
</details>

<details open><summary>

### spyrit.misc
</summary>

* #### spyrit.misc.matrix_tools
    * \- Permutation_Matrix() is deprecated (already defined in spyrit.misc.sampling.Permutation_Matrix())
* #### spyrit.misc.sampling
    * \- meas2img2() is deprecated (use meas2img() instead)
    * / meas2img() can now handle batch of images
    * \+ sort_by_significance() & reindex() to speed up permutation mattrix multiplication
</details>
