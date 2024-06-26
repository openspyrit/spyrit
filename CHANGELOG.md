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

* spyrit.core
    * spyrit.core.meas
        * \- self.H_adjoint for static classes
        * \- self.get_H() for static classes
* spyrit.misc
    * spyrit.misc.matrix_tools
        * \- Permutation_Matrix()
    * spyrit.misc.sampling
        * \- meas2img2()

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
