# Changelog

\- removals  
/ changes  
\+ additions  

---
## Changes to come in a future version
<details open>
<summary>spyrit.core</summary>
*spyrit.core.meas  
\- self.H_adjoint for static classes  
\- self.get_H() for static classes  
</details>

<details open>
<summary>spyrit.misc</summary>
* .matrix_tools  
\- Permutation_Matrix()  
* .sampling  
\- meas2img2()  
</details>

<details>
<summary>spyrit.external</summary>
        .drunet  
            \+ include some documentation ?  
</details>

## v2.3.0
<details open>
<summary>spyrit.core</summary>
    / no longer supports numpy.array as input, must use torch.tensor  
  
    .meas  
        - class LinearRowSplit (use LinearSplit instead)  
        + 3 dynamic classes: DynamicLinear, DynamicLinearSplit, DynamicHadamSplit that allow measurements over time  
  
        .Linear  
            - self.get_H() deprecated (use self.H)  
            - self.H_adjoint (you might want to use self.H.T)  
            / constructor argument 'reg' renamed to 'rtol'  
            / self.H no longer refers to a torch.nn.Linear, but to a torch.tensor (not callable)  
            / self.H_pinv no longer refers to a torch.nn.Linear, but to a torch.tensor (not callable)  
            + self.__init__() has 'Ord' and 'meas_shape' optional arguments  
            + self.pinv() now supports lstsq image reconstruction if self.H_pinv is not defined  
            + self.set_H_pinv(), self.reindex() inherited from spyrit.misc.torch  
            + self.meas_shape, self.indices, self.Ord, self.H_static  
  
        .LinearSplit  
            / [includes changes from Linear]  
            / self.P no longer refers to a torch.nn.Linear, but to a torch.tensor (not callable)  
  
        .HadamSplit  
            / [includes changes from LinearSplit]  
                - self.__init__() does not need 'meas_shape' argument, it is taken as (h,h)  
  
            - self.Perm (use self.reindex() instead)  
  
    .noise  
        .NoNoise  
            + self.reindex() inherited from spyrit.core.meas.Linear.reindex()  
  
    .prep  
        - class SplitRowPoisson (was used with LinearRowSplit)  
  
    .recon  
        .PseudoInverse  
            / self.forward() now has **kwargs that are passed to meas_op.pinv(), useful for lstsq image reconstruction  
  
    + .torch  
        contains torch-specific functions that are commonly used in spyrit.core. Mirrors some spyrit.misc functions that are numpy-specific  
  
    +. warp  
        + class AffineDeformationField  
            warps an image using an affine transformation matrix  
        + class DeformationField  
            warps an image using a deformation field  
</details>

<details open>
<summary>spyrit.misc</summary>
    .matrix_tools  
        - Permutation_Matrix() is deprecated (already defined in spyrit.misc.sampling.Permutation_Matrix())  
  
    .sampling  
        - meas2img2() is deprecated (use meas2img() instead)  
        / meas2img() can now handle batch of images  
        + sort_by_significance(), reindex() to speed up permutation mattrix multiplication  
</details>

<details>
<summary>spyrit.external</summary>
    + class DRUNet, inheriting from UNetRes
</details>
