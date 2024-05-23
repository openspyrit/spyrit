# TODO
# - remove optional parameter from DeformationField __init__
# - add mandatory t0 t1, fps and n_frames to AffineDeformationField __init__


"""
Stores deformation fields and warps images.

Contains :class:`DeformationField` and :class:`AffineDeformationField`, a
subclass of the former. These classes are used to warp images according to
a deformation field that is stored as as class attribute. They can be fed
an image (called "*original image*") and will return the warped image
("*deformed image*").

The function that maps the *original image* pixel coordinates to the *deformed
image* pixel coordinates is called the "*deformation field*" and is noted
:math:`v`. The function that maps the pixels of the *deformed image* to the
pixels of the *original image* is called the "*inverse deformation field*" and
is noted :math:`u`. The *deformation field* and the *inverse deformation field*
are related by the equation :math:`v = u^{-1}`.

Here, the two classes use and store the *inverse deformation field* :math:`u`
as a class attribute.
"""

import warnings

import torch
import torch.nn as nn


# =============================================================================
class DeformationField(nn.Module):
    # =========================================================================
    r"""
    Stores a discrete deformation field as a :math:`(b,h,w,2)` tensor.

    Warps a single image according to an *inverse deformation field* :math:`u`,
    i.e. the field that maps the *deformed image* pixel coordinates to the
    *original image* pixel coordinates.

    It is constructed from a tensor of shape :math:`(n\_frames,h,w,2)`, where
    :math:`n\_frames` is the number of frames in the animation, :math:`h` and
    :math:`w` are the number of pixels along the height and width of the image respectively.
    The last dimension contains the x and y coordinates of the original image
    pixel that is displayed in the warped image.

    .. important::
        The coordinates are given in the range [-1;1]. When referring to a
        pixel, its position is the position of its center. The position
        [-1;-1] corresponds to the center of the top-left pixel.

    Args:
        :attr:`inverse_grid_frames` (torch.tensor):
        *Inverse deformation field* :math:`u` of shape :math:`(n\_frames,H,W,2)`,
        where :math:`n\_frames` is the number of frames in the animation, and
        :math:`H` and :math:`W` are the height and width of the image to be
        warped. For accuracy reasons, the dtype is converted to `torch.float64`.

    Attributes:
        :attr:`self.inverse_grid_frames` (torch.tensor):
        *Inverse deformation field* :math:`u` of shape :math:`(n\_frames,h,w,2)`.
        If set manually, the dtype should be `torch.float64`. Default: `None`.

    **Example 1:** Rotating a 2x2 B&W image by 90 degrees counter-clockwise, using one frame
        >>> u = torch.tensor([[[[ 0.5, -0.5], [ 0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]]])
        >>> field = DeformationField(u)
        >>> print(field.inverse_grid_frames)
        tensor([[[[ 0.5, -0.5], [ 0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]]])

    **Example 2:** Rotating a 2x2 B&W image by 90 degrees clockwise, using one frame
        >>> u = torch.tensor([[[[-1, 1], [-1, -1]], [[ 1, 1], [ 1, -1]]]])
        >>> field = DeformationField(u)
        >>> print(field.inverse_grid_frames)
        tensor([[[[-1, 1], [-1, -1]], [[ 1, 1], [ 1, -1]]])
    """

    def __init__(self, inverse_grid_frames: torch.tensor):
        super().__init__()

        if inverse_grid_frames.dtype == torch.float32:
            if self.__class__ == DeformationField:
                msg = (
                    "Consider using float64 when storing the deformation "
                    "field in inverse_grid_frames for greater accuracy."
                )
            if self.__class__ == AffineDeformationField:
                msg = (
                    "Consider using float64 when defining the output type "
                    "of the affine transformation matrix "
                    "inverse_field_matrix for greater accuracy."
                )
            warnings.warn(msg, UserWarning)

        # store as nn.Parameter
        self.inverse_grid_frames = nn.Parameter(
            inverse_grid_frames, requires_grad=False
        )
        # set other properties / inv_grid_frames has shape (n_frames, H, W, 2)
        self.align_corners = True
        self.n_frames = inverse_grid_frames.shape[0]
        self.img_h = inverse_grid_frames.shape[1]
        self.img_w = inverse_grid_frames.shape[2]
        self.img_shape = (self.img_h, self.img_w)

    @property
    def field(self) -> torch.tensor:
        return self.inverse_grid_frames.data

    def forward(
        self,
        img: torch.tensor,
        n0: int = 0,
        n1: int = None,
        mode: str = "bilinear",
    ) -> torch.tensor:
        r"""
        Warps a vectorized image or batch of vectorized images with the stored
        *inverse deformation field* :math:`u`.

        Deforms the vectorized image according to the *inverse deformation
        field* :math:`u` contained in the attribute :attr:`inverse_grid_frames`,
        sliced between the frames :math:`n0` (included) and :math:`n1` (excluded).
        :math:`u` is the field that maps the pixels of the *deformed image* to
        the pixels of the *original image*.

        This method assumes the vectorzed image has the same number of pixels
        as the deformation field.

        Args:
            :attr:`img` (torch.tensor):
            The vectorized image or batch of vectorized images to deform of
            shape :math:`(c, h*w)`, where :math:`c` is the number of channels
            (usually 1 or 3), and :math:`h` and :math:`w` are the number of
            pixels along the height and width of the image respectively.

            :attr:`n0` (int, optional):
            The index of the first frame to use in the *inverse deformation
            field*. Defaults to 0.

            :attr:`n1` (int, optional):
            The index of the first frame to exclude in the *inverse deformation
            field*. If None, all the frames available are used. Defaults to None.

            :attr:`mode` (str, optional):
            The interpolation mode to use. It is directly passed to the
            function :func:`torch.nn.functional.grid_sample`. It must be one of the
            following: 'nearest', 'bilinear', 'bicubic'. Defaults to 'bilinear'.

        .. note::
            If :math:`n0 < n1`, :attr:`inverse_grid_frames` is sliced
            as follows: ``inv_grid_frames[n0:n1, :, :, :]``

        .. note::
            If :math:`n0 > n1`, :attr:`inverse_grid_frames` is sliced
            "backwards". The first frame of the warped animation corresponds to
            the index :math:`n0`, and the last frame corresponds to the index
            :math:`n1+1`. This behavior is identical to slicing a list with a
            step of -1.

        Returns:
            :attr:`output` (torch.tensor):
            The deformed batch of images of shape :math:`(|n1-n0|,c,h,w)`, where each
            image in the batch is deformed according to the *inverse deformation
            field* :math:`u` contained in the attribute :attr:`inverse_grid_frames`.

        Shape:
            :attr:`img`: :math:`(c,h,w)`, where :math:`c` is the number of
            channels, and :math:`h` and :math:`w` are the number of pixels
            along the heigth and width of the image respectively.

            :attr:`output`: :math:`(|n1-n0|,c,h,w)`

        Example 1: Rotating a 2x2 B&W image by 90 degrees counter-clockwise, using one
        frame

        >>> v = torch.tensor([[[[ 1., -1.], [ 1., 1.]],
                               [[-1., -1.], [-1., 1.]]]])
        >>> field = DeformationField(v)
        >>> image = torch.tensor([[[0. , 0.3],
                                   [0.7, 1. ]]])
        >>> deformed_image = field(image, 0, 1)
        >>> print(deformed_image)
        tensor([[[[0.3000, 1.0000],
                  [0.0000, 0.7000]]]])
        """
        # check that the image has the correct number of dimensions
        if img.ndim != 2:
            raise ValueError(
                f"img has incorrect number of dimensions: {img.ndim}, must "
                + "have 2: (channels, n_pixels)."
            )

        if n1 is None:
            n1 = self.n_frames

        # get the right slice of the inverse deformation field
        n_frames = abs(n1 - n0)
        if n1 < n0:
            sel_inv_grid_frames = torch.flip(
                self.inverse_grid_frames[n1 + 1 : n0 + 1, :, :, :], [0]
            )
        else:
            sel_inv_grid_frames = self.inverse_grid_frames[n0:n1, :, :, :]

        # img has current shape (c, n_pixels), make it (n_frames, c, h, w)
        img_frames = (
            img.unsqueeze(0)
            .expand(n_frames, *img.shape)
            .view(n_frames, img.shape[0], self.img_h, self.img_w)
        )
        # img has current shape (c, h, w), make it (n_frames, c, h, w)
        # img_frames = img.unsqueeze(0).expand(n_frames, *img.shape)

        out = nn.functional.grid_sample(
            img_frames.to(sel_inv_grid_frames.dtype),
            sel_inv_grid_frames,
            mode=mode,
            padding_mode="zeros",
            align_corners=self.align_corners,
        ).to(img.dtype)
        return out.reshape(img.shape[-2], n_frames, img.shape[-1])

    def _attributeslist(self):
        a = [
            ("field shape", self.inverse_grid_frames.shape),
            ("n_frames", self.n_frames),
            ("img_shape", self.img_shape),
        ]
        return a

    def __repr__(self):
        s_begin = f"{self.__class__.__name__}(\n  "
        s_fill = "\n  ".join([f"({k}): {v}" for k, v in self._attributeslist()])
        s_end = "\n  )"
        return s_begin + s_fill + s_end

    def __eq__(self, other) -> bool:
        if isinstance(other, DeformationField):
            return bool((self.field == other.field).all())
        return False


# =============================================================================
class AffineDeformationField(DeformationField):
    # =========================================================================
    r"""
    Stores an affine deformation field andn uses it to compute a discrete
    deformation field :class:`DeformationField`.

    Warps an image or batch of images according to an *inverse affine deformation
    field* :math:`u`, i.e. the field that maps the *deformed image* pixel
    coordinates to the *original image* pixel coordinates.

    It is constructed from a function of one parameter (time) that returns a
    tensor of shape :math:`(3,3)` representing a 2D affine homogeneous transformation
    matrix. The homogeneous transformation matrix corresponds to the *inverse
    deformation field* :math:`u`, i.e. the field that maps the pixels of the
    *deformed image* to the pixels of the *original image*.

    To warp an image, the affine transformation matrix is evaluated at each
    time corresponding to the frames of the animation. The *inverse deformation
    field* :math:`u` is then computed from the inverse of the affine
    transformation matrix, and the image is warped according to the *inverse
    deformation field* :math:`u`.

    Contrary to :class:`DeformationField`, this class can warp images of
    variable sizes, as the *inverse deformation field* :math:`u` is computed from the
    affine transformation matrix at the desired spatial resolution.

    .. important::
        The coordinates are given in the range [-1;1]. When referring to a
        pixel, its position is the position of its center. The position
        [-1;-1] corresponds to the center of the top-left pixel.

    Args:
        :attr:`inverse_field_matrix` (torch.tensor):
        Function of one parameter (time) that returns a tensor of shape
        :math:`(3,3)` representing a 2D affine homogeneous transformation
        matrix. That matrix is the *inverse deformation field* :math:`u`, i.e.
        the field that maps the pixels of the *deformed image* to the pixels of
        the *original image*.

    Attributes:
        :attr:`self.inverse_field_matrix` (function of one parameter):
        Function of one parameter (time) that returns a tensor of shape
        :math:`(3,3)` representing a 2D affine homogeneous transformation
        matrix.

        :attr:`t0` (float): First time at which the inverse deformation field is
        computed.

        :attr:`t1` (float): Last time at which the inverse deformation field is
        computed.

        :attr:`n_frames` (int): Number of frames in the animation.

        :attr:`self.inverse_grid_frames` (torch.tensor):
        Inverse grid frames that are computed from thr attribute
        :attr:`inverse_field_matrix` upon calling the method
        :meth:`save_inv_grid_frames`. If set manually,
        the dtype should be `torch.float64`. Default: `None`.

    Example 1: Progressive zooming **in**
        >>> def u(t):
        ...     return torch.tensor([[1-t/10, 0, 0], [0, 1-t/10, 0], [0, 0, 1]])
        >>> field = AffineDeformationField(u)

    Example 2: Rotation of an image **counter-clockwise**, at a frequency of 1Hz
        >>> import numpy as np
        >>> def s(t):
        ...     return np.sin(2*np.pi*t)
        >>> def c(t):
        ...     return np.cos(2*np.pi*t)
        >>> def u(t):
        ...     return torch.tensor([[c(t), s(t), 0], [-s(t), c(t), 0], [0, 0, 1]])
        >>> field = AffineDeformationField(u)
    """

    def __init__(
        self,
        inverse_field_matrix,
        time_vector: torch.tensor,
        # t1: float,
        # n_frames: int,
        img_shape: tuple,
    ) -> None:

        self.inverse_field_matrix = inverse_field_matrix
        self.time_vector = time_vector
        self.align_corners = True  # keep this for _generate_inv_grid_frames

        super().__init__(self._generate_inv_grid_frames(img_shape))

    def _generate_inv_grid_frames(
        self,
        grid_shape: tuple,
    ) -> torch.tensor:
        r"""Generates the inverse deformation field as a tensor of shape
        :math:`(n\_frames, h, w, 2)`.

        This function is called by the constructor to generate the inverse
        deformation field from the affine transformation matrix at the desired
        time points. It is not meant to be called directly.

        Args:
            t0 (float): First time at which the inverse deformation field is
            computed.

            t1 (float): Last time at which the inverse deformation field is
            computed.

            n_frames (int): Number of frames in the animation.

            grid_shape (tuple): shape of the 2D grid to be generated. Must be a
            tuple of the form (h, w), where h and w are respectively the height
            and width of the image to be warped.

        Returns:
            torch.tensor: The inverse deformation field as a tensor of shape
            :math:`(n\_frames, h, w, 2)`.
        """
        # time_vector = torch.linspace(t0, t1, n_frames, dtype=torch.float64)#[:n_frames]
        # self.time_vector = time_vector
        # inv_mat_frames = torch.zeros((n_frames, 2, 3), dtype=torch.float64)

        # get a batch of matrices of shape (n_frames, 2, 3)
        inv_mat_frames = torch.stack(
            [
                self.inverse_field_matrix(t)[:2, :]  # need only the first 2 rows
                for t in self.time_vector
            ]
        )
        # inv_grid_frames = torch.round(inv_mat_frames, decimals=6)

        # use them to generate the grid
        inv_grid_frames = nn.functional.affine_grid(
            inv_mat_frames,
            torch.Size(
                (len(self.time_vector), 1, *grid_shape)
            ),  # n_channels has no effect
            align_corners=self.align_corners,
        )
        return inv_grid_frames