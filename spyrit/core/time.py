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

import operator
import warnings

import torch
import torch.nn as nn
import numpy as np


# =============================================================================
class DeformationField(nn.Module):
    # =========================================================================
    r"""
    Stores a discrete deformation field as a :math:`(b,Nx,Ny,2)` tensor.

    Warps a single image according to an *inverse deformation field* :math:`u`,
    i.e. the field that maps the *deformed image* pixel coordinates to the
    *original image* pixel coordinates.

    It is constructed from a tensor of shape :math:`(n\_frames,Nx,Ny,2)`, where
    :math:`n\_frames` is the number of frames in the animation, :math:`Nx` and
    :math:`Ny` are the number of pixels along the x-axis and y-axis respectively.
    The last dimension contains the x and y coordinates of the original image
    pixel that is displayed in the warped image, at the position corresponding
    to the indices in the dimension :math:`(Nx, Ny)` of the tensor.

    .. note::
        The coordinates are given in the range [-1;1]. When referring to a
        pixel, its position is the position of its center.

    .. note::
        The position [-1;-1] corresponds to the top-left corner of the top-left
        pixel if :attr:`align_corners` is set to `False` (default), and
        to the center of the top-left pixel if :attr:`align_corners` is set to
        `True`.

    Args:
        :attr:`inverse_grid_frames` (torch.tensor, optional):
        *Inverse deformation field* :math:`u` of shape :math:`(n\_frames,Nx,Ny,2)`.
        Default: `None`.

        :attr:`align_corners` (bool, optional):
        Option passed to :func:`torch.nn.functional.grid_sample`.
        Geometrically, we consider the pixels of the input as squares
        rather than points. If set to `True`, the extrema (-1 and 1)
        are considered as referring to the center points of the input's
        corner pixels. If set to `False`, they are instead considered
        as referring to the corner points of the input's corner pixels,
        making the sampling more resolution agnostic. Default: `False`.

    Attributes:
        :attr:`self.inverse_grid_frames` (torch.tensor):
        *Inverse deformation field* :math:`u` of shape :math:`(n\_frames,Nx,Ny,2)`.
        If set manually, the dtype should be `torch.float32`. Default: `None`.

        :attr:`self.align_corners` (bool):
        Should the extrema (-1 and 1) be considered as referring to the center
        points of the input's corner pixels? Default: `False`.

    **Example 1:** Rotating a 2x2 B&W image by 90 degrees counter-clockwise, using one frame and align_corners=False
        >>> u = torch.tensor([[[[ 0.5, -0.5], [ 0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]]])
        >>> field = DeformationField(u, align_corners=False)
        >>> print(field.inverse_grid_frames)
        tensor([[[[ 0.5, -0.5], [ 0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]]])

    **Example 2:** Rotating a 2x2 B&W image by 90 degrees clockwise, using one frame and align_corners=True
        >>> u = torch.tensor([[[[-1, 1], [-1, -1]], [[ 1, 1], [ 1, -1]]]])
        >>> field = DeformationField(u, align_corners=True)
        >>> print(field.inverse_grid_frames)
        tensor([[[[-1, 1], [-1, -1]], [[ 1, 1], [ 1, -1]]])
    """

    def __init__(self, inverse_grid_frames: torch.tensor, align_corners=False):
        super().__init__()
        # convert to float 23 then store as nn.Parameter
        inverse_grid_frames = inverse_grid_frames.type(torch.float32)
        self.inverse_grid_frames = nn.Parameter(
            inverse_grid_frames, requires_grad=False
        )

        # set other properties
        self.align_corners = align_corners
        self.n_frames, self.Nx, self.Ny = inverse_grid_frames.size()[0:3]

    def get_inv_grid_frames(self):
        r"""
        Returns the *inverse deformation field* :math:`u`.

        Returns the *inverse deformation field* :math:`u` contained in the
        attribute :attr:`inverse_grid_frames`.

        Returns:
            :attr:`self.inverse_grid_frames` (torch.tensor):
            *Inverse deformation field* :math:`u` of shape :math:`(n\_frames,Nx,Ny,2)`.

        Example 1: Get the inverse deformation field of a 2x2 B&W image rotated by 90 degrees counter-clockwise
            >>> u = torch.tensor([[[[ 0.5, -0.5], [ 0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]]])
            >>> field = DeformationField(u, align_corners=False)
            >>> print(field.get_inv_grid_frames())
            tensor([[[[ 0.5, -0.5], [ 0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]])
        """
        if self.inverse_grid_frames is None:
            return None
        return self.inverse_grid_frames.data

    def forward(
        self, img: torch.tensor, n0: int, n1: int, mode: str = "bilinear"
    ) -> torch.tensor:
        r"""
        Warps an image or batch of images with the stored *inverse deformation field*
        :math:`u`.

        Deforms the image or batch of images according to the *inverse deformation
        field* :math:`u` contained in the attribute :attr:`inverse_grid_frames`,
        sliced between the frames :math:`n0` (included) and :math:`n1` (excluded).
        :math:`u` is the field that maps the pixels of the *deformed image* to
        the pixels of the *original image*.

        Args:
            :attr:`img` (torch.tensor):
            The image or batch of images to deform of shape :math:`(c,Nx,Ny)`
            or :math:`(B,c,Nx,Ny)`,
            where :math:`B` is the number of images in the batch, :math:`c` is
            the number of channels (usually 1 or 3), and :math:`Nx` and :math:`Ny`
            are the number of pixels along the x-axis and y-axis respectively.

            :attr:`n0` (int):
            The index of the first frame to use in the *inverse deformation
            field*.

            :attr:`n1` (int):
            The index of the first frame to exclude in the *inverse deformation
            field*.

            :attr:`mode` (str, optional):
            The interpolation mode to use. It is directly passed to the
            function :func:`torch.nn.functional.grid_sample`. It must be one of the
            following: 'nearest', 'bilinear', 'bicubic'. Defaults to 'bilinear'.

        .. important::
            The input shape must be either :math:`(c,Nx,Ny)` or :math:`(B,c,Nx,Ny)`.

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
            The deformed batch of images of shape :math:`(|n1-n0|,c,Nx,Ny)` or
            :math:`(B,|n1-n0|,c,Nx,Ny)` depending on the input shape, where each
            image in the batch is deformed according to the *inverse deformation
            field* :math:`u` contained in the attribute :attr:`inverse_grid_frames`.

        Shape:
            :attr:`img`: :math:`(c,Nx,Ny)` or :math:`(B,c,Nx,Ny)`, where :math:`B`
            is the batch size, :math:`c` is the number of channels, and
            :math:`Nx` and :math:`Ny` are the number of pixels along the x-axis
            and y-axis respectively.

            :attr:`output`: :math:`(|n1-n0|,c,Nx,Ny)` or :math:`(B,|n1-n0|,c,Nx,Ny)`,
            depending on the input shape.

        Example 1: Rotating a 2x2 B&W image by 90 degrees counter-clockwise, using one
        frame and align_corners=True

        >>> v = torch.tensor([[[[ 1., -1.], [ 1., 1.]],
                               [[-1., -1.], [-1., 1.]]]])
        >>> field = DeformationField(v, align_corners=True)
        >>> image = torch.tensor([[[0. , 0.3],
                                   [0.7, 1. ]]])
        >>> deformed_image = field(image, 0, 1)
        >>> print(deformed_image)
        tensor([[[[0.3000, 1.0000],
                  [0.0000, 0.7000]]]])
        """
        # check that the image has the correct number of dimensions
        img_size = img.size()

        if (len(img_size) < 3) or (len(img_size) > 4):
            raise ValueError(
                f"img has incorrect number of dimensions: {img_size} (must have at 3 or 4)."
            )
        elif len(img_size) == 3:
            img = img.unsqueeze(0)  # make it 4D with size (1, c, Nx, Ny)

        nb_frames = abs(n1 - n0)
        batch_size = img.size(0)

        # vvv no longer needed with nn.Parameter ? vvv
        # check that the deformation field and the image are on the same device
        # self.inverse_grid_frames = self.inverse_grid_frames.to(img.device)

        # get the right slice of the inverse deformation field v
        if n1 < n0:
            sel_inv_grid_frames = torch.flip(
                self.inverse_grid_frames[n1 + 1 : n0 + 1, :, :, :], [0]
            )
        else:
            sel_inv_grid_frames = self.inverse_grid_frames[n0:n1, :, :, :].clone()

        # img has current shape (B, c, Nx, Ny), B is the batch size
        # make it (B, n_frames, c, Nx, Ny)
        img_frames = img.unsqueeze(1).expand(-1, nb_frames, -1, -1, -1)
        out = torch.zeros_like(img_frames)

        for i in range(batch_size):
            # picture is (n_frames, c, Nx, Ny)
            out[i] = nn.functional.grid_sample(
                img_frames[i],
                sel_inv_grid_frames,
                mode=mode,
                padding_mode="zeros",
                align_corners=self.align_corners,
            )
        if batch_size == 1:
            return out.squeeze(0)
        return out

    def __repr__(self):
        s = f"DeformationField({self.inverse_grid_frames=}, {self.align_corners=})"
        return s


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

    .. note::
        The coordinates are given in the range [-1;1]. When referring to a
        pixel, its position is the position of its center.

    .. note::
        The position [-1;-1] corresponds to the top-left corner of the top-left
        pixel if :attr:`align_corners` is set to `False` (default), and
        to the center of the top-left pixel if :attr:`align_corners` is set to
        `True`.

    Args:
        :attr:`inverse_field_matrix` (torch.tensor):
        Function of one parameter (time) that returns a tensor of shape
        :math:`(3,3)` representing a 2D affine homogeneous transformation
        matrix. That matrix is the *inverse deformation field* :math:`u`, i.e.
        the field that maps the pixels of the *deformed image* to the pixels of
        the *original image*.

        :attr:`align_corners` (bool, optional):
        Geometrically, we consider the pixels of the input as squares rather
        than points. If set to `True`, the extrema (-1 and 1) are
        considered as referring to the center points of the input's corner
        pixels. If set to `False`, they are instead considered as
        referring to the corner points of the input's corner pixels, making
        the sampling more resolution agnostic. Default: `False`.
        See :func:`torch.nn.functional.grid_sample` for more details.

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
        the dtype should be `torch.float32`. Default: `None`.

        :attr:`self.align_corners` (bool):
        Should the extrema (-1 and 1) be considered as referring to the center
        points of the input's corner pixels? Default: `False`.

    Example 1: Progressive zooming **in**
        >>> def u(t):
        ...     return torch.tensor([[1-t/10, 0, 0], [0, 1-t/10, 0], [0, 0, 1]])
        >>> field = AffineDeformationField(u, align_corners=False)

    Example 2: Rotation of an image **counter-clockwise**, at a frequency of 1Hz
        >>> import numpy as np
        >>> def s(t):
        ...     return np.sin(2*np.pi*t)
        >>> def c(t):
        ...     return np.cos(2*np.pi*t)
        >>> def u(t):
        ...     return torch.tensor([[c(t), s(t), 0], [-s(t), c(t), 0], [0, 0, 1]])
        >>> field = AffineDeformationField(u, align_corners=False)
    """

    def __init__(
        self,
        inverse_field_matrix: torch.tensor,
        t0: float,
        t1: float,
        n_frames: int,
        img_size: tuple,
        align_corners=False,
    ) -> None:

        self.inverse_field_matrix = inverse_field_matrix
        self.align_corners = align_corners
        self.t0, self.t1 = t0, t1

        super().__init__(
            self._generate_inv_grid_frames(t0, t1, n_frames, img_size), align_corners
        )

    def _generate_inv_grid_frames(
        self,
        t0: float,
        t1: float,
        n_frames: int,
        grid_size: tuple,
    ) -> torch.tensor:
        r"""Generates the inverse deformation field as a tensor of shape
        :math:`(n\_frames, Nx, Ny, 2)`.

        This function is called by the constructor to generate the inverse
        deformation field from the affine transformation matrix at the desired
        time points. It is not meant to be called directly.

        Args:
            t0 (float): First time at which the inverse deformation field is
            computed.

            t1 (float): Last time at which the inverse deformation field is
            computed.

            n_frames (int): Number of frames in the animation.

            grid_size (tuple): size of the 2D grid to be generated. Must be a
            tuple of the form (Nx, Ny), where Nx and Ny are the number of pixels
            in the image to be warped along the x-axis and y-axis respectively.

        Returns:
            torch.tensor: The inverse deformation field as a tensor of shape
            :math:`(n\_frames, Nx, Ny, 2)`.
        """
        time_vector = torch.linspace(t0, t1, n_frames)
        inv_mat_frames = torch.zeros((n_frames, 2, 3), dtype=torch.float32)

        # get a batch of matrices of shape (n_frames, 2, 3)
        for i, t in enumerate(time_vector):
            inv_mat_frames[i] = self.inverse_field_matrix(t)[:2, :]

        # use them to generate the grid
        inv_grid_frames = nn.functional.affine_grid(
            inv_mat_frames,
            torch.Size((n_frames, 1, *grid_size)),  # n_channels has no effect
            align_corners=self.align_corners,
        )
        return inv_grid_frames

    # def forward(self,
    #             img: torch.tensor,
    #             t0: float,
    #             t1: float = None,
    #             n_frames: int = None,
    #             mode: str = "bilinear",
    #             ) -> torch.tensor:
    #     r"""
    #     Warps an image or batch of images according to the time interpolation parameters.

    #     Similarly to the method :meth:`DeformationField.forward` from the parent
    #     class, it warps an image or batch of images according to the *inverse
    #     deformation field* :math:`u` contained in the attribute
    #     :attr:`inverse_grid_frames`, between the times :math:`t0` and
    #     :math:`t1`. The number of frames in the animation is given by either
    #     :math:`n\_frames`, or :math:`fps` if :math:`n\_frames` is `None`.

    #     Args:
    #         :attr:`img` (torch.tensor):
    #         Image to deform of shape :math:`(c,Nx,Ny)` or batch of images to
    #         deform of shape :math:`(B,c,Nx,Ny)`, where :math:`B` is the number
    #         of images in the batch, :math:`c` is the
    #         number of channels, and :math:`Nx` and
    #         :math:`Ny` are the number of pixels along the x-axis and y-axis
    #         respectively. The number of channels is usually 1 (grayscale) or
    #         3 (color).

    #         :attr:`t0` (float):
    #         Start time of the animation.

    #         :attr:`t1` (float):
    #         End time of the animation. If `None`, a single frame is warped.

    #         :attr:`n_frames` (int):
    #         Number of frames in the animation. If `None`,
    #         :attr:`fps` is used to compute the number of frames.

    #         :attr:`fps` (float):
    #         Number of frames per second. If `None`, :attr:`n_frames`
    #         is used, or a single frame is warped.

    #         :attr:`mode` (str, optional):
    #         The interpolation mode to use. It is
    #         directly passed to the function :func:`torch.nn.functional.grid_sample`.
    #         It must be one of the following: 'nearest', 'bilinear',
    #         'bicubic'. Defaults to 'bilinear'.

    #     Returns:
    #         :attr:`output` (torch.tensor):
    #         The deformed batch of images of shape :math:`(n\_frames,c,Nx,Ny)`
    #         or :math:`(B,n\_frames,c,Nx,Ny)`, where each image is
    #         deformed according to the *inverse deformation field* :math:`u`
    #         contained in the attribute :attr:`inverse_grid_frames`.

    #     Example 1: Rotate a single image by 90° counter-clockwise
    #         >>> def u(t):
    #         ...     return torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    #         >>> field = AffineDeformationField(u, align_corners=False)
    #         >>> image = torch.tensor([[[0. , 0.3],
    #                                    [0.7, 1. ]]])
    #         >>> deformed_image = field(image, 0)
    #         >>> print(deformed_image)
    #         tensor([[[[0.3000, 1.0000],
    #                   [0.0000, 0.7000]]]])

    #     Example 2: Animate an image between t0=0 and t1=1, with 5 frames, with counter-clockwise rotation
    #         >>> def s(t):
    #         ...     return np.sin(2*np.pi*t)
    #         >>> def c(t):
    #         ...     return np.cos(2*np.pi*t)
    #         >>> def u(t):
    #         ...     return torch.tensor([[c(t), -s(t), 0], [s(t), c(t), 0], [0, 0, 1]])
    #         >>> field = AffineDeformationField(u, align_corners=False)
    #         >>> image = torch.tensor([[[0. , 0.3],
    #                                    [0.7, 1. ]]])
    #         >>> deformed_image = field(image, 0, 1, 5)
    #         >>> print(deformed_image)
    #         tensor([[[[0.0000, 0.3000],
    #                   [0.7000, 1.0000]]],

    #                 [[[0.3000, 1.0000],
    #                   [0.0000, 0.7000]]],

    #                 [[[1.0000, 0.7000],
    #                   [0.3000, 0.0000]]],

    #                 [[[0.7000, 0.0000],
    #                   [1.0000, 0.3000]]],

    #                 [[[0.0000, 0.3000],
    #                   [0.7000, 1.0000]]]])
    #     """
    #     # get the right t1 and n_frames
    #     t0, t1, n_frames = self.format_params(t0, t1, n_frames, fps)
    #     # get the inverse deformation field as batch of 3x3 matrices
    #     inv_mat_frames = self.get_inv_mat_frames(t0, t1, n_frames)
    #     # save the corresponding inverse deformation field in attribute
    #     self.save_inv_grid_frames(inv_mat_frames, [n_frames, *img.size()[-3:]])
    #     # use the parent class forward method
    #     return super().forward(img, 0, n_frames, mode=mode)

    # def format_params(self, t0: float, t1: float, n_frames: int, fps: float) -> tuple:
    #     r"""
    #     Returns the corrected parameters :math:`t0`, :math:`t1` and :math:`n_frames`

    #     Returns the parameters :math:`t0`, :math:`t1` and :math:`n_frames` in a
    #     format that can be used to animate an image in multiple frames or warp
    #     an image in a single frame.

    #     .. note::
    #         The parameters :attr:`n_frames` and :attr:`fps` are mutually
    #         exclusive. If both are given, :attr:`n_frames` is used. If both are
    #         :attr:`None`, a single frame is warped.

    #     .. note::
    #         If :attr:`fps` is given and :attr:`n_frames` is `None`, the
    #         end time :math:`t1` is truncated to the closest lowest multiple of
    #         :math:`1/\text{fps}`. The number of frames is then computed as
    #         usual between :math:`t0` and the truncated :math:`t1`.

    #     Args:
    #         :attr:`t0` (float):
    #         Start time of the animation.

    #         :attr:`t1` (float):
    #         End time of the animation. If `None`, a single frame is warped.

    #         :attr:`n_frames` (int):
    #         Number of frames in the animation. If `None`, a single frame
    #         is warped.

    #         :attr:`fps` (float):
    #         Number of frames per second. If `None`, a single frame is warped.

    #     Returns:
    #         :attr:`(t0, t1, n_frames)` (tuple):
    #         Where :math:`t0` (float) and :math:`t1` (float) are the start and
    #         end time of the animation respectively, and :math:`n_frames` (int)
    #         is the number of frames in the animation.

    #     The output follows this pattern::

    #         (t1 is None)
    #         ├── Yes
    #         │    └── (n_frames is None) and (fps is None)
    #         │         ├── Yes ──> (t0, t0, 1): a single frame is warped.
    #         │         └── No ──> ValueError (conflicting parameters).
    #         └── No
    #             └── (n_frames is None)
    #                 ├── Yes
    #                 │    └── (fps is None)
    #                 │         ├── Yes ──> ValueError (conflicting parameters).
    #                 │         └── No ─┬─ n_frames = int(np.floor(1 + (t1-t0) * fps))
    #                 │                 └──> (t0, t0 + (n_frames-1) / fps, n_frames)
    #                 └── No ──> (t0, t1, n_frames)

    #     Example 1: Warp a single image at t0=0, with 1 frame
    #         >>> field = AffineDeformationField(None, align_corners=False)
    #         >>> field.format_params(0)
    #         (0, 0, 1)

    #     Example 2: Warp a single image between t0=0 and t1=1, with 11 frames
    #         >>> field = AffineDeformationField(None, align_corners=False)
    #         >>> field.format_params(0, 1, 11)
    #         (0, 1, 11)

    #     Example 3: Warp a single image between t0=0 and t1=1 at 24 fps
    #         >>> field = AffineDeformationField(None, align_corners=False)
    #         >>> field.format_params(0, 1, None, 24)
    #         (0, 1, 25)

    #     Example 4: Provide n_frames and fps
    #         >>> field = AffineDeformationField(None, align_corners=False)
    #         >>> field.format_params(0, 1, 11, 24)
    #         (0, 1, 11)

    #     Example 5: Provide conflicting parameters
    #         >>> field = AffineDeformationField(None, align_corners=False)
    #         >>> field.format_params(0, 1, None, None)
    #         ValueError: Unable to animate multiple frames: t1 (1) was given, but fps and n_frames were not given.

    #     Example 6: Provide conflicting parameters
    #         >>> field = AffineDeformationField(None, align_corners=False)
    #         >>> field.format_params(0, None, 11, 24)
    #         ValueError: Unable to animate multiple frames: t1 was not given, but n_frames (11) or fps (24) or were given.
    #     """
    #     if t1 is None:
    #         if (fps is None) and (n_frames is None):
    #             return (t0, t0, 1)  # no animation
    #         else:
    #             raise ValueError(
    #                 "Unable to animate multiple frames: t1 was not given, "
    #                 + f"but n_frames ({n_frames}) or fps ({fps}) were given."
    #             )
    #     else:
    #         # if fps and n_frames are given, use n_frames
    #         if n_frames is None:
    #             if fps is None:
    #                 raise ValueError(
    #                     f"Unable to warp one image: t1 ({t1}) was given, "
    #                     + "but fps and n_frames were not given."
    #                 )
    #             else:
    #                 # t1 is truncated to the closest lowest multiple of 1/fps
    #                 n_frames = int(np.floor(1 + (t1 - t0) * fps))
    #                 return (t0, t0 + (n_frames - 1) / fps, n_frames)
    #         else:
    #             return (t0, t1, n_frames)

    # def get_inv_mat_frames(self,
    #                        t0: float,
    #                        t1: float = None,
    #                        n_frames: int = 1
    #                        ) -> torch.tensor:
    #     r"""
    #     Returns a batch of affine transformation matrices of shape :math:`(n\_frames,3,3)`.

    #     Returns a batch of affine transformation matrices corresponding to the
    #     *inverse deformation field* :math:`u`, evaluated at the times defined
    #     by the parameters :math:`t0`, :math:`t1` and :math:`n\_frames`.

    #     .. note::
    #         The time vector is created using the function :func:`numpy.linspace`
    #         with the parameters :math:`t0`, :math:`t1` and :math:`n\_frames` (
    #         ``time_vector = np.linspace(t0, t1, n_frames)``). If :math:`t0 > t1`,
    #         the time vector is created in reverse order, giving a "backwards"
    #         animation.

    #     Args:
    #         :attr:`t0` (float):
    #         The first time at which to evaluate the function that gives the
    #         affine transformation matrix.

    #         :attr:`t1` (float, optional):
    #         The last time at which to evaluate the function. If `None`,
    #         the function is evaluated at the time :math:`t0`. Default: `None`.

    #         :attr:`n_frames` (int, optional):
    #         The number of frames in the animation. Default: 1.

    #     Returns:
    #         :attr:`inv_mat_frames` (torch.tensor):
    #         A batch of affine transformation matrices of shape :math:`(n\_frames,3,3)`,
    #         with dtype `torch.float32`.

    #     Example 1: Evaluate the affine transformation matrix between t0=0 and t1 = 10, with 11 frames
    #         >>> def u(t):
    #         ...     return torch.tensor([[1-t/10, 0, 0], [0, 1-t/10, 0], [0, 0, 1]])
    #         >>> field = AffineDeformationField(u, align_corners=False)
    #         >>> inv_mat_frames = field.get_inv_mat_frames(0, 10, 11)
    #         >>> print(inv_mat_frames.size())
    #         torch.Size([11, 3, 3])

    #     Example 2: Evaluate the affine transformation matrix at t0=4.5, with 1 frame
    #         >>> def u(t):
    #         ...     return torch.tensor([[1-t/10, 0, 0], [0, 1-t/10, 0], [0, 0, 1]])
    #         >>> field = AffineDeformationField(u, align_corners=False)
    #         >>> inv_mat_frames = field.get_inv_mat_frames(4.5)
    #         >>> print(inv_mat_frames.size())
    #         torch.Size([1, 3, 3])
    #     """
    #     if t1 is None:
    #         return self.inverse_field_matrix(t0).unsqueeze(0)

    #     n_frames = operator.index(n_frames)
    #     time_vector = np.linspace(t0, t1, n_frames)
    #     inv_mat_frames = torch.zeros((n_frames, 3, 3), dtype=torch.float32)

    #     for i, t in enumerate(time_vector):
    #         inv_mat_frames[i] = self.inverse_field_matrix(t)
    #     return inv_mat_frames

    # def save_inv_grid_frames(
    #     self, inv_mat_frames: torch.tensor, size: torch.Size
    # ) -> torch.tensor:
    #     r"""
    #     Saves as a class attribute the inverse deformation field :math:`u`.

    #     Saves as a class attribute :attr:`self.inverse_grid_frames` the
    #     *inverse deformation field* :math:`u` computed from the inverse of the affine
    #     transformation matrices, evaluated at multiple times.

    #     .. note::
    #         The name of the attribute (:attr:`inverse_grid_frames`) is the
    #         same as the attribute of the parent class :class:`DeformationField`.
    #         This allows both classes to use the same method to warp images.

    #     Args:
    #         :attr:`inv_mat_frames` (torch.tensor):
    #         shape :math:`(n\_frames,3,3)`
    #         Batch of inverse affine transformation matrices.

    #         :attr:`size` (torch.Size):
    #         shape :math:`(n\_frames,c,Nx,Ny)`
    #         Target output image size. It is a 4-tuple of integers, where
    #         :math:`n\_frames` is the number of frames in the animation,
    #         :math:`c` is the number of channels, and :math:`Nx` and :math:`Ny`
    #         are the number of pixels along the x-axis and y-axis respectively.
    #         The number of channels is usually 1 (grayscale) or 3 (color).

    #     Returns:
    #         :attr:`self.inverse_grid_frames`: has the following
    #         shape: :math:`(n\_frames,Nx,Ny,2)`.

    #     Example: Save the *inverse deformation field* from a rotation matrix, for a RGB 128x128 image
    #         >>> def s(t):
    #         ...     return np.sin(2*np.pi*t)
    #         >>> def c(t):
    #         ...     return np.cos(2*np.pi*t)
    #         >>> def u(t):
    #         ...     return torch.tensor([[c(t), -s(t), 0], [s(t), c(t), 0], [0, 0, 1]])
    #         >>> field = AffineDeformationField(u, align_corners=False)
    #         >>> inv_mat_frames = field.get_inv_mat_frames(0, 10, 101)
    #         >>> field.save_inv_grid_frames(inv_mat_frames, [101, 3, 128, 128])
    #         >>> print(field.inverse_grid_frames.size())
    #         torch.Size([101, 128, 128, 2])
    #     """
    #     # affine_grid needs the first two rows
    #     theta = inv_mat_frames[:, :2, :].float()
    #     inv_grid_frames = nn.functional.affine_grid(
    #         theta, size, align_corners=self.align_corners
    #     )
    #     self.inverse_grid_frames = nn.Parameter(inv_grid_frames, requires_grad=False)
    #     return self.inverse_grid_frames

    def __repr__(self):
        s = (
            f"AffineDeformationField({self.inverse_field_matrix.__name__=}, "
            + f"{self.align_corners=}, {self.inverse_grid_frames=})"
        )
        return s
