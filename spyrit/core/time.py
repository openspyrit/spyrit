"""
Stores deformation fields and warps images.

Contains :class:`DeformationField` and :class:`AffineDeformationField`, a
subclass of the former. These classes are used to warp images according to
a deformation field that is stored as as class attribute. They can be fed
an image (called "*original image*") and will return the warped image
("*deformed image*").

The function that maps the pixels of the *original image* to the pixels of
the *deformed image* is called the "*deformation field*" and is noted
:math:`u`. The function that maps the pixels of the *deformed image* to the
pixels of the *original image* is called the "*inverse deformation field*" and
is noted :math:`v`. The *deformation field* and the *inverse deformation field*
are related by the equation :math:`u = v^{-1}`.

Here, the two classes use and store the *inverse deformation field* :math:`v`
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

    Warps a single image according to an *inverse deformation field* :math:`v`,
    i.e. the field that maps the pixels of the deformed image to the pixels of
    the original image.

    It is constructed from a tensor of shape :math:`(n\_frames,Nx,Ny,2)`, where
    :math:`n\_frames` is the number of frames in the animation, :math:`Nx` and
    :math:`Ny` are the number of pixels along the x-axis and y-axis respectively.
    The last dimension contains the x and y coordinates of the original image
    pixel that is displayed in the warped image, at the position corresponding
    to the indices in the dimension :math:`(Nx,Ny)` of the tensor.

    .. note::
        The coordinates are given in the range [-1;1]. When referring to a
        pixel, its position is the position of its center.

    .. note::
        The position [-1;-1] corresponds to the top-left corner of the top-left
        pixel if :attr:`align_corners` is set to :attr:`False` (default), and
        to the center of the top-left pixel if :attr:`align_corners` is set to
        :attr:`True`.

    Args:
        :attr:`inverse_grid_frames` (torch.tensor, optional):
        *Inverse deformation field* :math:`v` of shape :math:`(n\_frames,Nx,Ny,2)`.
        Default: :attr:`None`.

        :attr:`align_corners` (bool, optional):
        Option passed to `torch.nn.functional.grid_sample`.
        Geometrically, we consider the pixels of the input as squares
        rather than points. If set to :attr:`True`, the extrema (-1 and 1)
        are considered as referring to the center points of the input's
        corner pixels. If set to :attr:`False`, they are instead considered
        as referring to the corner points of the input's corner pixels,
        making the sampling more resolution agnostic. Default: :attr:`False`.

    Attributes:
        :attr:`self.inverse_grid_frames` (torch.tensor):
        *Inverse deformation field* :math:`v` of shape :math:`(n\_frames,Nx,Ny,2)`.
        If set manually, the dtype should be :attr:`torch.float32`. Default: :attr:`None`.

        :attr:`self.align_corners` (bool):
        Should the extrema (-1 and 1) be considered as referring to the center
        points of the input's corner pixels? Default: :attr:`False`.

    **Example 1:** Rotating a 2x2 B&W image by 90 degrees counter-clockwise, using one frame and align_corners=False
        >>> v = torch.tensor([[[[ 0.5, -0.5], [ 0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]]])
        >>> field = DeformationField(v, align_corners=False)
        >>> print(field.inverse_grid_frames)
        tensor([[[[ 0.5, -0.5], [ 0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]]])

    **Example 2:** Rotating a 2x2 B&W image by 90 degrees clockwise, using one frame and align_corners=True
        >>> v = torch.tensor([[[[-1, 1], [-1, -1]], [[ 1, 1], [ 1, -1]]]])
        >>> field = DeformationField(v, align_corners=True)
        >>> print(field.inverse_grid_frames)
        tensor([[[[-1, 1], [-1, -1]], [[ 1, 1], [ 1, -1]]])
    """

    def __init__(self, inverse_grid_frames: torch.tensor = None, align_corners=False):
        super().__init__()
        if inverse_grid_frames is not None:
            self.inverse_grid_frames = inverse_grid_frames.float()
        else:
            self.inverse_grid_frames = None
        self.align_corners = align_corners

    def warp(
        self, img: torch.tensor, n0: int, n1: int, mode: str = "bilinear"
    ) -> torch.tensor:
        r"""
        Warps a given image with the stored *inverse deformation field* :math:`v`.

        Deforms the image according to the *inverse deformation field* :math:`v`
        contained in the attribute :attr:`inverse_grid_frames`, sliced between
        the frames :math:`n0` (included) and :math:`n1` (excluded).

            .. note::
                If :math:`n0 < n1`, :attr:`inverse_grid_frames` is sliced
                as follows: ``inv_grid_frames[n0:n1, :, :, :]``

            .. note::
                If :math:`n0 > n1`, :attr:`inverse_grid_frames` is sliced
                "backwards". The first frame of the warped animation corresponds to
                the index :math:`n0`, and the last frame corresponds to the index
                :math:`n1+1`. This behavior is identical to slicing a list with a
                step of -1.

        Args:
            :attr:`img` (torch.tensor):
            The image to deform of shape :math:`(c,Nx,Ny)`, where :math:`c`
            is the number of channels, and :math:`Nx` and :math:`Ny` are the
            number of pixels along the x-axis and y-axis respectively. The
            number of channels is usually 1 (grayscale) or 3 (color), if not a
            warning is raised. If the image has not 3 dimensions, an error is
            raised.

            :attr:`n0` (int):
            The index of the first frame to use in the *inverse deformation
            field*.

            :attr:`n1` (int):
            The index of the first frame to exclude in the *inverse deformation
            field*.

            :attr:`mode` (str, optional):
            The interpolation mode to use. It is directly passed to the
            function `torch.nn.functional.grid_sample`. It must be one of the
            following: 'nearest', 'bilinear', 'bicubic'. Defaults to 'bilinear'.

        Returns:
            :attr:`output` (torch.tensor):
            The deformed batch of images of shape :math:`(|n1-n0|,c,Nx,Ny)`,
            where each image is deformed according to the *inverse deformation
            field* :math:`v` contained in the attribute :attr:`inverse_grid_frames`.

        Shape:
            - :attr:`img`: :math:`(c,Nx,Ny)`, where c is the number of channels,
                Nx and Ny are the number of pixels along the x-axis and y-axis
                respectively.
            - :attr:`output`: :math:`(|n1-n0|,c,Nx,Ny)`

        Example 1: Rotating a 2x2 B&W image by 90 degrees counter-clockwise, using one
        frame and align_corners=True

        >>> v = torch.tensor([[[[ 1., -1.], [ 1., 1.]],
                               [[-1., -1.], [-1., 1.]]]])
        >>> field = DeformationField(v, align_corners=True)
        >>> image = torch.tensor([[[0. , 0.3],
                                   [0.7, 1. ]]])
        >>> deformed_image = field.warp(image, 0, 1)
        >>> print(deformed_image)
        tensor([[[[0.3000, 1.0000],
                  [0.0000, 0.7000]]]])
        """
        # check that the image has the correct number of dimensions
        img_size = img.size()
        if len(img_size) == 3:
            if img_size[0] not in [1, 3]:
                warnings.warn(
                    "The first dimension of the image should be the number of"
                    + f"channels (1 or 3), found: {img_size[0]}."
                )
        else:
            raise ValueError(
                f"Image has incorrect number of dimensions: {img_size} (must have 3)."
            )

        # check that the deformation field and the image are on the same device
        self.inverse_grid_frames = self.inverse_grid_frames.to(img.device)

        # get the right slice of the inverse deformation field v
        if n1 < n0:
            sel_inv_grid_frames = torch.flip(
                self.inverse_grid_frames[n1 + 1 : n0 + 1, :, :, :], [0]
            )
        else:
            sel_inv_grid_frames = self.inverse_grid_frames[n0:n1, :, :, :].clone()

        nb_frames = abs(n1 - n0)
        img_frames = img.expand(nb_frames, -1, -1, -1)

        warped = nn.functional.grid_sample(
            img_frames,
            sel_inv_grid_frames,
            mode=mode,
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        return warped

    def __repr__(self):
        s = f"DeformationField({self.inverse_grid_frames=}, {self.align_corners=})"
        return s


# =============================================================================
class AffineDeformationField(DeformationField):
    # =========================================================================
    r"""
    Stores an affine deformation field as a function of time.

    Warps a single image according to an *inverse affine deformation field*
    :math:`v`, i.e. the field that maps the pixels of the *deformed image* to
    the pixels of the *original image*.

    It is constructed from a function of one parameter (time) that returns a
    tensor of shape :math:`(3,3)` representing a 2D affine homogeneous transformation
    matrix. The homogeneous transformation matrix corresponds to the *inverse
    deformation field* :math:`v`, i.e. the field that maps the pixels of the
    *deformed image* to the pixels of the *original image*.

    To warp an image, the affine transformation matrix is evaluated at each
    time corresponding to the frames of the animation. The *inverse deformation
    field* :math:`v` is then computed from the inverse of the affine
    transformation matrix, and the image is warped according to the *inverse
    deformation field* :math:`v`.

    Contrary to :class:`DeformationField`, this class can warp images of
    variable sizes, as the *inverse deformation field* :math:`v` is computed from the
    affine transformation matrix at the desired spatial resolution.

    .. note::
        The coordinates are given in the range [-1;1]. When referring to a
        pixel, its position is the position of its center.

    .. note::
        The position [-1;-1] corresponds to the top-left corner of the top-left
        pixel if :attr:`align_corners` is set to :attr:`False` (default), and
        to the center of the top-left pixel if :attr:`align_corners` is set to
        :attr:`True`.

    Args:
        :attr:`inverse_field_matrix` (torch.tensor):
        Function of one parameter (time) that returns a tensor of shape
        :math:`(3,3)` representing a 2D affine homogeneous transformation
        matrix. That matrix is the *inverse deformation field* :math:`v`, i.e.
        the field that maps the pixels of the *deformed image* to the pixels of
        the *original image*.

        :attr:`align_corners` (bool, optional):
        Geometrically, we consider the pixels of the input as squares rather
        than points. If set to :attr:`True`, the extrema (-1 and 1) are
        considered as referring to the center points of the input's corner
        pixels. If set to :attr:`False`, they are instead considered as
        referring to the corner points of the input's corner pixels, making
        the sampling more resolution agnostic. Default: :attr:`False`.

    Attributes:
        :attr:`self.inverse_field_matrix` (torch.tensor):
        Function of one parameter (time) that returns a tensor of shape
        :math:`(3,3)` representing a 2D affine homogeneous transformation
        matrix.

        :attr:`self.align_corners` (bool):
        Should the extrema (-1 and 1) be considered as referring to the center
        points of the input's corner pixels? Default: :attr:`False`.

        :attr:`self.inverse_grid_frames` (torch.tensor):
        Inverse grid frames that are computed from :attr:`self.inverse_field_matrix`
        upon calling the method :meth:`save_inv_grid_frames`. If set manually,
        the dtype should be :attr:`torch.float32`. Default: :attr:`None`.

    Example 1: Progressive zooming **in**
        >>> def v(t):
        ...     return torch.tensor([[1-t/10, 0, 0], [0, 1-t/10, 0], [0, 0, 1]])
        >>> field = AffineDeformationField(v, align_corners=False)

    Example 2: Rotation of an image **counter-clockwise**, at a frequency of 1Hz
        >>> import numpy as np
        >>> def s(t):
        ...     return np.sin(2*np.pi*t)
        >>> def c(t):
        ...     return np.cos(2*np.pi*t)
        >>> def v(t):
        ...     return torch.tensor([[c(t), s(t), 0], [-s(t), c(t), 0], [0, 0, 1]])
        >>> field = AffineDeformationField(v, align_corners=False)
    """

    def __init__(self, inverse_field_matrix: torch.tensor, align_corners=False):
        super().__init__(None, align_corners)
        self.inverse_field_matrix = inverse_field_matrix

    def get_inv_mat_frames(
        self, t0: float, t1: float = None, n_frames: int = 1
    ) -> torch.tensor:
        r"""
        Returns a batch of affine transformation matrices of shape :math:`(n\_frames,3,3)`.

        Returns a batch of affine transformation matrices corresponding to the
        *inverse deformation field* :math:`v`, evaluated at the times defined
        by the parameters :math:`t0`, :math:`t1` and :math:`n\_frames`.

        .. note::
            The time vector is created using the function `numpy.linspace` with
            the parameters :math:`t0`, :math:`t1` and :math:`n\_frames` (
            ``time_vector = np.linspace(t0, t1, n_frames)``). If :math:`t0 > t1`,
            the time vector is created in reverse order, giving a "backwards"
            animation.

        Args:
            :attr:`t0` (float):
            The first time at which to evaluate the function that gives the
            affine transformation matrix.

            :attr:`t1` (float, optional):
            The last time at which to evaluate the function. If :attr:`None`,
            the function is evaluated at the time :attr:`t0`. Default: :attr:`None`.

            :attr:`n_frames` (int, optional):
            The number of frames in the animation. Default: 1.

        Returns:
            :attr:`inv_mat_frames` (torch.tensor):
            A batch of affine transformation matrices of shape :math:`(n\_frames,3,3)`,
            with dtype :attr:`torch.float32`.

        Example 1: Evaluate the affine transformation matrix between t0=0 and t1 = 10, with 11 frames
            >>> def v(t):
            ...     return torch.tensor([[1-t/10, 0, 0], [0, 1-t/10, 0], [0, 0, 1]])
            >>> field = AffineDeformationField(v, align_corners=False)
            >>> inv_mat_frames = field.get_inv_mat_frames(0, 10, 11)
            >>> print(inv_mat_frames.size())
            torch.Size([11, 3, 3])

        Example 2: Evaluate the affine transformation matrix at t0=4.5, with 1 frame
            >>> def v(t):
            ...     return torch.tensor([[1-t/10, 0, 0], [0, 1-t/10, 0], [0, 0, 1]])
            >>> field = AffineDeformationField(v, align_corners=False)
            >>> inv_mat_frames = field.get_inv_mat_frames(4.5)
            >>> print(inv_mat_frames.size())
            torch.Size([1, 3, 3])
        """
        if t1 is None:
            return self.inverse_field_matrix(t0).unsqueeze(0)

        n_frames = operator.index(n_frames)
        time_vector = np.linspace(t0, t1, n_frames)
        inv_mat_frames = torch.zeros((n_frames, 3, 3), dtype=torch.float32)

        for i, t in enumerate(time_vector):
            inv_mat_frames[i] = self.inverse_field_matrix(t)
        return inv_mat_frames

    def save_inv_grid_frames(
        self, inv_mat_frames: torch.tensor, size: torch.Size
    ) -> torch.tensor:
        r"""
        Saves as a class attribute the inverse deformation field :math:`v`.

        Saves as a class attribute in :attr:`self.inverse_grid_frames` the
        *inverse deformation field* :math:`v` computed from the inverse of the affine
        transformation matrices, evaluated at multiple times.

        .. note::
            The name of the attribute (:attr:`inverse_grid_frames`) is the
            same as the attribute of the parent class :class:`DeformationField`.
            This allows both classes to use the same method to warp images.

        Args:
            :attr:`inv_mat_frames` (torch.tensor):
            shape :math:`(n\_frames,3,3)`
            Batch of inverse affine transformation matrices.

            :attr:`size` (torch.Size):
            shape :math:`(n\_frames,c,Nx,Ny)`
            Target output image size. It is a 4-tuple of integers, where
            :math:`n\_frames` is the number of frames in the animation,
            :math:`c` is the number of channels, and :math:`Nx` and :math:`Ny`
            are the number of pixels along the x-axis and y-axis respectively.
            The number of channels is usually 1 (grayscale) or 3 (color).

        Returns:
            :attr:`self.inverse_grid_frames`:
            :attr:`self.inverse_grid_frames` has the following
            shape: :math:`(n\_frames,Nx,Ny,2)`.

        Example: Save the *inverse deformation field* from a rotation matrix, for a RGB 128x128 image
            >>> def s(t):
            ...     return np.sin(2*np.pi*t)
            >>> def c(t):
            ...     return np.cos(2*np.pi*t)
            >>> def v(t):
            ...     return torch.tensor([[c(t), -s(t), 0], [s(t), c(t), 0], [0, 0, 1]])
            >>> field = AffineDeformationField(v, align_corners=False)
            >>> inv_mat_frames = field.get_inv_mat_frames(0, 10, 101)
            >>> field.save_inv_grid_frames(inv_mat_frames, [101, 3, 128, 128])
            >>> print(field.inverse_grid_frames.size())
            torch.Size([101, 128, 128, 2])
        """
        # affine_grid needs the first two rows
        theta = inv_mat_frames[:, :2, :].float()
        inv_grid_frames = nn.functional.affine_grid(
            theta, size, align_corners=self.align_corners
        )
        self.inverse_grid_frames = inv_grid_frames
        return self.inverse_grid_frames

    def warp(
        self,
        img: torch.tensor,  # single image (1|3, Nx, Ny)
        t0: float,
        t1: float = None,
        n_frames: int = None,
        fps: float = None,
        mode: str = "bilinear",
    ) -> torch.tensor:
        r"""
        Warps an image according to the time interpolation parameters.

        Similarly to the method :meth:`DeformationField.warp` from the parent
        class, it warps a single image according to the *inverse
        deformation field* :math:`v` contained in the attribute
        :attr:`inverse_grid_frames`, between the times :math:`t0` and
        :math:`t1`. The number of frames in the animation is given by either
        :math:`n\_frames`, or :math:`fps` if :math:`n\_frames` is :attr:`None`.

        Args:
            :attr:`img` (torch.tensor):
            Image to deform of shape :math:`(c,Nx,Ny)`, where :math:`c` is the
            number of channels, and :math:`Nx` and
            :math:`Ny` are the number of pixels along the x-axis and y-axis
            respectively. The number of channels is usually 1 (grayscale) or
            3 (color), if not a warning is raised.

            :attr:`t0` (float):
            Start time of the animation.

            :attr:`t1` (float):
            End time of the animation. If :attr:`None`,
            a single frame is warped.

            :attr:`n_frames` (int):
            Number of frames in the animation. If :attr:`None`,
            :attr:`fps` is used to compute the number of frames.

            :attr:`fps` (float):
            Number of frames per second. If :attr:`None`, :attr:`n_frames`
            is used, or a single frame is warped.

            :attr:`mode` (str, optional):
            The interpolation mode to use. It is
            directly passed to the function `torch.nn.functional.grid_sample`.
            It must be one of the following: 'nearest', 'bilinear',
            'bicubic'. Defaults to 'bilinear'.

        Returns:
            :attr:`output` (torch.tensor):
            The deformed batch of images of
            shape :math:`(n\_frames,c,Nx,Ny)`, where each image is
            deformed according to the *inverse deformation field* :math:`v`
            contained in the attribute :attr:`inverse_grid_frames`.

        Example 1: Rotate a single image by 90° counter-clockwise
            >>> def v(t):
            ...     return torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            >>> field = AffineDeformationField(v, align_corners=False)
            >>> image = torch.tensor([[[0. , 0.3],
                                       [0.7, 1. ]]])
            >>> deformed_image = field.warp(image, 0)
            >>> print(deformed_image)
            tensor([[[[0.3000, 1.0000],
                      [0.0000, 0.7000]]]])

        Example 2: Animate an image between t0=0 and t1=1, with 5 frames, with counter-clockwise rotation
            >>> def s(t):
            ...     return np.sin(2*np.pi*t)
            >>> def c(t):
            ...     return np.cos(2*np.pi*t)
            >>> def v(t):
            ...     return torch.tensor([[c(t), -s(t), 0], [s(t), c(t), 0], [0, 0, 1]])
            >>> field = AffineDeformationField(v, align_corners=False)
            >>> image = torch.tensor([[[0. , 0.3],
                                       [0.7, 1. ]]])
            >>> deformed_image = field.warp(image, 0, 1, 5)
            >>> print(deformed_image)
            tensor([[[[0.0000, 0.3000],
                      [0.7000, 1.0000]]],

                    [[[0.3000, 1.0000],
                      [0.0000, 0.7000]]],

                    [[[1.0000, 0.7000],
                      [0.3000, 0.0000]]],

                    [[[0.7000, 0.0000],
                      [1.0000, 0.3000]]],

                    [[[0.0000, 0.3000],
                      [0.7000, 1.0000]]]])
        """
        t0, t1, n_frames = format_params(t0, t1, n_frames, fps)
        inv_mat_frames = self.get_inv_mat_frames(t0, t1, n_frames)
        self.save_inv_grid_frames(inv_mat_frames, [n_frames, *img.size()])
        return super().warp(img, 0, n_frames, mode=mode)

    def __repr__(self):
        s = (
            f"AffineDeformationField({self.inverse_field_matrix.__name__=}, "
            + f"{self.align_corners=}, {self.inverse_grid_frames=})"
        )
        return s


# =============================================================================
# FUNCTIONS
# =============================================================================


def format_params(self, t0: float, t1: float, n_frames: int, fps: float) -> tuple:
    r"""
    Returns the corrected parameters :attr:`t0`, :attr:`t1` and :attr:`n_frames`

    Returns the parameters :attr:`t0`, :attr:`t1` and :attr:`n_frames` in a
    format that can be used to animate an image in multiple frames or warp
    an image in a single frame.

    .. note::
        The parameters :attr:`n_frames` and :attr:`fps` are mutually
        exclusive. If both are given, :attr:`n_frames` is used. If both are
        :attr:`None`, a single frame is warped.

    .. note::
        If :attr:`fps` is given and :attr:`n_frames` is :attr:`None`, the
        end time :attr:`t1` is truncated to the closest lowest multiple of
        :math:`1/\text{fps}`. The number of frames is then computed as
        usual between :attr:`t0` and the truncated :attr:`t1`.

    Args:
        :attr:`t0` (float):
        Start time of the animation.

        :attr:`t1` (float):
        End time of the animation. If :attr:`None`, a single frame is warped.

        :attr:`n_frames` (int):
        Number of frames in the animation. If :attr:`None`, a single frame
        is warped.

        :attr:`fps` (float):
        Number of frames per second. If :attr:`None`, a single frame is warped.

    Returns:
        :attr:`(t0, t1, n_frames)` (tuple):
        Where :attr:`t0` (float) and :attr:`t1` (float) are the start and
        end time of the animation respectively, and :attr:`n_frames` (int)
        is the number of frames in the animation.

    The output follows this pattern::

        Is :attr:`t1` :attr:`None`?
        ├── Yes
        │    └── Are :attr:`n_frames` and :attr:`fps` :attr:`None`?
        │         ├── Yes ──> :attr:`(t0, t0, 1)`: a single frame is warped.
        │         └── No ──> :attr:`ValueError` (conflicting parameters).
        └── No
            └── Is :attr:`n_frames` :attr:`None`?
                ├── Yes
                │    └── Is :attr:`fps` :attr:`None`?
                │         ├── Yes ──> :attr:`ValueError` (conflicting parameters).
                │         └── No ─┬─ :attr:`n_frames = int(np.floor(1 + (t1-t0) * fps))`
                │                 └──> :attr:`(t0, t0 + (n_frames-1) / fps, n_frames)`
                └── No ──> :attr:`(t0, t1, n_frames)`

    Example 1: Warp a single image at t0=0, with 1 frame
        >>> field = AffineDeformationField(None, align_corners=False)
        >>> field.format_params(0)
        (0, 0, 1)

    Example 2: Warp a single image between t0=0 and t1=1, with 11 frames
        >>> field = AffineDeformationField(None, align_corners=False)
        >>> field.format_params(0, 1, 11)
        (0, 1, 11)

    Example 3: Warp a single image between t0=0 and t1=1 at 24 fps
        >>> field = AffineDeformationField(None, align_corners=False)
        >>> field.format_params(0, 1, None, 24)
        (0, 1, 25)

    Example 4: Provide n_frames and fps
        >>> field = AffineDeformationField(None, align_corners=False)
        >>> field.format_params(0, 1, 11, 24)
        (0, 1, 11)

    Example 5: Provide conflicting parameters
        >>> field = AffineDeformationField(None, align_corners=False)
        >>> field.format_params(0, 1, None, None)
        ValueError: Unable to animate multiple frames: t1 (1) was given, but fps and n_frames were not given.

    Example 6: Provide conflicting parameters
        >>> field = AffineDeformationField(None, align_corners=False)
        >>> field.format_params(0, None, 11, 24)
        ValueError: Unable to animate multiple frames: t1 was not given, but n_frames (11) or fps (24) or were given.
    """
    if t1 is None:
        if (fps is None) and (n_frames is None):
            return (t0, t0, 1)  # no animation
        else:
            raise ValueError(
                "Unable to animate multiple frames: t1 was not given, "
                + f"but n_frames ({n_frames}) or fps ({fps}) were given."
            )
    else:
        # if fps and n_frames are given, use n_frames
        if n_frames is None:
            if fps is None:
                raise ValueError(
                    f"Unable to warp one image: t1 ({t1}) was given, "
                    + "but fps and n_frames were not given."
                )
            else:
                # t1 is truncated to the closest lowest multiple of 1/fps
                n_frames = int(np.floor(1 + (t1 - t0) * fps))
                return (t0, t0 + (n_frames - 1) / fps, n_frames)
        else:
            return (t0, t1, n_frames)
