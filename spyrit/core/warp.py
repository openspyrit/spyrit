"""
This module contains classes that are used to warp images according to
a deformation field. Let :math:`t_0 \in \mathbb{R_+}`,
:math:`f(t_0, x, y): \mathbb{R}^2 \mapsto \mathbb{R}^2` be a reference scene
and :math:`u(t, x, y): \mathbb{R}^3 \mapsto \mathbb{R}^2` be a deformation field. 
These classes compute the moving scene:

.. math::
    f(t, x, y) = f(t_0, u(t, x, y))

.. note::
    These classes rely on backward mapping to perform the warping,
    and therefore store the *inverse deformation field* :math:`v=u^{-1}` as a class attribute
    instead of the *deformation field* :math:`u`.
"""

import warnings

import math
import torch
import torch.nn as nn
from torchvision.transforms import v2


# =============================================================================
class DeformationField(nn.Module):
    # =========================================================================
    r""" Stores a discrete deformation field :math:`v` of shape :math:`(n\_frames,h,w,2)`.
    
    The deformations has :math:`n\_frames` is the number of frames, and its height and width
    are denoted by :math:`h` and :math:`w`. The last dimension contains the x
    and y coordinates of the deformation field w.r.t the reference time :math:`t_0`.

    .. math::
        f(t, x, y) = f(t_0, u(t, x, y))

    where :math:`f(t_0, x, y)` is the reference image and :math:`u(t, x, y)` is the *deformation field*.

    Forward call generates a video warping the input image according to the deformation field :math:`u = v^{-1}`.

    .. important::
        The coordinates are given in the range [-1, 1]. When referring to a
        pixel, its position is the position of its center. The position
        (-1, -1) corresponds to the center of the top-left pixel.

    .. note::
        This class relies on backward mapping,
        therefore the input deformation field must be the *inverse deformation field* :math:`v=u^{-1}`
        that maps the *deformed image* pixel coordinates to the *original image* pixel coordinates.

    Args:
        :attr:`field` (torch.tensor):
        *Inverse deformation field* :math:`v` of shape :math:`(n\_frames,h,w,2)`,
        where :math:`n\_frames` is the number of deformation frames, :math:`h` and :math:`w` 
        are the height and width of the deformation field. 
        For accuracy reasons, it is recommended the dtype to be `torch.float64`.

    Attributes:
        :attr:`self.field` (torch.tensor):
        *Inverse deformation field* :math:`v` of shape :math:`(n\_frames,h,w,2)`.

        :attr:`self.n_frames` (int): Number of frames in the video.

        :attr:`self.img_shape` (tuple): Shape of the image to be warped, i.e.
        :math:`(h,w)`, where :math:`h` and :math:`w` are the height and width
        of the image respectively.

        :attr:`img_h` (int): Height of the image to be warped in pixels.

        :attr:`img_w` (int): Width of the image to be warped in pixels.

        :attr:`self.align_corners` (bool): Always True. This argument is passed
        to the functions :func:`torch.nn.functional.grid_sample` and
        :func:`torch.nn.functional.affine_grid` to ensure the corners of the
        image are aligned with the corners of the grid.

    Example: 
        Storing a 90 degrees counter-clockwise rotation for 2x2 image. 
        >>> v = torch.tensor([[[[ 1, -1], [ 1, 1]], [[-1, -1], [-1, 1]]]])
        >>> field = DeformationField(v)
        >>> print(field.field)
        tensor([[[[ 1, -1],
                  [ 1,  1]],
                 [[-1, -1],
                 [-1,  1]]]])
        >>> print(field.field.shape)
        torch.Size([1, 2, 2, 2])

    """

    def __init__(self, field: torch.tensor):
        super().__init__()

        self._align_corners = True
        self._device_tracker = nn.Parameter(torch.tensor([0.0]), requires_grad=False)

        # field is None if AffineDeformationField is used
        if field is not None:
            # store as nn.Parameter and ensure proper device/dtype handling
            self._field = nn.Parameter(field.detach().clone(), requires_grad=False)
            # Move device tracker to same device as field
            self._device_tracker = nn.Parameter(
                torch.tensor([0.0], device=field.device, dtype=field.dtype), 
                requires_grad=False
            )

        self.warn_range = False  # warn the user if the field goes beyond +/-2
        # self._warn_field()

    @property
    def align_corners(self) -> bool:
        return self._align_corners

    @property
    def n_frames(self) -> int:
        return self._field.shape[0]

    @property
    def img_shape(self) -> tuple:
        return self._field.shape[1:3]

    @property
    def img_h(self) -> int:
        return self._field.shape[1]

    @property
    def img_w(self) -> int:
        return self._field.shape[2]

    @property
    def field(self) -> torch.tensor:
        return self._field.data

    @property
    def device(self) -> torch.device:
        return self._device_tracker.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the deformation field."""
        return self._field.dtype if hasattr(self, '_field') else self._device_tracker.dtype

    def forward(
        self,
        img: torch.tensor,
        n0: int = 0,
        n1: int = None,
        mode: str = "bilinear",
    ) -> torch.tensor:
        r"""
        Generates a video from a batch of 2D images according to the 
        *deformation field* :math:`u=v^{-1}`.

        The deformation is taken between the frames :math:`n0` (included) 
        and :math:`n1` (excluded).

        Args:
            :attr:`img` (torch.tensor):
            Batch of 2D images to deform of shape :math:`(c, h, w)` or :math:`(b, c, h, w)`, where
            :math:`b` is the number of images in the batch, :math:`c` is the
            number of channels, and :math:`h` and :math:`w`
            are the height and width of the images.

            :attr:`n0` (int, optional):
            The index of the first frame to use in the *inverse deformation
            field*. Defaults to 0.

            :attr:`n1` (int, optional):
            The index of the first frame to exclude in the *inverse deformation
            field*. If None, the last available frame is used. Defaults to None.

            :attr:`mode` (str, optional):
            The interpolation mode to use. It must be one of the following:
            'nearest', 'bilinear', 'bicubic', 'biquintic'. The `nearest`,
            `bilinear`, and `bicubic` modes are directly supported by the
            function :func:`torch.nn.functional.grid_sample`. The `biquintic` mode
            relies on scikit-image. Defaults to 'bilinear'.

        .. note::
            If using mode='bicubic' or mode='biquintic', the warped image may
            contain values outside the original range.

        .. note::
            If :math:`n0 < n1`, :attr:`field` is sliced
            as follows: ``field[n0:n1, :, :, :]``

        .. note::
            If :math:`n0 > n1`, :attr:`field` is sliced
            "backwards". The first frame of the warped animation corresponds to
            the index :math:`n0`, and the last frame corresponds to the index
            :math:`n1+1`. This behavior is identical to slicing a list with a
            step of -1.

        .. note::
            If the number of pixels is different in the image and the field,
            the field is interpolated to match the image size (see the 
            behavior of :func:`torch.nn.functional.grid_sample`).

        .. note::
            If the input image three dimensional, a batch dimension is added in the first dimension. 

        Returns:
            :attr:`output` (torch.tensor):
            The deformed batch of 2D images of shape :math:`(|n1-n0|, c, h, w)`
            or :math:`(b, |n1-n0|, c, h, w)` depending on the input shape, where each
            image in the batch is deformed according to the *deformation
            field* :math:`u`, whose inverse :math:`v` is stored in the attribute :attr:`field`.

        Example: 
            Rotating a 2x2 grayscale image by 90 degrees counter-clockwise, using one frame:

            >>> v = torch.tensor([[[[ 1., -1.], [ 1., 1.]], [[-1., -1.], [-1., 1.]]]])
            >>> field = DeformationField(v)
            >>> image = torch.tensor([0., 0.3, 0.7, 1.]).view(1, 1, 2, 2)
            >>> print(image)
            tensor([[[[0.0000, 0.3000],
                    [0.7000, 1.0000]]]])
            >>> deformed_image = field(image, 0, 1)
            >>> print(deformed_image)
            tensor([[[[0.3000, 1.0000],
                    [0.0000, 0.7000]]]])
        """

        if img.ndim == 3:
            img = img.unsqueeze(0)
            no_batch = True
        else:
            no_batch = False
        # check that the input is shaped (b, c, h, w)
        b, c, h, w = img.shape

        if n1 is None:
            n1 = self.n_frames

        # get the right slice of the inverse deformation field
        n_frames = abs(n1 - n0)
        if n1 < n0:
            sel_inv_grid_frames = torch.flip(self.field[n1 + 1 : n0 + 1, :, :, :], [0])
        else:
            sel_inv_grid_frames = self.field[n0:n1, :, :, :]

        # Ensure the grid is on the same device as the input image
        sel_inv_grid_frames = sel_inv_grid_frames.to(device=img.device)

        # img has current shape (b, c, h, w), make it (n_frames, b*c, h, w)
        # because grid_sample will create the frames in the batch dimension
        img_frames = img.reshape(1, b * c, h, w).expand(n_frames, -1, -1, -1)

        # Ensure dtype compatibility for grid_sample
        warped_frames = self._grid_sample(
            img_frames, sel_inv_grid_frames, mode
        )
        # has shape (n_frames, b*c, h, w), make it (b, n_frames, c, h, w)
        warped_frames = warped_frames.reshape(n_frames, b, c, h, w).moveaxis(0, 1)
        if no_batch:
            return warped_frames.squeeze(0)
        return warped_frames

    def _grid_sample(self, img_frames, inverse_grid_frames, mode):
        """Warp frames of 2D images with a deformation field. 
        Each image of the collection will get a different deformation. 
        This function matches the behavior of nn.functional.grid_sample.

        Inputs:
            :attr:`img_frames` (torch.tensor): batch of 2D images of shape
            `(n_frames, c, h, w)`, where `n_frames` is the number of frames in
            the animation, `c` is the number of channels, and `h` and `w` are
            the height and width of the image respectively.

            :attr:`inverse_grid_frames` (torch.tensor): batch of inverse
            deformation fields of shape `(n_frames, h, w, 2)`, indicating the
            pixel coordinates of the original image that are displayed in the
            warped image.

            :attr:`mode` (str): The interpolation mode to use. It must be one of
            the following: 'nearest', 'bilinear', 'bicubic', 'biquintic'. If either
            `nearest`, `bilinear`, or `bicubic`, it is directly passed to the
            function :func:`torch.nn.functional.grid_sample`. If `biquintic`,
            it is passed to the package scikit-image, which requires skimage and numpy.

        Returns:
            :attr:`out` (torch.tensor): The deformed batch of 2D images of shape
            `(n_frames, c, h, w)`. Each image in the batch is deformed according
            to the inverse deformation field :math:`v` contained in the attribute
            :attr:`field`.
        """
        if mode == "biquintic":
            import skimage
            import numpy as np

            n_frames, c, h, w = img_frames.shape
            out = np.empty((n_frames, c, h, w))

            # use scikit-image's order 5 warp. This implies:
            # putting the origin pixel coordinate (x,y) at dimension 0, not 3
            # using numpy instead of pytorch
            inverse_grid_frames = inverse_grid_frames.moveaxis(-1, 0).cpu().numpy()
            # changing from 'xy' notation to 'ij'
            inverse_grid_frames = inverse_grid_frames[::-1, :, :, :]
            # rescaling from [-1, 1] to [0, height-1] (same for width)
            inverse_grid_frames = (
                (inverse_grid_frames + 1)
                / 2
                * np.array([self.img_h, self.img_w]).reshape(2, 1, 1, 1)
            )

            # use 2 for loops, faster than 5D warp (because 5D interpolation)
            for frame in range(n_frames):
                inverse_grid = inverse_grid_frames[:, frame, :, :]

                for channel in range(c):
                    out[frame, channel, :, :] = skimage.transform.warp(
                        img_frames[frame, channel, :, :].cpu().numpy(),
                        inverse_grid,
                        order=5,
                        clip=False,
                    )

            return torch.from_numpy(out).to(device=img_frames.device, dtype=img_frames.dtype)

        else:
            # Ensure both tensors are on the same device and compatible dtypes
            if img_frames.device != inverse_grid_frames.device:
                inverse_grid_frames = inverse_grid_frames.to(device=img_frames.device)
            
            # For grid_sample, we need to ensure the grid is float32 or float64
            if inverse_grid_frames.dtype not in [torch.float32, torch.float64]:
                inverse_grid_frames = inverse_grid_frames.float()
            
            out = nn.functional.grid_sample(
                img_frames,
                inverse_grid_frames,
                mode=mode,
                padding_mode="zeros",
                align_corners=self.align_corners,
            )

            return out  # has shape (n_frames, c, h, w)

    def _warn_field(self):
        # using float64 is preferred for accuracy
        if self.field.dtype == torch.float32:
            if self.__class__ == DeformationField:
                msg = "Consider using float64 when storing the deformation field for greater accuracy."
            if self.__class__ == AffineDeformationField:
                msg = "Consider using float64 when defining the output type of the affine transformation matrix :attr:`func` for greater accuracy."
            warnings.warn(msg, UserWarning)

        # if the field goes bayond +/-2, warn the user
        if self.warn_range and (self.field.abs() > 2).any():
            msg = "The deformation field goes beyond the range [-2;2], everything mapped outside [-1;1] will not be visible. Suppress this warning by setting self.warn_range = False."
            warnings.warn(msg, UserWarning)

    def _attributeslist(self):
        a = [
            ("field shape", self.field.shape),
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

    def __hash__(self) -> int:
        return hash(self.field)


# =============================================================================
class AffineDeformationField(DeformationField):
    # =========================================================================
    r"""Stores and applies affine deformation fields defined by transformation matrices.
    
    This class generates video sequences by warping images according to time-varying affine 
    transformations. It constructs a discrete *inverse deformation field* :math:`v` from 
    a user-defined function that returns 3x3 affine transformation matrices at different time points.
    
    The forward call generates a video warping the input image according to the deformation field :math:`u = v^{-1}`.

    .. math::
        f(t, x, y) = f(t_0, u(t, x, y))

    where :math:`f(t_0, x, y)` is the reference image and :math:`u(t, x, y)` is the *deformation field*.

    .. important::
        The coordinates are given in the range [-1, 1]. When referring to a
        pixel, its position is the position of its center. The position
        (-1, -1) corresponds to the center of the top-left pixel.

    .. note::
        This class relies on backward mapping,
        therefore the input transformation function must define the *inverse deformation field* :math:`v=u^{-1}`
        that maps the *deformed image* pixel coordinates to the *original image* pixel coordinates.

    .. note::
        The image size is requested upon construction, but the warping can be done
        with images of different sizes. The grid is simply interpolated to match
        the image size. It is also possible to change the image size after
        construction by setting the attribute :attr:`img_shape`, or the attributes
        :attr:`img_h` and :attr:`img_w`.

    .. important::
        TODO: Check with Nicolas: the function used to generate fields dynamically, I felt that it was unnecessary
        and more ressource demanding.

    Args:
        :attr:`func` (Callable[[float], torch.tensor]): Function of one parameter (time) that returns a tensor 
        of shape :math:`(3,3)` representing an affine homogeneous transformation matrix. This matrix corresponds 
        to the *inverse deformation field* :math:`v`, i.e. the field that maps the pixels of the *deformed image* 
        to the pixels of the *original image*.

        :attr:`time_vector` (torch.tensor): Vector of time points at which the transformation function is evaluated
        to generate the inverse deformation field. Shape :math:`(n\_frames,)`.

        :attr:`img_shape` (tuple): Shape of the image to be warped, i.e. :math:`(h,w)`, where :math:`h` and :math:`w`
        are the height and width of the image respectively.

        :attr:`dtype` (torch.dtype, optional): Data type of the deformation field tensor. 
        For accuracy reasons, it is recommended to use `torch.float64`. 
        Defaults to `torch.float32`.

        :attr:`device` (torch.device, optional): Device on which the deformation field tensor is stored.
        Defaults to `torch.device('cpu')`.

    Attributes:
        :attr:`self.func` (Callable[[float], torch.tensor]): Function of one parameter (time) that returns a tensor of shape
        :math:`(3,3)` representing an affine homogeneous transformation matrix.

        :attr:`self.field` (torch.tensor):*Inverse deformation field* :math:`v` of shape :math:`(n\_frames,h,w,2)`.

        :attr:`self.time_vector` (torch.tensor): Vector of time points at which the function is evaluated to generate 
        the inverse deformation field.

        :attr:`self.n_frames` (int): Number of frames in the video.

        :attr:`self.img_shape` (tuple): Shape of the image to be warped, i.e. :math:`(h,w)`, where :math:`h` and :math:`w` 
        are the height and width of the image respectively.

        :attr:`self.img_h` (int): Height of the image to be warped in pixels.

        :attr:`self.img_w` (int): Width of the image to be warped in pixels.

        :attr:`self.align_corners` (bool): Always True. This argument is passed to the functions 
        :func:`torch.nn.functional.grid_sample` and :func:`torch.nn.functional.affine_grid` 
        to ensure the corners of the image are aligned with the corners of the grid.

    Example 1: Progressive scaling
        >>> def scaling(t):
        ...     scale = 1 - t/10
        ...     return torch.tensor([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        >>> time_vector = torch.linspace(0, 1, 10)
        >>> def_field = AffineDeformationField(scaling, time_vector, (64, 64))
        >>> print(def_field.n_frames)
        10

    Example 2: Rotation counter-clockwise at 1Hz frequency
        >>> def rotation(t):
        ...     angle = 2 * np.pi * t  # 1Hz rotation
        ...     c, s = np.cos(angle), np.sin(angle)
        ...     return torch.tensor([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=torch.float64)
        >>> time_vector = torch.linspace(0, 1, 30)  # 30 frames for 1 second
        >>> def_field = AffineDeformationField(rotation, time_vector, (128, 128))
        >>> print(def_field.field.shape)
        torch.Size([30, 128, 128, 2])
        
    """

    def __init__(
        self,
        func,
        time_vector: torch.tensor,
        img_shape: tuple,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
    ) -> None:

        self._align_corners = True  
        self.func = func
        self.time_vector = time_vector

        field = self._generate_inv_grid_frames(img_shape, time_vector, func, dtype=dtype, device=device)

        super().__init__(field)


    def _generate_inv_grid_frames(
        self,
        grid_shape: tuple,
        time_vector: torch.tensor,
        func,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu')
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

            n_frames (int): Number of frames in the video.

            grid_shape (tuple): shape of the 2D grid to be generated. Must be a
            tuple of the form (h, w), where h and w are respectively the height
            and width of the image to be warped.

        Returns:
            torch.tensor: The inverse deformation field as a tensor of shape
            :math:`(n\_frames, h, w, 2)`.
        """
        # get a batch of matrices of shape (n_frames, 2, 3)
        inv_mat_frames = torch.stack(
            [
                func(t.item())[:2, :]  # need only the first 2 rows
                for t in time_vector
            ]
        )
        
        # Ensure matrices are on the correct device and dtype
        inv_mat_frames = inv_mat_frames.to(dtype=dtype, device=device)

        # use them to generate the grid
        inv_grid_frames = nn.functional.affine_grid(
            inv_mat_frames,
            torch.Size(
                (len(time_vector), 1, *grid_shape)
            ),  # n_channels has no effect
            align_corners=self._align_corners,
        )
        return inv_grid_frames

    def forward(
        self,
        img: torch.tensor,
        n0: int = 0,
        n1: int = None,
        mode: str = "bilinear",
    ) -> torch.tensor:
        r"""Generates a video from a batch of 2D images according to the *deformation field* :math:`u=v^{-1}`.

        The deformation is taken between the frames :math:`n0` (included) 
        and :math:`n1` (excluded).

        Args:
            :attr:`img` (torch.tensor):
            Batch of 2D images to deform of shape :math:`(c, h, w)` or :math:`(b, c, h, w)`, where
            :math:`b` is the number of images in the batch, :math:`c` is the
            number of channels, and :math:`h` and :math:`w`
            are the height and width of the images.

            :attr:`n0` (int, optional):
            The index of the first frame to use in the *inverse deformation
            field*. Defaults to 0.

            :attr:`n1` (int, optional):
            The index of the first frame to exclude in the *inverse deformation
            field*. If None, the last available frame is used. Defaults to None.

            :attr:`mode` (str, optional):
            The interpolation mode to use. It must be one of the following:
            'nearest', 'bilinear', 'bicubic', 'biquintic'. The `nearest`,
            `bilinear`, and `bicubic` modes are directly supported by the
            function :func:`torch.nn.functional.grid_sample`. The `biquintic` mode
            relies on scikit-image. Defaults to 'bilinear'.

        .. note::
            If using mode='bicubic' or mode='biquintic', the warped image may
            contain values outside the original range.

        .. note::
            If :math:`n0 < n1`, :attr:`field` is sliced
            as follows: ``field[n0:n1, :, :, :]``

        .. note::
            If :math:`n0 > n1`, :attr:`field` is sliced
            "backwards". The first frame of the warped animation corresponds to
            the index :math:`n0`, and the last frame corresponds to the index
            :math:`n1+1`. This behavior is identical to slicing a list with a
            step of -1.

        .. note::
            If the number of pixels is different in the image and the field,
            the field is interpolated to match the image size (see the 
            behavior of :func:`torch.nn.functional.grid_sample`).

        Returns:
            :attr:`output` (torch.tensor):
            The deformed batch of 2D images of shape :math:`(|n1-n0|, c, h, w)`
            or :math:`(b, |n1-n0|, c, h, w)` depending on the input shape, where each
            image in the batch is deformed according to the *deformation
            field* :math:`u`, whose inverse :math:`v` is stored in the attribute :attr:`field`.

        **Example 1:** Progressive scaling
            >>> def scaling(t):
            ...     scale = 1 - t/10
            ...     return torch.tensor([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
            >>> time_vector = torch.linspace(0, 1, 10)
            >>> def_field = AffineDeformationField(scaling, time_vector, (64, 64))
            >>> images = torch.randn(16, 1, 64, 64)  # Batch of 16 grayscale image
            >>> scaled_video = def_field(images)
            >>> print(scaled_video.shape)
            torch.Size([16, 10, 1, 64, 64])

        **Example 2:** Rotation counter-clockwise at 1Hz frequency
            >>> def rotation(t):
            ...     angle = 2 * np.pi * t  # 1Hz rotation
            ...     c, s = np.cos(angle), np.sin(angle)
            ...     return torch.tensor([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=torch.float64)
            >>> time_vector = torch.linspace(0, 1, 30)  # 30 frames for 1 second
            >>> def_field = AffineDeformationField(rotation, time_vector, (128, 128), dtype=torch.float64)
            >>> image = torch.randn(3, 128, 128).to(dtype=torch.float64)  # a single RGB image
            >>> rotated_video = def_field(image)
            >>> print(rotated_video.shape)
            torch.Size([30, 3, 128, 128])
        """

        return super().forward(img, n0, n1, mode)


# =============================================================================
class ElasticDeformation(DeformationField):
    r"""Generates and stores a random elastic deformation where each pixel is
    sampled from an uniform distribution and then smoothed in space and time.

    This class inherits from the random generation of TorchVision's
    :class:`torchvision.transforms.v2.ElasticTransform`. It will generate several
    frames of static elastic deformation using the torchvision class, and then
    smooth out these frames in the time domain to create a continuous animation.
    The deformation field is generated at instantiation and stored as a class
    attribute.

    The spatial magnitude of the displacements is controlled by the parameter
    :attr:`alpha`, the spatial smoothness of the displacements is controlled by
    the parameter :attr:`sigma`, and the time-domain smoothness is controlled by
    the parameter :attr:`n_interpolation`.

    .. note::
        The spatial and temporal smoothing are done **after** the
        displacements of magnitude :attr:`alpha` are generated. This means that
        the actual spatial displacement magnitude might be significantly lower
        than then one specified by :attr:`alpha`. To get the actual standard
        deviation of the deformation field, call the :meth:`compute_field_std` method.

    .. note::
        The parameters :attr:`alpha`, :attr:`sigma`, and :attr:`n_interpolation`
        are defined at initialization and cannot be changed after instantiation.

    Args:
        alpha (float): Magnitude of displacements. This argument is passed to
        the constructor of :class:`torchvision.transforms.v2.ElasticTransform`.

        sigma (float): Smoothness of displacements in the spatial domain. This
        argument is passed to the constructor of 
        :class:`torchvision.transforms.v2.ElasticTransform`.

        img_shape (tuple): Shape of the deformation field, i.e. :math:`(h,w)`,
        where :math:`h` and :math:`w` are the height and width of the field
        respectively.

        n_frames (int): Number of frames in the video.

        n_interpolation (int): Period in frames of the time-domain interpolation.
        Every :attr:`n_interpolation` frames, a 2D elastic transform is randomly
        generated. Between these frames, the deformation field is equal to the
        identity. A truncated gaussian smoothing of length equal to 3 times
        :attr:`n_interpolation` (to capture a real-looking movement between 3
        points in 2D space) and with a standard deviation of :math:`\frac{3}{4}`
        :attr:`n_interpolation` is applied to the deformation field.

        dtype (torch.dtype): Data type of the tensors. Default is torch.float32.

        device (torch.device): Device on which the tensors are stored. Default is CPU.

    Attributes:
        :attr:`field` (torch.tensor): The inverse deformation field as a tensor of shape
        :math:`(n\_frames,h,w,2)`.

        :attr:`img_shape` (tuple): Shape of the deformation field, i.e. :math:`(h,w)`,
        where :math:`h` and :math:`w` are the height and width of the field
        respectively.

        :attr:`n_frames` (int): Number of frames in the animation.

        :attr:`alpha` (float): Magnitude of displacements.

        :attr:`sigma` (float): Smoothness of displacements in the spatial domain.

        :attr:`n_interpolation` (int): Period in frames of the time-domain interpolation.

        :attr:`ElasticTransform` (torchvision.transforms.v2.ElasticTransform): The
        random generator of static elastic deformation, with parameters :attr:`alpha`
        and :attr:`sigma`.
    """

    def __init__(
        self, 
        alpha, 
        sigma, 
        img_shape, 
        n_frames, 
        n_interpolation, 
        dtype=torch.float32,
        device=torch.device('cpu')
    ): 

        field = self._generate_inv_grid_frames(img_shape, n_frames, n_interpolation, alpha, sigma, dtype, device)
        field = field.to(dtype=dtype, device=device)
        
        super().__init__(field)

        # Set additional attributes (after init)
        self.alpha = alpha
        self.sigma = sigma
        self.n_interpolation = n_interpolation
        self.ElasticTransform = v2.ElasticTransform(alpha, sigma)

    def compute_field_std(self):
        r"""Computes the theoretical standard deviation (in pixels) of the deformation field."""

        sigma_t = 3 * self.n_interpolation / 4
        var_dz = 1 / 3
        var_gdz = var_dz / (4 * math.pi * self.sigma ** 2)
        std = self.alpha * (var_gdz / (2 * math.pi ** 0.5 * sigma_t)) ** 0.5

        return std


    def _generate_inv_grid_frames(self, img_shape, n_frames, n_interpolation, alpha, sigma, dtype, device):
        r"""Generates the frames of the elastic deformation field of shape :math:`(n_frames, h, w, 2)`. """

        # create base frame between -1 and 1
        base_frame_i = torch.linspace(-1, 1, img_shape[0], dtype=dtype, device=device)
        base_frame_j = torch.linspace(-1, 1, img_shape[1], dtype=dtype, device=device)

        # shape (h, w, 2)
        base_frame = torch.stack(
            torch.meshgrid(base_frame_i, base_frame_j, indexing="xy"), dim=-1
        )
        window_width = n_interpolation * 3

        elastic_frames_to_generate = 1 + int(math.ceil(n_frames / n_interpolation))
        total_frames_after_conv = 1 + (elastic_frames_to_generate - 1) * n_interpolation
        # account for the window width
        total_frames_to_generate = total_frames_after_conv + window_width - 1
        grid = base_frame.repeat(total_frames_to_generate, 1, 1, 1)

        for i in range(total_frames_to_generate // n_interpolation):
            # generate a random field - create tensor on correct device
            dummy_input = torch.empty([1, *img_shape], device=device, dtype=dtype)
            # Note: ElasticTransform needs to be created with the correct parameters
            elastic_transform = v2.ElasticTransform(alpha, sigma)
            displacement = elastic_transform._get_params(dummy_input)["displacement"][0, :, :, :]
            grid[i * n_interpolation] += displacement.to(device=device, dtype=dtype)

        # Define Gaussian convolution operator
        Conv = nn.Conv1d(1, 1, window_width, bias=False, padding=0)
        gaussian_window = torch.signal.windows.gaussian(
            window_width, std=window_width / 4, dtype=dtype, device=device
        )  # , std=self.sigma_time)
        gaussian_window /= gaussian_window.sum()
        Conv.weight = nn.Parameter(gaussian_window.view(1, 1, -1), requires_grad=False)
        
        # Move Conv to correct device
        Conv = Conv.to(device=device)

        # reshape, convolute, reshape back
        grid = grid.permute(1, 2, 3, 0)  # put time in the last dimension
        grid = grid.reshape(-1, 1, total_frames_to_generate)  # (h*w*2, 1, n_frames)
        grid = Conv(grid)
        grid = grid.reshape(*img_shape, 2, total_frames_after_conv)
        grid = grid.permute(3, 0, 1, 2)  # (n_frames, h, w, 2)

        # truncate to the desired number of frames
        return grid[:n_frames, ...]
