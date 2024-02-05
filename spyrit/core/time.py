"""
    Author: Romain Phan
    
    Contains DeformationField and AffineDeformationField, a subclass of the
    former.
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
    Warps a single image according to an inverse deformation field :math:`v`,
    i.e. the field that maps the pixels of the deformed image to the pixels of
    the original image. 
    
    It is constructed from a tensor of shape :math:`(n_frames,Nx,Ny,2)`, where
    :math:`n_frames` is the number of frames in the animation, :math:`Nx` and 
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
        :attr:`inverse_grid_frames` (torch.Tensor): Inverse deformation field :math:`v`
            of shape :math:`(n_frames,Nx,Ny,2)`.
        :attr:`align_corners` (bool, optional): Geometrically, we consider the
            pixels of the input as squares rather than points. If set to 
            :attr:`True`, the extrema (-1 and 1) are considered as referring to
            the center points of the input's corner pixels. If set to
            :attr:`False`, they are instead considered as referring to the
            corner points of the input's corner pixels, making the sampling
            more resolution agnostic. Default: :attr:`False`.
    
    Example 1: Rotating a 2x2 B&W image by 90 degrees counter-clockwise, 
    using one frame and align_corners=False
        >>> frame = torch.tensor([[[[ 0.5, -0.5], [ 0.5, 0.5]],
                                   [[-0.5, -0.5], [-0.5, 0.5]]]])
        >>> field = DeformationField(frame, align_corners=False)
        >>> print(field.inverse_grid_frames) 
        tensor([[[[ 0.5, -0.5], [ 0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]]])
        
    Example 2: Rotating a 2x2 B&W image by 90 degrees clockwise, using one
    frame and align_corners=True
        >>> frame = torch.tensor([[[[-1, 1], [-1, -1]],
                                   [[ 1, 1], [ 1, -1]]]])
        >>> field = DeformationField(frame, align_corners=True)
        >>> print(field.inverse_grid_frames) 
        tensor([[[[-1, 1], [-1, -1]], [[ 1, 1], [ 1, -1]]])
    """
        # >>> image = tensor([[[0. , 0.3],
        #                      [0.7, 1. ]]])
        # >>> rot
    
    def __init__(
            self, 
            inverse_grid_frames: torch.Tensor=None, 
            align_corners=False
            ):
        super().__init__()
        self.inverse_grid_frames = inverse_grid_frames
        self.align_corners = align_corners
        
    def warp(
            self, 
            img: torch.Tensor,
            n0: int,
            n1: int,
            mode: str='bilinear'
            ) -> torch.Tensor:
        r"""
        Deforms the image according to the inverse deformation field :math:`v`
        contained in the attribute :attr:`inverse_grid_frames`, sliced between
        the frames :math:`n0` (included) and :math:`n1` (excluded). 
        
        .. note::
            If :attr:`n0` < :attr:`n1`, :attr:`inverse_grid_frames` is sliced
            as follows: :math:`\text{{inv\_grid\_frames}}[n0:n1, :, :, :]`
        
        .. note::
            If :attr:`n0` > :attr:`n1`, :attr:`inverse_grid_frames` is sliced
            "backwards". The first frame of the warped animation corresponds to
            the index :attr:`n0`, and the last frame corresponds to the index
            :attr:`n1+1`. This behavior is identical to slicing a list with a
            step of -1.

        Args:
            :attr:`img` (torch.Tensor): The image to deform of shape
                :math:`(c,Nx,Ny)`, where :math:`c` is the number of
                channels, and :math:`Nx` and :math:`Ny` are the number of 
                pixels along the x-axis and y-axis respectively. The number of
                channels is usually 1 (grayscale) or 3 (color), if not a
                warning is raised. If the image has not 3 dimensions, an error
                is raised.
            :attr:`n0` (int): The index of the first frame to use in the inverse
                deformation field.
            :attr:`n1` (int): The index of the first frame to exclude in the
                inverse deformation field.
            :attr:`mode` (str, optional): The interpolation mode to use. It is
                directly passed to the function `torch.nn.functional.grid_sample`.
                It must be one of the following: 'nearest', 'bilinear',
                'bicubic'. Defaults to 'bilinear'.
               
        Returns:
            :attr:`output` (torch.Tensor): The deformed batch of images of 
                shape :math:`(\abs(n1-n0),c,Nx,Ny)`, where each image is 
                deformed according to the inverse deformation field :math:`v`
                contained in the attribute :attr:`inverse_grid_frames`.
         
        Shape:
            - :attr:`img`: :math:`(c,Nx,Ny)`, where c is the number of channels,
                Nx and Ny are the number of pixels along the x-axis and y-axis
                respectively.
            - :attr:`output`: :math:`(\abs(n1-n0),c,Nx,Ny)` 
        
        Example 1: Rotating a 2x2 B&W image by 90 degrees clockwise, using one
    frame and align_corners=True
        >>> frame = torch.tensor([[[[-1, 1], [-1, -1]],
                                   [[ 1, 1], [ 1, -1]]]])
        >>> field = DeformationField(frame, align_corners=True)
        >>> image = torch.tensor([[[0. , 0.3], 
                                   [0.7, 1. ]]])
        >>> deformed_image = field.warp(image, 0, 1)
        >>> print(deformed_image)
        tensor([[[[0.7, 0. ], 
                  [1. , 0.3]]]])
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
            sel_inv_grid_frames = \
                torch.flip(self.inverse_grid_frames[n1+1:n0+1, :, :, :], [0])
        else:
            sel_inv_grid_frames = \
                self.inverse_grid_frames[n0:n1, :, :, :].clone()
        
        nb_frames = abs(n1-n0)
        img_frames = img.expand(nb_frames, -1, -1, -1)
        
        warped = nn.functional.grid_sample(
            img_frames, 
            sel_inv_grid_frames, 
            mode=mode,
            padding_mode='zeros',
            align_corners=False
        )
        return warped
        
        
# =============================================================================
class AffineDeformationField(DeformationField):
    # =========================================================================
    r"""
    Warps a single image according to an inverse affine deformation field :math:`v`, i.e.
    the field that maps the pixels of the deformed image to the pixels of the
    original image. 
    
    It is constructed from a function of one parameter (time) that returns a
    tensor of shape :math:`(3,3)` representing a 2D affine homogeneous
    transformation matrix. The homogeneous transformation matrix corresponds to
    the inverse deformation field :math:`v`, i.e. the field that maps the pixels of the
    deformed image to the pixels of the original image. 
    
    To warp an image, the affine transformation matrix is evaluated at each
    time corresponding to the frames of the animation. The inverse deformation 
    field :math:`v` is then computed from the inverse of the affine
    transformation matrix, and the image is warped according to the deformation
    field :math:`v`.
    
    Contrary to :class:`DeformationField`, this class can warp images of
    variable sizes, as the inverse deformation field :math:`v` is computed from the
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
        :attr:`inverse_field_matrix` (torch.Tensor): function of one parameter
            (time) that returns a tensor of shape :math:`(3,3)` representing a
            2D affine homogeneous transformation matrix. That matrix is the
            inverse deformation field :math:`v`, i.e. the field that maps the
            pixels of the deformed image to the pixels of the original image.
        :attr:`align_corners` (bool, optional): Geometrically, we consider the
            pixels of the input as squares rather than points. If set to 
            :attr:`True`, the extrema (-1 and 1) are considered as referring to
            the center points of the input's corner pixels. If set to
            :attr:`False`, they are instead considered as referring to the
            corner points of the input's corner pixels, making the sampling
            more resolution agnostic. Default: :attr:`False`.
    
    Example 1: Progressive zooming in
        >>> def field(t):
        ...     return torch.tensor([[1-t/10, 0, 0], [0, 1-t/10, 0], [0, 0, 1]])
        >>> field = AffineDeformationField(field, align_corners=False)
        
    Example 2: Rotation of an image counter-clockwise, at a frequency of 1Hz
        >>> import numpy as np
        >>> def s(t):
        ...     return np.sin(2*np.pi*t)
        >>> def c(t):
        ...     return np.cos(2*np.pi*t)
        >>> def field(t):
        ...     return torch.tensor([[c(t), -s(t), 0], [s(t), c(t), 0], [0, 0, 1]])
        >>> field = AffineDeformationField(field, align_corners=False)
    """
    
    def __init__(
            self,
            inverse_field_matrix: torch.Tensor,
            align_corners=False):
        super().__init__(None, align_corners)
        self.inverse_field_matrix = inverse_field_matrix
        
    def get_inv_mat_frames(
            self, 
            t0: float, 
            t1: float=None, 
            n_frames: int=1
            ) -> torch.Tensor:
        r"""
        Returns a batch of affine transformation matrices corresponding to the
        inverse deformation field :math:`v`, evaluated at the times defined by the
        parameters :attr:`t0`, :attr:`t1` and :attr:`n_frames`.
        
        .. note::
            The time vector is created using the function `numpy.linspace` with
            the parameters :attr:`t0`, :attr:`t1` and :attr:`n_frames`. If 
            :math:`t0 > t1`, the time vector is created in reverse order, 
            giving a "backwards" animation.
        
        Args:
            :attr:`t0` (float): The first time at which to evaluate the function that
                gives the affine transformation matrix.
            :attr:`t1` (float, optional): The last time at which to evaluate the
                function. If :attr:`None`, the function is evaluated at the time
                :attr:`t0`. Defaults to :attr:`None`.
            :attr:`n_frames` (int, optional): The number of frames in the animation. 
                Defaults to 1.
        
        Returns:
            :attr:`inv_mat_frames` (torch.Tensor): A batch of affine 
                transformation matrices of shape :math:`(n_frames,3,3)`.
        
        Example 1: Evaluate the affine transformation matrix between t0=0 and
            t1 = 10, with 11 frames
            >>> def field(t): 
            ...     return torch.tensor([[1-t/10, 0, 0], [0, 1-t/10, 0], [0, 0, 1]])
            >>> field = AffineDeformationField(field, align_corners=False)
            >>> inv_mat_frames = field.get_inv_mat_frames(0, 10, 11)
        
        Example 2: Evaluate the affine transformation matrix at t0=4.5, with 1 frame
            >>> def field(t): 
            ...     return torch.tensor([[1-t/10, 0, 0], [0, 1-t/10, 0], [0, 0, 1]])
            >>> field = AffineDeformationField(field, align_corners=False)
            >>> inv_mat_frames = field.get_inv_mat_frames(4.5)
        """
        if t1 is None:
            return self.inverse_field_matrix(t0).unsqueeze(0)
        
        n_frames = operator.index(n_frames)
        time_vector = np.linspace(t0, t1, n_frames)
        inv_mat_frames = torch.zeros(n_frames, 3, 3)
        
        for i, t in enumerate(time_vector):
            inv_mat_frames[i] = self.inverse_field_matrix(t)
        return inv_mat_frames

    def format_params(
            self, 
            t0: float, 
            t1: float, 
            n_frames: int,
            fps: float
            ) -> tuple:
        r"""
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
            :math:`1/\text{{fps}}`. The number of frames is then computed as
            usual between :attr:`t0` and the truncated :attr:`t1`.
            
        Args:
            :attr:`t0` (float): Start time of the animation.
            :attr:`t1` (float): End time of the animation. If :attr:`None`,
                a single frame is warped.
            :attr:`n_frames` (int): Number of frames in the animation. If :attr:`None`,
                a single frame is warped.
            :attr:`fps` (float): Number of frames per second. If :attr:`None`, a single
                frame is warped.
        
        Returns:
            :attr:`(t0, t1, n_frames)` (tuple), where:
            :attr:`t0` (float): Start time of the animation.
            :attr:`t1` (float): End time of the animation.
            :attr:`n_frames` (int): Number of frames in the animation.
        
        The output follows this pattern:
        
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
                   │                 └─> :attr:`(t0, t0 + (n_frames-1) / fps, n_frames)`
                   └── No ──> :attr:`(t0, t1, n_frames)`
        """
        if t1 is None:
            if (fps is None) and (n_frames is None):   
                return (t0, t0, 1)    # no animation 
            else:
                raise ValueError(
                    "Unable to animate multiple frames: t1 was not given, "\
                    + f"but fps ({fps}) or n_frames ({n_frames}) was given."
                )
        else:
            # if fps and n_frames are given, use n_frames
            if n_frames is None:
                if fps is None:   
                    raise ValueError(
                        f"Unable to warp one image: t1 ({t1}) was given, "\
                        + "but fps and n_frames were not given."
                    )
                else:
                    # t1 is truncated to the closest lowest multiple of 1/fps
                    n_frames = int(np.floor(1 + (t1-t0) * fps))
                    return (t0, t0 + (n_frames-1) / fps, n_frames)
            else:
                return (t0, t1, n_frames)
    
    def save_inv_grid_frames(
            self, 
            inv_mat_frames: torch.Tensor, 
            size: torch.Size
            ) -> torch.Tensor:
        r"""
        Saves as a class attribute (:attr:`self.inverse_grid_frames`) the
        inverse deformation fields :math:`v` computed from the inverse of the affine
        transformation matrices evaluated at multiple times. 
        
        .. note::
            The name of the attribute (:attr:`inverse_grid_frames`) is the
            same as the attribute of the parent class (:class:`DeformationField`).
            This allows both classes to use the same method to warp images.

        Args:
            :attr:`inv_mat_frames` (torch.Tensor): Batch of inverse affine
                transformation matrices of shape :math:`(n_frames,3,3)`.
            :attr:`size` (torch.Size): Target output image size. It is a 4-tuple of
                integers :math:`(n_frames,c,Nx,Ny)`, where :math:`n_frames` is
                the number of frames in the animation, :math:`c` is the number
                of channels, and :math:`Nx` and :math:`Ny` are the number of
                pixels along the x-axis and y-axis respectively. The number of
                channels is usually 1 (grayscale) or 3 (color).
        
        Returns:
            :attr:`None`
        """
        # affine_grid needs the first two rows
        theta = inv_mat_frames[:, :2, :]
        inv_grid_frames = nn.functional.affine_grid(
            theta, size, align_corners=self.align_corners
        )
        self.inverse_grid_frames = inv_grid_frames
        return None
    
    def warp(
            self,
            img: torch.Tensor,      # single image (1|3, Nx, Ny)
            t0: float,
            t1: float=None,
            n_frames: int=None,
            fps: float=None,
            mode: str='bilinear'
            ) -> torch.Tensor:
        r"""
        Similarly to the method :meth:`DeformationField.warp` from the parent
        class, it warps a single image (:attr:`img`) according to the inverse 
        deformation field :math:`v` contained in the attribute
        :attr:`inverse_grid_frames`, between the times :attr:`t0` and
        :attr:`t1`. The number of frames in the animation is given by
        :attr:`n_frames` or :attr:`fps` if :attr:`n_frames` is :attr:`None`.

        Args:
            :attr:`img` (torch.Tensor): Image to deform of shape :math:`(c,Nx,Ny)`, where
                :math:`c` is the number of channels, and :math:`Nx` and 
                :math:`Ny` are the number of pixels along the x-axis and y-axis
                respectively. The number of channels is usually 1 (grayscale) or
                3 (color), if not a warning is raised. 
            :attr:`t0` (float): Start time of the animation.
            :attr:`t1` (float): End time of the animation. If :attr:`None`,
                a single frame is warped.
            :attr:`n_frames` (int): Number of frames in the animation. If :attr:`None`,
                :attr:`fps` is used to compute the number of frames.
            :attr:`fps` (float): Number of frames per second. If :attr:`None`, :attr:`n_frames`
                is used, or a single frame is warped.
            :attr:`mode` (str, optional): The interpolation mode to use. It is
                directly passed to the function `torch.nn.functional.grid_sample`.
                It must be one of the following: 'nearest', 'bilinear',
                'bicubic'. Defaults to 'bilinear'.

        Returns:
            :attr:`output` (torch.Tensor): The deformed batch of images of 
                shape :math:`(n_frames,c,Nx,Ny)`, where each image is 
                deformed according to the inverse deformation field :math:`v`
                contained in the attribute :attr:`inverse_grid_frames`.
        """
        t0, t1, n_frames = self.format_params(t0, t1, n_frames, fps)
        inv_mat_frames = self.get_inv_mat_frames(t0, t1, n_frames)
        self.save_inv_grid_frames(inv_mat_frames, [n_frames, *img.size()])
        return super().warp(img, 0, n_frames, mode=mode)
    
    def __repr__(self):
        s= f"AffineDeformationField({self.inverse_field_matrix.__name__=})" 
        return s
        
        