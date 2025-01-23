"""
Reconstruction networks.
"""

import warnings
from typing import Union, OrderedDict

import torch
import torch.nn as nn

import spyrit.core.meas as meas
import spyrit.core.inverse as inverse
import spyrit.core.prep as prep

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")


# =============================================================================
class FullNet(nn.Sequential):
    r"""Defines an arbitrary full (measurement + reconstruction) network.

    The forward pass of this network simulates measurements of a signal (or image)
    and reconstructs it from the measurements. To this end, it sequentially
    applies the measurement and reconstruction modules stored in the network
    under the keys `acqu_modules` and `recon_modules`, respectively.

    The modules contained within the measurement and reconstruction modules can
    be arbitrary.

    Args:
        acqu_modules (Union[OrderedDict, nn.Sequential]): Measurement modules.

        recon_modules (Union[OrderedDict, nn.Sequential]): Reconstruction modules.

    Raises:
        TypeError: If `acqu_modules` or `recon_modules` are not of type
        :class:`OrderedDict` or :class:`nn.Sequential`.

    Attributes:
        acqu_modules (nn.Sequential): Measurement modules.

        recon_modules (nn.Sequential): Reconstruction modules.

        Any (key, value) pair in `acqu_modules` and `recon_modules` is also stored
        as an attribute of the network. For example, if `acqu_modules` contains
        the key `acqu` for the acquisition operator, then it can be accessed as
        `self.acqu`. If two different modules in `acqu_modules` and `recon_modules`
        have the same key, the one in `acqu_modules` will be accessed first.
    """

    def __init__(
        self,
        acqu_modules: Union[OrderedDict, nn.Sequential],
        recon_modules: Union[OrderedDict, nn.Sequential],
    ):
        if isinstance(acqu_modules, OrderedDict):
            acqu_modules = nn.Sequential(acqu_modules)
        if not isinstance(acqu_modules, nn.Sequential):
            raise TypeError(
                "acqu_modules must be an OrderedDict or torch.nn.Sequential"
            )

        if isinstance(recon_modules, OrderedDict):
            recon_modules = nn.Sequential(recon_modules)
        if not isinstance(recon_modules, nn.Sequential):
            raise TypeError(
                "recon_modules must be an OrderedDict or torch.nn.Sequential"
            )

        all_modules = OrderedDict(
            {"acqu_modules": acqu_modules, "recon_modules": recon_modules}
        )
        super().__init__(all_modules)

    def forward(self, x):
        r"""Simulates measurements and reconstructs the signal.

        This is done by first simulating measurements of the input signal from
        the stored measurement modules `self.acqu_modules`. The measurements are
        then passed to the reconstruction modules `self.recon_modules` to
        reconstruct the signal.

        Args:
            x (torch.tensor): input tensor. For images, it is usually shaped
            `(b, c, h, w)` where `b` is the batch size, `c` is the number of
            channels, and `h` and `w` are the height and width of the images.

        Returns:
            torch.tensor: output tensor. Its shape depends on the output of the
            reconstruction modules.
        """
        x = self.acquire(x)  # use custom measurement operator
        x = self.reconstruct(x)  # use custom reconstruction operator
        return x

    def acquire(self, x):
        r"""Simulates measurements of the input signal.

        The measurements are simulated using the measurement modules stored in
        the network under the key `acqu_modules`.  They are all successively
        applied to the input tensor `x`.

        Args:
            x (torch.tensor): Input tensor. For images, it is usually shaped
            `(b, c, h, w)` where `b` is the batch size, `c` is the number of
            channels, and `h` and `w` are the height and width of the images.

        Returns:
            torch.tensor: Output tensor. Its shape depends on the output of the
            measurement modules.
        """
        return self.acqu_modules(x)

    def reconstruct(self, y):
        r"""Reconstructs the signal from measurements.

        The signal is reconstructed using the reconstruction modules stored in
        the network under the key `recon_modules`. They are all successively
        applied to the input tensor `y`.

        Args:
            y (torch.tensor): Input measurement tensor. It usually has measurements
            in the last dimension.

        Returns:
            torch.tensor: Output tensor. Its shape depends on the output of the
            reconstruction modules.
        """
        return self.recon_modules(y)


class _PrebuiltFullNet(FullNet):
    r"""Pre-built full (measurement + reconstruction) network. Designed so that
    other prebuilt networks inherit from this class. It adds the following
    attributes:
        - `acqu` (spyrit.core.meas): Acquisition operator.
        - `prep` (spyrit.core.prep): Preprocessing operator.
        - `denoi` (torch.nn.Module): Image denoising operator.

    The inverse operator is not added as an attribute because its name
    changes depending on the network. It is added as an attribute in the
    child classes.
    """

    def __init__(self, acqu_modules, recon_modules):
        super().__init__(acqu_modules, recon_modules)

    @property
    def acqu(self):
        return self.acqu_modules.acqu

    @acqu.setter
    def acqu(self, value):
        self.acqu_modules.acqu = value

    @acqu.deleter
    def acqu(self):
        del self.acqu_modules.acqu

    @property
    def prep(self):
        return self.recon_modules.prep

    @prep.setter
    def prep(self, value):
        self.recon_modules.prep = value

    @prep.deleter
    def prep(self):
        del self.recon_modules.prep

    @property
    def denoi(self):
        return self.recon_modules.denoi

    @denoi.setter
    def denoi(self, value):
        self.recon_modules.denoi = value

    @denoi.deleter
    def denoi(self):
        del self.recon_modules.denoi


# =============================================================================
class PositiveParameters(nn.Module):
    def __init__(self, params, requires_grad=True):
        super(PositiveParameters, self).__init__()
        self.params = torch.tensor(params, requires_grad=requires_grad)

    def forward(self):
        return torch.abs(self.params)


# =============================================================================
# class Denoise_layer(nn.Module):
#     r"""Defines a learnable Wiener filter that assumes additive white Gaussian noise.

#     The filter is pre-defined upon initialization with the standard deviation prior
#     (if known), or with an integer representing the size of the input vector.
#     In the second case, the standard deviation prior is initialized at random
#     from a uniform (0,2/size) distribution.

#     Using the foward method (the implicit call method), the filter is fully
#     defined:

#     .. math::
#         \sigma_\text{prior}^2/(\sigma^2_\text{prior} + \sigma^2_\text{meas})

#     where :math:`\sigma^2_\text{prior}` is the variance prior defined at
#     initialization and :math:`\sigma^2_\text{meas}` is the measurement variance
#     defined using the forward method. The value given by the equation above
#     can then be multiplied by the measurement vector to obtain the denoised
#     measurement vector.

#     .. note::
#         The weight (defined at initialization or accessible through the
#         attribute :attr:`weight`) should not be squared (as it is squared when
#         the forward method is called).

#     Args:
#         :attr:`std_dev_or_size` (torch.tensor or int): 1D tensor representing
#         the standard deviation prior or an integer defining the size of the
#         randomly-initialized standard deviation prior. If an array is passed
#         and it is not 1D, it is flattened. It is stored internally as a
#         :class:`nn.Parameter`, whose :attr:`data` attribute is accessed through
#         the :attr:`sigma` attribute, and whose :attr:`requires_grad` attribute
#         is accessed through the :attr:`requires_grad` attribute.

#     Shape for forward call:
#         - Input: :math:`(*, in\_features)` measurement variance.
#         - Output: :math:`(*, in\_features)` fully defined Wiener filter.

#     Attributes:
#         :attr:`weight`:
#         The learnable standard deviation prior :math:`\sigma_\text{prior}` of
#         shape :math:`(in\_features, 1)`. The values are initialized from
#         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k = 1/in\_features`.

#         :attr:`sigma`:
#         The learnable standard deviation prior :math:`\sigma_\text{prior}` of shape
#         :math:`(, in\_features)`. If the input is an integer, the standard deviation prior
#         is initialized at random from  :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`,
#         where :math:`k = 1/in\_features`.

#         :attr:`in_features`:
#         The number of input features.

#         :attr:`requires_grad`:
#         A boolean indicating whether the autograd should record operations on
#         the standard deviation tensor. Default is True.

#     Example:
#         >>> m = Denoise_layer(30)
#         >>> input = torch.randn(128, 30)
#         >>> output = m(input)
#         >>> print(output.size())
#         torch.Size([128, 30])
#     """

#     def __init__(
#         self, std_dev_prior_or_size: Union[torch.tensor, int], requires_grad=True
#     ):
#         super(Denoise_layer, self).__init__()

#         warnings.warn(
#             "This class is deprecated and will be removed in a future release. "
#             "Please use the `TikhonovMeasurementPriorDiag` class instead.",
#             DeprecationWarning,
#         )

#         if isinstance(std_dev_prior_or_size, int):
#             self.weight = nn.Parameter(
#                 torch.Tensor(std_dev_prior_or_size), requires_grad=requires_grad
#             )
#             self.reset_parameters()

#         else:
#             if not isinstance(std_dev_prior_or_size, torch.Tensor):
#                 raise TypeError(
#                     "std_dev_or_size should be an integer or a torch.Tensor"
#                 )
#             self.weight = nn.Parameter(
#                 std_dev_prior_or_size.reshape(-1), requires_grad=requires_grad
#             )

#     @property
#     def in_features(self):
#         return self.weight.data.numel()

#     def reset_parameters(self):
#         r"""
#         Resets the standard deviation prior :math:`\sigma_\text{prior}`.

#         The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`,
#         where :math:`k = 1/in\_features`. They are stored in the :attr:`weight`
#         attribute.
#         """
#         nn.init.uniform_(self.weight, 0, 2 / math.sqrt(self.in_features))

#     def forward(self, sigma_meas_squared: torch.tensor) -> torch.tensor:
#         r"""
#         Fully defines the Wiener filter with the measurement variance.

#         This outputs :math:`\sigma_\text{prior}^2/(\sigma_\text{prior}^2 + \sigma^2_\text{meas})`,
#         where :math:`\sigma^2_\text{meas}` is the measurement variance (see :attr:`sigma_meas_squared`) and
#         :math:`\sigma_\text{prior}` is the standard deviation prior defined
#         upon construction of the class (see :attr:`self.weight`).

#         .. note::
#             The measurement variance should be squared before being passed to
#             this method, unlike the standard deviation prior (defined at construction).

#         Args:
#             :attr:`sigma_meas_squared` (torch.tensor): input tensor :math:`\sigma^2_\text{meas}`
#             of shape :math:`(*, in\_features)`

#         Returns:
#             torch.tensor: The multiplicative filter of shape
#             :math:`(*, in\_features)`

#         Shape:
#             - Input: :math:`(*, in\_features)`
#             - Output: :math:`(*, in\_features)`
#         """
#         if sigma_meas_squared.shape[-1] != self.in_features:
#             raise ValueError(
#                 "The last dimension of the input tensor "
#                 + f"({sigma_meas_squared.shape[-1]})should be equal to the number of "
#                 + f"input features ({self.in_features})."
#             )
#         return self.tikho(sigma_meas_squared, self.weight)

#     def extra_repr(self):
#         return "in_features={}".format(self.in_features)

#     @staticmethod
#     def tikho(inputs: torch.tensor, weight: torch.tensor) -> torch.tensor:
#         # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
#         r"""
#         Applies a transformation to the incoming data: :math:`y = \sigma_\text{prior}^2/(\sigma_\text{prior}^2+x)`.

#         :math:`x` is the input tensor (see :attr:`inputs`) and :math:`\sigma_\text{prior}` is the
#         standard deviation prior (see :attr:`weight`).

#         Args:
#             :attr:`inputs` (torch.tensor): input tensor :math:`x` of shape
#             :math:`(N, *, in\_features)`

#             :attr:`weight` (torch.tensor): standard deviation prior :math:`\sigma_\text{prior}` of
#             shape :math:`(in\_features)`

#         Returns:
#             torch.tensor: The transformed data :math:`y` of shape
#             :math:`(N, in\_features)`

#         Shape:
#             - :attr:`inputs`: :math:`(N, *, in\_features)` where `*` means any number of
#               additional dimensions - Variance of measurements
#             - :attr:`weight`: :math:`(in\_features)` - corresponds to the standard deviation
#               of our prior.
#             - :attr:`output`: :math:`(N, in\_features)`
#         """
#         a = weight**2  # prefer to square it, because when learnt, it can go to the
#         # negative, which we do not want to happen.
#         # TO BE Potentially done : square inputs.
#         b = a + inputs
#         return a / b


# =============================================================================
class PinvNet(_PrebuiltFullNet):
    r"""Pre-built :class:`FullNet` that uses a pseudo inverse.

    As a :class:`FullNet`, this network has two modules: one for measurements
    and one for reconstruction.

    The measurement module only contains the acquisition operator. The
    reconstruction module contains a preprocessing operator, a pseudo inverse
    operator, and a denoising operator.

    The optional keyword arguments passed at initialization are fed in the
    pseudo inverse operator. This way, the regularization can be controlled
    directly from the :class:`PinvNet` constructor.

    .. important::
        If using a non-Identity denoiser, consider setting the optional keyword
        parameter `reshape_output` to `True` in the :class:`PinvNet` constructor.
        This will reshape the output of the pseudo inverse operator to match the
        acquisition operator input shape.

    Args:
        acqu (spyrit.core.meas): Acquisition operator (see :mod:`~spyrit.core.meas`)

        prep (spyrit.core.prep): Preprocessing operator (see :mod:`~spyrit.core.prep`)

        denoi (torch.nn.Module, optional): Image denoising operator. Default
        is :class:`~torch.nn.Identity`.

        **pinv_kwargs: Optional keyword arguments passed to the pseudo inverse
        operator (see :class:`~spyrit.core.inverse.PseudoInverse`).

    Attributes:
        :attr:`acqu_modules` (nn.Sequential): Measurement modules. Only contains
        the acquisition operator.

        :attr:`recon_modules` (nn.Sequential): Reconstruction modules. Contains
        the preprocessing operator, the pseudo inverse operator, and the denoising
        operator.

        :attr:`acqu` (spyrit.core.meas): Acquisition operator.

        :attr:`prep` (spyrit.core.prep): Preprocessing operator.

        :attr:`inv` (spyrit.core.inverse.PseudoInverse): Pseudo inverse operator.

        :attr:`denoi` (torch.nn.Module): Image denoising operator.

        :attr:`pinv_kwargs` (dict): Optional keyword arguments passed to the
        pseudo inverse operator.
    """

    def __init__(self, acqu, prep, denoi=nn.Identity(), **pinv_kwargs):

        pinv = inverse.PseudoInverse(acqu, **pinv_kwargs)
        acqu_modules = OrderedDict({"acqu": acqu})
        recon_modules = OrderedDict({"prep": prep, "pinv": pinv, "denoi": denoi})

        super().__init__(acqu_modules, recon_modules)
        self.pinv_kwargs = pinv_kwargs

    @property
    def pinv(self):
        return self.recon_modules.pinv

    @pinv.setter
    def pinv(self, value):
        self.recon_modules.pinv = value

    @pinv.deleter
    def pinv(self):
        del self.recon_modules.pinv

    # def meas2img(self, y):
    #     """Returns images from raw measurement vectors

    #     Args:
    #         :attr:`x`: raw measurement vectors

    #     Shape:
    #         :attr:`x`: :math:`(*,2M)`

    #         :attr:`output`: :math:`(*,H,W)`

    #     Example:
    #         >>> B, C, H, M = 10, 3, 64, 64**2
    #         >>> Ord = torch.ones(H,H)
    #         >>> meas = HadamSplit(M, H, Ord)
    #         >>> noise = NoNoise(meas)
    #         >>> prep = SplitPoisson(1.0, M, H**2)
    #         >>> recnet = PinvNet(noise, prep)
    #         >>> x = torch.rand((B,C,2*M), dtype=torch.float32)
    #         >>> z = recnet.reconstruct(x)
    #         >>> print(z.shape)
    #         torch.Size([10, 3, 64, 64])
    #     """
    #     m = self.prep(y)
    #     m = torch.nn.functional.pad(m, (0, self.acqu.meas_op.N - self.acqu.meas_op.M))

    #     # reindex the measurements
    #     z = self.acqu.meas_op.reindex(m, "cols", False)
    #     return z.reshape(*z.shape[:-1], self.acqu.meas_op.h, self.acqu.meas_op.w)

    def reconstruct_pinv(self, y):
        r"""Reconstructs measurement vectors without denoising.

        This method applies the :attr:`prep` and :attr:`pinv` modules of the
        reconstruction network to the input measurement vectors. It is
        somewhat equivalent to the :meth:`reconstruct` method, but without the
        denoising step (it is strictly equivalent if no additional reconstruction
        modules have been user-added to the network).

        .. note::
            This method may differ significantly from the :meth:`reconstruct`
            if more reconstruction modules have been user-added to the network.

        Args:
            y (torch.tensor): Input measurement tensor.

        Returns:
            torch.tensor: Output tensor. Its shape depends on the output of the
            reconstruction modules.
        """
        y = self.prep(y)
        y = self.pinv(y)
        return y

    def reconstruct_expe(self, y):
        r"""Reconstructs signal from experimental data.

        This is the same as :meth:`reconstruct` except that:

        1. Before the denoising step, the output is normalized to [0, 1] (i.e.
        it is divided by its maximum value).

        2. The output is de-normalized after the denoising step (i.e. it is
        multiplied its original maximum value found in 1.).

        Args:
            :attr:`y`: Raw measurement vectors.

        Returns:
            torch.tensor: Reconstructed experimental signal.
        """
        y = self.prep(y)
        y = self.pinv(y, **self.kwargs)
        max_val = y.max()

        y = y / max_val
        y = self.denoi(y)
        y = y * max_val

        return y, max_val


# =============================================================================
class DCNet(_PrebuiltFullNet):
    r"""Pre-built :class:`FullNet` that uses a :class:`TikhonovMeasurementPriorDiag` reconstruction operator.

    As a :class:`FullNet`, this network has two modules: one for measurements
    and one for reconstruction.

    The measurement module only contains the acquisition operator. The
    reconstruction module contains a preprocessing operator, a Tikhonov
    regularization :class:`TikhonovMeasurementPriorDiag` reconstruction
    operator, and a denoising operator.

    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`)

        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)

        :attr:`sigma`: Covariance prior (for details, see the
        :class:`~spyrit.core.recon.TikhonovMeasurementPriorDiag()` class)

        :attr:`denoi` (optional): Image denoising operator
        (see :class:`~spyrit.core.nnet`).
        Default :class:`~spyrit.core.nnet.Identity`

    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(b,c,h,w)`, with
        :math:`b` being the batch size, :math:`c` the number of channels, and
        :math:`h` and :math:`w` the height and width of the images.

        :attr:`output`: Reconstructed images with shape :math:`(b,c,h,w)`.

    Attributes:
        :attr:`acqu`: Acquisition operator initialized as :attr:`acqu`

        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`

        :attr:`tikho`: Data consistency layer initialized as :attr:`tikho`

        :attr:`denoi`: Image denoising operator initialized as :attr:`denoi`
    """

    def __init__(
        self,
        acqu: meas.Linear,
        prep,
        sigma: torch.tensor,
        denoi=nn.Identity(),
    ):
        sigma = acqu.reindex(sigma, "rows", False)
        sigma = acqu.reindex(sigma, "cols", True)
        tikho = inverse.TikhonovMeasurementPriorDiag(acqu, sigma)

        acqu_modules = OrderedDict({"acqu": acqu})
        recon_modules = OrderedDict({"prep": prep, "tikho": tikho, "denoi": denoi})
        super().__init__(acqu_modules, recon_modules)

    @property
    def tikho(self):
        return self.recon_modules.tikho

    @tikho.setter
    def tikho(self, value):
        self.recon_modules.tikho = value

    @tikho.deleter
    def tikho(self):
        del self.recon_modules.tikho

    def reconstruct(self, y):
        r"""Reconstruct an image from measurements.

        This method sucessively applies the preprocessing operator :attr:`prep`,
        the Tikhonov regularization operator :attr:`tikho`, and the denoising
        operator :attr:`denoi` to the input measurement vectors :attr:`x`.

        Args:
            :attr:`y`: raw measurement vectors. Has shape :math:`(b, c, m)`

        Shape:
            :attr:`y`: raw measurement vectors with shape :math:`(b, c, m)`

            :attr:`output`: reconstructed images with shape :math:`(b,c,h,w)`
        """
        # estimate the variance of the measurements
        var_noi = self.prep.sigma(y)
        y = self.prep(y)
        y = self.tikho.forward_no_prior(y, var_noi)
        y = self.denoi(y)
        return y

    def reconstruct_expe(self, x):
        r"""Reconstruction step of a reconstruction network

        Same as :meth:`reconstruct` reconstruct except that:

            1. The preprocessing step estimates the image intensity. The
            estimated intensity is used for both normalizing the raw
            data and computing the variance of the normalized data.

            2. The output images are "denormalized", i.e., have units of photon
            counts

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: :math:`(BC,2M)`

            :attr:`output`: :math:`(BC,1,H,W)`
        """
        *batches, n_measurements = x.shape

        # Preprocessing expe
        var_noi = self.prep.sigma_expe(x)
        x, N0_est = self.prep.forward_expe(x, self.acqu.meas_op)  # x <- x/N0_est
        x = x / self.prep.gain
        norm = self.prep.gain * N0_est

        # variance of preprocessed measurements
        var_noi = torch.div(
            var_noi, (norm.reshape(-1, 1).expand(*batches, self.acqu.M)) ** 2
        )

        # measurements to image domain processing
        x_0 = torch.zeros((*batches, *self.Acq.meas_op.img_shape), device=x.device)
        x = self.tikho(x, x_0, var_noi, self.Acq.meas_op)

        # Image domain denoising
        x = self.denoi(x)

        # Denormalization
        x = self.prep.denormalize_expe(
            x, norm, self.acqu.meas_op.h, self.acqu.meas_op.w
        )

        return x


# =============================================================================
class TikhoNet(_PrebuiltFullNet):
    r"""Pre-built :class:`FullNet` that uses a :class:`Tikhonov` reconstruction operator.

    As a :class:`FullNet`, this network has two modules: one for measurements
    and one for reconstruction.

    The measurement module only contains the acquisition operator. The
    reconstruction module contains a preprocessing operator, a Tikhonov inverse
    operator, and a denoising operator.


    Args:
        :attr:`noise` (spyrit.core.noise): Acquisition operator (see :mod:`~spyrit.core.meas`)

        :attr:`prep` (spyrit.core.prep): Preprocessing operator (see :mod:`~spyrit.core.prep`)

        :attr:`sigma` (torch.tensor): Image-domain covariance prior (for details, see the
        :class:`~spyrit.core.recon.Tikhonov()` class)

        :attr:`denoi` (torch.nn.Module, optional): Image denoising operator
        (see :class:`~spyrit.core.nnet`). Default :class:`~spyrit.core.nnet.Identity`

        :attr:`kwargs` (dict): Optional keyword arguments passed to the
        :class:`~spyrit.core.recon.Tikhonov()` constructor. May contain
        the following keys:
            - `approx` (bool): If True, the Tikhonov inversion step is approximated
            using a diagonal matrix. Default is False.
            - `reshape_output` (bool): If True, the output of the Tikhonov
            inversion step is reshaped to match the acquisition operator input
            shape. Default is False.

    .. important::
        If using a non-Identity denoiser, consider setting the optional keyword
        parameter `reshape_output` to `True` in the :class:`TikhoNet` constructor.

    Input / Output:
        :attr:`input` (torch.tensor): Ground-truth images with shape :math:`(b,c,h,w)`.

        :attr:`output` (torch.tensor): Reconstructed images with shape :math:`(b,c,h,w)`.

    Attributes:
        :attr:`acqu`: Acquisition operator initialized as :attr:`noise`

        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`

        :attr:`tikho`: Data consistency layer initialized as :attr:`Tikhonov(noise.meas_op, sigma)`

        :attr:`denoi`: Image denoising operator initialized as :attr:`denoi`
    """

    def __init__(self, acqu, prep, sigma: torch.tensor, denoi=nn.Identity(), **kwargs):

        tikho = inverse.Tikhonov(acqu, sigma, **kwargs)
        acqu_modules = OrderedDict({"acqu": acqu})
        recon_modules = OrderedDict({"prep": prep, "tikho": tikho, "denoi": denoi})
        super().__init__(acqu_modules, recon_modules)

    @property
    def tikho(self):
        return self.recon_modules.tikho

    @tikho.setter
    def tikho(self, value):
        self.recon_modules.tikho = value

    @tikho.deleter
    def tikho(self):
        del self.recon_modules.tikho

    def reconstruct(self, x):
        r"""Reconstruction (measurement-to-image mapping)

        Args:
            :attr:`x` (torch.tensor): Raw measurement vectors with shape :math:`(B,C,M)`.

        Output:
            torch.tensor: Reconstructed images with shape :math:`(B,C,H,W)`
        """
        # covariance of measurements
        cov_meas = self.prep.sigma(x)
        cov_meas = torch.diag_embed(cov_meas)  #
        # Preprocessing
        x = self.prep(x)
        # measurements to image domain processing
        x = self.tikho(x, cov_meas)
        # x = x.reshape(*x.shape[:-1], self.acqu.meas_op.h, self.acqu.meas_op.w)
        # Image domain denoising
        x = self.denoi(x)
        return x

    def reconstruct_expe(self, x):
        r"""Reconstruction (measurement-to-image mapping) for experimental data.

        Args:
            :attr:`x` (torch.tensor): Raw measurement vectors with shape :math:`(B,C,M)`.

        Output:
            torch.tensor: Reconstructed images with shape :math:`(B,C,H,W)`
        """
        # Preprocessing
        cov_meas = self.prep.sigma_expe(x)
        # print(cov_meas)
        # print(self.prep.nbin, self.prep.mudark)
        # x, norm = self.prep.forward_expe(x, self.acqu.meas_op, (-2,-1)) # shape: [*, M]

        # Alternative where the mean is computed on each row
        x, norm = self.prep.forward_expe(x, self.acqu.meas_op)  # shape: [*, M]

        # covariance of measurements
        cov_meas = cov_meas / norm**2
        cov_meas = torch.diag_embed(cov_meas)

        # measurements to image domain processing
        x = self.tikho(x, cov_meas)
        # x = x.reshape(*x.shape[:-1], self.acqu.meas_op.h, self.acqu.meas_op.w)

        # Image domain denoising
        x = self.denoi(x)

        # Denormalization
        x = self.prep.denormalize_expe(x, norm, x.shape[-2], x.shape[-1])

        return x, norm


# =============================================================================
class LearnedPGD(nn.Module):
    r"""Learned Proximal Gradient Descent reconstruction network.
    Iterative algorithm that alternates between a gradient step and a proximal step,
    where the proximal operator is learned denoiser. The update rule is given by:

    :math:`x_{k+1} = prox(\hat{x_k} - step * H^T (Hx_k - y))=
    denoi(\hat{x_k} - step * H^T (Hx_k - y))`

    Args:
        :attr:`acqu`: Acquisition operator (see :class:`~spyrit.core.meas`)

        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)

        :attr:`denoi` (optional): Image denoising operator
        (see :class:`~spyrit.core.nnet`).
        Default :class:`~spyrit.core.nnet.Identity`

        :attr:`iter_stop` (int): Number of iterations of the LPGD algorithm
        (commonly 3 to 10, trade-off between accuracy and speed).
        Default 3 (for speed and with higher accuracy than post-processing denoising)

        :attr:`step` (float): Step size of the LPGD algorithm. Default is None,
        and it is estimated as the inverse of the Lipschitz constant of the gradient of the
        data fidelity term.
            - If :math:`meas_op.N` is available, the step size is estimated as
            :math:`step=1/L=1/\text{meas_op.N}`, true for Hadamard operators.
            - If not, the step size is estimated from by computing
            the Lipschitz constant as the largest singular value of the
            Hessians, :math:`L=\lambda_{\max}(H^TH)`. If this fails,
            the step size is set to 1e-4.

        :attr:`step_estimation` (bool): Default False. See :attr:`step` for details.

        :attr:`step_grad` (bool): Default False. If True, the step size is learned
        as a parameter of the network. Not tested yet.

        :attr:`wls` (bool): Default False. If True, the data fidelity term is
        modified to be the weighted least squares (WLS) term, which approximates
        the Poisson likelihood. In this case, the data fidelity term is
        :math:`\|Hx-y\|^2_{C^{-1}}`, where :math:`C` is the covariance matrix.
        We assume that :math:`C` is diagonal, and the diagonal elements are
        the measurement noise variances, estimated from :class:`~spyrit.core.prep.sigma`.

        :attr:`gt` (torch.tensor): Ground-truth images. If available, the mean
        squared error (MSE) is computed and logged. Default None.

        :attr:`log_fidelity` (bool): Default False. If True, the data fidelity term
        is logged for each iteration of the LPGD algorithm.

    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(B,C,H,W)`

        :attr:`output`: Reconstructed images with shape :math:`(B,C,H,W)`

    Attributes:
        :attr:`acqu`: Acquisition operator initialized as :attr:`acqu`

        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`

        :attr:`pinv`: Analytical reconstruction operator initialized as
        :class:`~spyrit.core.recon.PseudoInverse()`

        :attr:`denoi`: Image denoising operator initialized as :attr:`denoi`

    Example:
        >>> B, C, H, M = 10, 1, 64, 64**2
        >>> Ord = torch.ones((H,H))
        >>> meas = HadamSplit(M, H, Ord)
        >>> noise = NoNoise(meas)
        >>> prep = SplitPoisson(1.0, M, H*H)
        >>> recnet = LearnedPGD(noise, prep)
        >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
        >>> z = recnet(x)
        >>> print(z.shape)
        torch.Size([10, 1, 64, 64])
        >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
        tensor(5.8912e-06)
    """

    def __init__(
        self,
        acqu: meas.LinearSplit,
        prep: prep.UnsplitRescale,
        denoi=nn.Identity(),
        *,
        iter_stop=3,
        x0=0,
        step=None,
        step_estimation=False,
        step_grad=False,
        step_decay=1,
        wls=False,
        gt=None,
        log_fidelity=False,
        res_learn=False,
        **pinv_kwargs,
    ):
        super().__init__()
        # nn.module
        self.acqu = acqu
        self.prep = prep
        self.denoi = denoi
        self.pinv_kwargs = pinv_kwargs

        self.pinv = inverse.PseudoInverse(self.acqu, **pinv_kwargs)

        # LPGD algo
        self.x0 = x0
        self.iter_stop = iter_stop
        self.step = step
        self.step_estimation = step_estimation
        self.step_grad = step_grad
        self.step_decay = step_decay
        self.res_learn = res_learn

        # Init step size (estimate)
        self.set_stepsize(step)

        # WLS
        self.wls = wls

        # Log fidelity
        self.log_fidelity = log_fidelity

        # Log MSE (Ground truth available)
        if gt is not None:
            self.x_gt = nn.Parameter(
                torch.tensor(gt.reshape(gt.shape[0], -1)), requires_grad=False
            )
        else:
            self.x_gt = None

    def step_schedule(self, step):
        if self.step_decay != 1:
            step = [step * self.step_decay**i for i in range(self.iter_stop)]
        elif self.iter_stop > 1:
            step = [step for i in range(self.iter_stop)]
        else:
            step = [step]
        return step

    def set_stepsize(self, step):
        if step is None:
            # Stimate stepsize from Lipschitz constant
            if hasattr(self.acqu, "N"):
                step = 1 / self.acqu.N
            else:
                # Estimate step size as 1/sv_max(H^TH); if failed, set to 1e-4
                self.step_estimation = True
                step = 1e-4

        step = self.step_schedule(step)
        # step = nn.Parameter(torch.tensor(step), requires_grad=self.step_grad)
        step = PositiveParameters(step, requires_grad=self.step_grad)
        self.step = step

    def forward(self, x):
        r"""Full pipeline of reconstruction network

        Args:
            :attr:`x`: ground-truth images

        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`

            :attr:`output`: reconstructed images with shape :math:`(B,C,H,W)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> recnet = LearnedPGD(noise, prep)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
            >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
            tensor(5.8912e-06)
        """
        x = self.acquire(x)
        x = self.reconstruct(x)
        return x

    def acquire(self, x):
        r"""Simulate data acquisition

        Args:
            :attr:`x`: ground-truth images

        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`

            :attr:`output`: measurement vectors with shape :math:`(BC,2M)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet.acquire(x)
            >>> print(z.shape)
            torch.Size([10, 8192])
        """
        return self.acqu(x)

    def hessian_sv(self):
        H = self.acqu.H
        if self.wls:
            std_mat = 1 / torch.sqrt(self.meas_variance)
            std_mat = torch.diag(std_mat.reshape(-1))
            H = torch.matmul(std_mat, H)
        try:
            s = torch.linalg.svdvals(torch.matmul(H.t(), H))
        except:
            print("svdvals(H^T*H) failed, trying svdvals(H) instead")
            s = torch.linalg.svdvals(H) ** 2
        return s

    def stepsize_gd(self):
        s = self.hessian_sv()
        self.step = 2 / (s.min() + s.max())  # Kressner, EPFL, GD #1/(2*s.max()**2)

    def cost_fun(self, x, y):
        proj = self.acqu.measure_H(x)
        res = proj - y
        if self.wls:
            res = res / torch.sqrt(self.meas_variance)
        return torch.linalg.norm(res) ** 2

    def mse_fun(self, x, x_gt):
        return torch.linalg.norm(x - x_gt)

    def reconstruct(self, x):
        r"""Reconstruction step of a reconstruction network

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: :math:`(BC,2M)`

            :attr:`output`: :math:`(BC,1,H,W)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H**2)
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """

        # Compute the stepsize from the Lipschitz constant
        if self.step_estimation:
            self.stepsize_gd()

        step = self.step
        if not isinstance(step, torch.Tensor):
            step = step.params

        # Preprocessing in the measurement domain
        m = self.prep(x)

        if self.wls:
            # Get variance of the measurements
            if hasattr(self.prep, "sigma"):
                self.meas_variance = self.prep.sigma(x)
            else:
                print(
                    "WLS requires the variance of the measurements to be known!. Estimating var==m"
                )
                meas_variance = m

            # Normalize the stepsize to account for the variance
            meas_variance_img_min, _ = torch.min(meas_variance, 1)  # 128
            step = step.reshape(self.iter_stop, 1).to(x.device)
            # Multiply meas_variance_img_min and step
            step = meas_variance_img_min * step

        # If pinv method is defined
        if self.x0 != 0:
            x = self.pinv(m)
            # if hasattr(self.acqu, "pinv"):
            #     x = self.acqu.pinv(m)

            # proximal step (prior)
            if isinstance(self.denoi, nn.ModuleList):
                x = self.denoi[0](x)
            else:
                x = self.denoi(x)

            if self.res_learn:
                z0 = x.detach().clone()
        else:
            # zero init
            x = torch.zeros((*x.shape[:-1], *self.acqu.meas_shape), device=x.device)

        if self.log_fidelity:
            self.cost = []
            with torch.no_grad():
                # data_fidelity.append(self.data_fidelity(torch.zeros_like(x), m).cpu().numpy().tolist())
                self.cost.append(self.cost_fun(x, m).cpu().numpy().tolist())
        if self.x_gt is not None:
            self.mse = []
            with torch.no_grad():
                self.mse.append(self.mse_fun(x, self.x_gt).cpu().numpy().tolist())

        # u = None  # is this line useless ??

        for i in range(self.iter_stop):
            # gradient step (data fidelity)
            res = self.acqu.measure_H(x) - m
            if self.wls:
                res = res / meas_variance
                upd = step[i].reshape(-1, 1) * self.acqu.adjoint_H(res)
            else:
                upd = step[i] * self.acqu.adjoint_H(res)
            x = x - self.acqu.unvectorize(upd)

            if i == 0 and self.res_learn and self.x0 == 0:
                # if x0 does not exist
                z0 = x.detach().clone()

            # proximal step (prior)
            if isinstance(self.denoi, nn.ModuleList):
                x = self.denoi[i](x)
            else:
                x = self.denoi(x)

            if self.log_fidelity:
                with torch.no_grad():
                    self.cost.append(self.cost_fun(x, m).cpu().numpy().tolist())
            # Compute mse if ground truth is field
            if self.x_gt is not None:
                with torch.no_grad():
                    self.mse.append(self.mse_fun(x, self.x_gt).cpu().numpy().tolist())

        if self.log_fidelity:
            print(f"Data fidelity: {(self.cost)}. Stepsize: {self.step}")
        if self.x_gt is not None:
            print(f"|x - x_gt| = {self.mse}")

        if self.res_learn:
            # z=x-step*grad(L), x = P(z), x_end = z0 + P(z)
            x = x + z0
        return x

    def reconstruct_expe(self, x):
        r"""Reconstruction step of a reconstruction network

        .. warning ::
            !! This method hasn't been updated to the incoming v3 !!

        Same as :meth:`reconstruct` reconstruct except that:

        1. The preprocessing step estimates the image intensity for normalization

        2. The output images are "denormalized", i.e., have units of photon counts

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: :math:`(BC,2M)`

            :attr:`output`: :math:`(BC,1,H,W)`
        """
        # Preprocessing
        x, N0_est = self.prep.forward_expe(x, self.acqu.meas_op)  # shape x = [b*c, M]
        # print(N0_est)

        # measurements to image domain processing
        x = self.pinv(x, self.acqu.meas_op)

        # Denoising
        x = self.denoi(x)
        # print(x.max())

        # Denormalization
        x = self.prep.denormalize_expe(
            x, N0_est, self.acqu.meas_op.h, self.acqu.meas_op.w
        )
        return x
