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

    Example:
        # >>> import torch.nn as nn
        # >>> acqu1 = lambda x: x*2
        # >>> acqu2 = lambda x: x - 10
        # >>> acqu = nn.Sequential(acqu1, acqu2)
        # >>> recon1 = lambda x: (x + 10) / 2
        # >>> recon = nn.Sequential(recon1)
        # >>> net = FullNet(acqu, recon)
    """

    def __init__(
        self,
        acqu_modules: Union[OrderedDict, nn.Sequential],
        recon_modules: Union[OrderedDict, nn.Sequential],
        *,
        device: torch.device = torch.device("cpu"),
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
        self.to(device)

    def forward(self, x):
        r"""Apply the full network to the input signal.

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

        Example:
            # >>> acqu1 = lambda x: x*2
            # >>> acqu2 = lambda x: x - 10
            # >>> acqu = nn.Sequential(acqu1, acqu2)
            # >>> recon1 = lambda x: (x + 10) / 2
            # >>> recon = nn.Sequential(recon1)
            # >>> net = FullNet(acqu, recon)
            # >>> x = torch.tensor(5.0)
            # >>> y = net(x)
            # >>> print(y)
            tensor(5.0000)
        """
        x = self.acquire(x)  # use custom measurement operator
        x = self.reconstruct(x)  # use custom reconstruction operator
        return x

    def acquire(self, x):
        r"""Apply the measurement modules to the input signal.

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

        Example:
            # >>> acqu1 = lambda x: x*2
            # >>> acqu2 = lambda x: x - 10
            # >>> acqu = nn.Sequential(acqu1, acqu2)
            # >>> recon1 = lambda x: (x + 10) / 2
            # >>> recon = nn.Sequential(recon1)
            # >>> net = FullNet(acqu, recon)
            # >>> x = torch.tensor(5.0)
            # >>> z = net.acquire(x)
            # >>> print(z)
            tensor(0.0000)
        """
        return self.acqu_modules(x)

    def reconstruct(self, y):
        r"""Apply the reconstruction modules to the input measurements.

        The signal is reconstructed using the reconstruction modules stored in
        the network under the key `recon_modules`. They are all successively
        applied to the input tensor `y`.

        Args:
            y (torch.tensor): Input measurement tensor. It usually has measurements
            in the last dimension.

        Returns:
            torch.tensor: Output tensor. Its shape depends on the output of the
            reconstruction modules.

        Example:
            # >>> acqu1 = lambda x: x*2
            # >>> acqu2 = lambda x: x - 10
            # >>> acqu = nn.Sequential(acqu1, acqu2)
            # >>> recon1 = lambda x: (x + 10) / 2
            # >>> recon = nn.Sequential(recon1)
            # >>> net = FullNet(acqu, recon)
            # >>> y = torch.tensor(0.0)
            # >>> z = net.reconstruct(y)
            # >>> print(z)
            tensor(5.0000)
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

    .. note::
        For more details, see the :class:`FullNet` class.
    """

    def __init__(
        self,
        acqu_modules,
        recon_modules,
        *,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(acqu_modules, recon_modules, device=device)

    @property
    def acqu(self) -> meas.Linear:
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
    r"""Module that stores a signed tensor and returns its absolute value.

    This module is used to store the step size of the LearnedPGD network. The
    step size must be positive, so it is stored as a signed tensor and its
    absolute value is returned when the module is called.

    Args:
        params (array_like): Signed array-like object. It is used to construct
        a new tensor.

        requires_grad (bool): If True, the tensor requires gradient. Default is True.

    Attributes:
        :attr:`params` (torch.tensor): Signed tensor.

    Methods:
        :meth:`forward`: Returns the absolute value of the signed tensor.

    Example:
        >>> values = [-1., 2., -3., 4.]
        >>> pos_params = PositiveParameters(values)
        >>> print(pos_params.params)
        tensor([-1.,  2., -3.,  4.], requires_grad=True)
        >>> print(pos_params())
        tensor([1., 2., 3., 4.], grad_fn=<AbsBackward0>)
    """

    def __init__(self, params, requires_grad=True):
        super(PositiveParameters, self).__init__()
        self.params = torch.tensor(params, requires_grad=requires_grad)

    def forward(self):
        r"""Returns the absolute value of the stored signed tensor.

        Example:
            >>> values = [-1., 2., -3., 4.]
            >>> pos_params = PositiveParameters(values)
            >>> print(pos_params())
            tensor([1., 2., 3., 4.], grad_fn=<AbsBackward0>)
        """
        return torch.abs(self.params)


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

    Args:
        acqu (spyrit.core.meas): Acquisition operator (see :mod:`~spyrit.core.meas`)

        prep (spyrit.core.prep): Preprocessing operator (see :mod:`~spyrit.core.prep`)

        denoi (torch.nn.Module, optional): Image denoising operator. Default
        is :class:`~torch.nn.Identity`.

        **pinv_kwargs: Optional keyword arguments passed to the pseudo inverse
        operator (see :class:`~spyrit.core.inverse.PseudoInverse`).

    Attributes:
        :attr:`acqu` (spyrit.core.meas): Acquisition operator.

        :attr:`acqu_modules` (nn.Sequential): Measurement modules. Only contains
        the acquisition operator.

        :attr:`prep` (spyrit.core.prep): Preprocessing operator.

        :attr:`inv` (spyrit.core.inverse.PseudoInverse): Pseudo inverse operator.

        :attr:`denoi` (torch.nn.Module): Image denoising operator.

        :attr:`recon_modules` (nn.Sequential): Reconstruction modules. Contains
        the preprocessing operator, the pseudo inverse operator, and the denoising
        operator.

        :attr:`pinv_kwargs` (dict): Optional keyword arguments passed to the
        pseudo inverse operator.

    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(b,c,h,w)`, with
        :math:`b` being the batch size, :math:`c` the number of channels, and
        :math:`h` and :math:`w` the height and width of the images.

        :attr:`output`: Reconstructed images with shape :math:`(b,c,h,w)`.

    Example:
        >>> import spyrit
        >>> acqu = spyrit.core.meas.HadamSplit2d(32)
        >>> prep = spyrit.core.prep.Rescale(1.0)
        >>> pinv = PinvNet(acqu, prep, device=torch.device("cpu"))

    Example with a regularized pseudo inverse:
        >>> import spyrit
        >>> noise_model = spyrit.core.noise.Poisson(100)
        >>> acqu = spyrit.core.meas.HadamSplit2d(32, noise_model=noise_model)
        >>> prep = spyrit.core.prep.Rescale(100)
        >>> pinv = PinvNet(acqu, prep, use_fast_pinv=False, store_H_pinv=True, regularization='H1', eta=1e-6, img_shape=(32, 32))
    """

    def __init__(
        self,
        acqu: meas.Linear,
        prep,
        denoi=nn.Identity(),
        *,
        device: torch.device = torch.device("cpu"),
        **pinv_kwargs,
    ):

        pinv = inverse.PseudoInverse(acqu, **pinv_kwargs)
        acqu_modules = OrderedDict({"acqu": acqu})
        recon_modules = OrderedDict({"prep": prep, "pinv": pinv, "denoi": denoi})

        super().__init__(acqu_modules, recon_modules, device=device)
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
            y (torch.tensor): Input measurement tensor. Its shape depends on the
            preprocessing operator input shape.

        Returns:
            torch.tensor: Output tensor. Its shape depends on the output of the
            reconstruction modules.

        Example:
            >>> import spyrit
            >>> acqu = spyrit.core.meas.HadamSplit2d(32)
            >>> prep = spyrit.core.prep.Rescale(1.0)
            >>> pinv = PinvNet(acqu, prep, device=torch.device("cpu"))
            >>> x = torch.rand(10, 1, 32, 32)
            >>> y = pinv.acquire(x)
            >>> z = pinv.reconstruct_pinv(y)
            >>> print(z.shape)
            torch.Size([10, 1, 32, 32])
        """
        y = self.prep(y)
        y = self.pinv(y)
        return y


# =============================================================================
class DCNet(_PrebuiltFullNet):
    r"""Pre-built :class:`FullNet` that uses a :class:`~spyrit.core.inverse.TikhonovMeasurementPriorDiag` reconstruction operator.

    As a :class:`FullNet`, this network has two modules: one for the measurements
    and one for the reconstruction.

    The measurement module only contains the acquisition operator. The acquisition
    operator must be a :class:`spyrit.core.meas.HadamSplit2d` operator.
    The reconstruction module contains a preprocessing operator, a Tikhonov
    regularization :class:`spyrit.core.inverse.TikhonovMeasurementPriorDiag` reconstruction
    operator, and a denoising operator.

    Args:
        :attr:`acqu`: Acquisition operator (see :class:`~spyrit.core.meas.HadamSplit2d`)

        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)

        :attr:`sigma`: Measurement covariance prior (for details, see the
        :class:`~spyrit.core.recon.TikhonovMeasurementPriorDiag()` class)

        :attr:`denoi` (optional): Image denoising operator
        (see :class:`~spyrit.core.nnet`). Default :class:`~spyrit.core.nnet.Identity`

    Attributes:
        :attr:`acqu`: Acquisition operator initialized as :attr:`acqu`

        :attr:`acqu_modules` (nn.Sequential): Measurement modules. Only contains
        the acquisition operator.

        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`

        :attr:`tikho`: Tikhonv regularization operator initialized as a
        :class:`~spyrit.core.recon.TikhonovMeasurementPriorDiag` operator.

        :attr:`denoi`: Image denoising operator initialized as :attr:`denoi`

        :attr:`recon_modules` (nn.Sequential): Reconstruction modules. Contains
        the preprocessing operator, the Tikhonov regularizaiton operator, and
        the denoising operator.

    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(b,c,h,w)`, with
        :math:`b` being the batch size, :math:`c` the number of channels, and
        :math:`h` and :math:`w` the height and width of the images.

        :attr:`output`: Reconstructed images with shape :math:`(b,c,h,w)`.

    Example:
        >>> from spyrit.core.meas import HadamSplit2d
        >>> from spyrit.core.prep import UnsplitRescale
        >>> from spyrit.core.recon import DCNet
        >>> import torch
        >>> acqu = HadamSplit2d(32)
        >>> prep = UnsplitRescale()
        >>> sigma = torch.eye(32*32,32*32)
        >>> dcnet = DCNet(acqu, prep, sigma)
        >>> y = torch.randn(10, 1, 2048)
        >>> z = dcnet.reconstruct(y)
        >>> print(z.shape)
        torch.Size([10, 1, 32, 32])
    """

    def __init__(
        self,
        acqu: meas.HadamSplit2d,
        prep: Union[prep.Rescale, prep.RescaleEstim],
        sigma: torch.tensor,
        denoi=nn.Identity(),
        *,
        device: torch.device = torch.device("cpu"),
    ):
        sigma = acqu.reindex(sigma, "rows", False)
        sigma = acqu.reindex(sigma, "cols", True)
        tikho = inverse.TikhonovMeasurementPriorDiag(acqu, sigma)

        acqu_modules = OrderedDict({"acqu": acqu})
        recon_modules = OrderedDict({"prep": prep, "tikho": tikho, "denoi": denoi})
        super().__init__(acqu_modules, recon_modules, device=device)

    @property
    def tikho(self) -> inverse.TikhonovMeasurementPriorDiag:
        return self.recon_modules.tikho

    @tikho.setter
    def tikho(self, value):
        self.recon_modules.tikho = value

    @tikho.deleter
    def tikho(self):
        del self.recon_modules.tikho

    def reconstruct(self, y: torch.tensor) -> torch.tensor:
        r"""Reconstruct an image from measurements.

        This method sucessively applies the preprocessing operator :attr:`prep`,
        the Tikhonov regularization operator :attr:`tikho`, and the denoising
        operator :attr:`denoi` to the input measurement vectors :attr:`x`.

        Args:
            :attr:`y`: raw measurement vectors with shape :math:`(b, c, M)`

        Returns:
            torch.tensor: Reconstructed images with shape :math:`(b,c,h,w)`

        Example:
            >>> from spyrit.core.meas import HadamSplit2d
            >>> from spyrit.core.prep import UnsplitRescale
            >>> from spyrit.core.recon import DCNet
            >>> import torch
            >>> acqu = HadamSplit2d(32)
            >>> prep = UnsplitRescale()
            >>> sigma = torch.eye(32*32,32*32)
            >>> dcnet = DCNet(acqu, prep, sigma)
            >>> y = torch.randn(10, 1, 2048)
            >>> z = dcnet.reconstruct(y)
            >>> print(z.shape)
            torch.Size([10, 1, 32, 32])
        """
        y = self.reconstruct_pinv(y)
        y = self.denoi(y)
        return y

    def reconstruct_pinv(self, y: torch.tensor) -> torch.tensor:
        r"""Reconstruct an image from measurements without denoising.

        This method sucessively applies the preprocessing operator :attr:`prep`
        and the Tikhonov regularization operator :attr:`tikho` to the input
        measurement vectors :attr:`x`.

        Args:
            :attr:`y`: raw measurement vectors. Have shape :math:`(b, c, m)`

        Returns:
            torch.tensor: Reconstructed images. Have shape :math:`(b,c,h,w)`

        Example:
            >>> from spyrit.core.meas import HadamSplit2d
            >>> from spyrit.core.prep import UnsplitRescale
            >>> from spyrit.core.recon import DCNet
            >>> import torch
            >>> acqu = HadamSplit2d(32)
            >>> prep = UnsplitRescale()
            >>> sigma = torch.eye(32*32,32*32)
            >>> dcnet = DCNet(acqu, prep, sigma)
            >>> y = torch.randn(10, 1, 2048)
            >>> z = dcnet.reconstruct_pinv(y)
            >>> print(z.shape)
            torch.Size([10, 1, 32, 32])
        """
        # estimate the variance of the measurements
        var_noi = self.prep.sigma(y)
        y = self.prep(y)
        y = self.tikho.forward_no_prior(y, var_noi)
        return y


# =============================================================================
class TikhoNet(_PrebuiltFullNet):
    r"""Pre-built :class:`FullNet` that uses a :class:`Tikhonov` reconstruction operator.

    As a :class:`FullNet`, this network has two modules: one for measurements
    and one for reconstruction.

    The measurement module only contains the acquisition operator. The
    reconstruction module contains a preprocessing operator, a Tikhonov inverse
    operator, and a denoising operator.

    The optional keyword arguments passed at initialization are fed in the
    :class:`Tikhonov` operator. This way, the regularization can be controlled
    directly from the :class:`TikhoNet` constructor.

    Args:
        :attr:`acqu` (spyrit.core.meas): Acquisition operator (see :mod:`~spyrit.core.meas`)

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
            shape. Default is True.

    Attributes:
        :attr:`acqu`: Acquisition operator initialized as :attr:`acqu`.

        :attr:`acqu_modules` (nn.Sequential): Measurement modules. Only contains
        the acquisition operator.

        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`.

        :attr:`tikho`: Data consistency layer initialized as :attr:`Tikhonov(noise.meas_op, sigma)`.

        :attr:`denoi`: Image denoising operator initialized as :attr:`denoi`.

        :attr:`recon_modules` (nn.Sequential): Reconstruction modules. Contains
        the preprocessing operator, the pseudo inverse operator, and the denoising
        operator.

    Input / Output:
        :attr:`input` (torch.tensor): Ground-truth images with shape :math:`(b,c,h,w)`.

        :attr:`output` (torch.tensor): Reconstructed images with shape :math:`(b,c,h,w)`.

    Example 1:
        # >>> noise = spyrit.core.noise.Poisson(100)
        # >>> acqu = spyrit.core.meas.HadamSplit2d(64, noise_model=noise)
        # >>> prep = spyrit.core.prep.Rescale(100)
        # >>> sigma = torch.ones(64, 64)
        # >>> tikho = TikhoNet(acqu, prep, sigma, device=torch.device("cuda"))
        # >>> x = torch.rand(10, 1, 64, 64)
        # >>> z = tikho(x)
        # >>> print(z.shape)
        # torch.Size([10, 1, 64, 64])

    Example 2:
        # >>> noise = spyrit.core.noise.Gaussian(1.0)
        # >>> acqu = spyrit.core.meas.HadamSplit2d(64, noise_model=noise)
        # >>> prep = spyrit.core.prep.Rescale(1.0)
        # >>> sigma = torch.ones(64, 64)
        # >>> tikho = TikhoNet(acqu, prep, sigma, approx=True, reshape_output=False)
        # >>> x = torch.rand(10, 1, 64, 64)
        # >>> z = tikho(x)
        # >>> print(z.shape)
        # torch.Size([10, 1, 4096])
    """

    def __init__(
        self,
        acqu: meas.Linear,
        prep,
        sigma: torch.tensor,
        denoi=nn.Identity(),
        *,
        device: torch.device = torch.device("cpu"),
        **tikho_kwargs,
    ):

        tikho = inverse.Tikhonov(acqu, sigma, **tikho_kwargs)
        acqu_modules = OrderedDict({"acqu": acqu})
        recon_modules = OrderedDict({"prep": prep, "tikho": tikho, "denoi": denoi})
        super().__init__(acqu_modules, recon_modules, device=device)

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

        .. important::
            The measurements passed as input must *NOT* be preprocessed.

        Args:
            :attr:`y`: raw measurement vectors. Have shape :math:`(b, c, m)`

        Returns:
            torch.tensor: Reconstructed images. Have shape :math:`(b,c,h,w)` if
            :attr:`reshape_output` is True in the :attr:`kwargs` dictionary (default)
            or :math:`(b,c,hw)` otherwise.

        Example:
            # >>> acqu = spyrit.core.meas.HadamSplit2d(64)
            # >>> prep = spyrit.core.prep.Rescale(1)
            # >>> sigma = torch.ones(64, 64)
            # >>> tikho = TikhoNet(acqu, prep, sigma)
            # >>> x = torch.rand(10, 1, 64, 64)
            # >>> y = acqu(x)
            # >>> z = tikho.reconstruct(y)
            # >>> print(z.shape)
            # torch.Size([10, 1, 64, 64])
        """
        y = self.reconstruct_pinv(y)
        y = self.denoi(y)
        return y

    def reconstruct_pinv(self, y):
        r"""Reconstruct an image from measurements without denoising.

        This method sucessively applies the preprocessing operator :attr:`prep`
        and the Tikhonov regularization operator :attr:`tikho` to the input
        measurement vectors :attr:`x`.

        .. important::
            The measurements passed as input must *NOT* be preprocessed.

        Args:
            :attr:`y`: raw measurement vectors. Have shape :math:`(b, c, m)`

        Returns:
            torch.tensor: Reconstructed images. Have shape :math:`(b,c,h,w)` if
            :attr:`reshape_output` is True in the :attr:`kwargs` dictionary (default)
            or :math:`(b,c,hw)` otherwise.

        Example:
            # >>> acqu = spyrit.core.meas.HadamSplit2d(64)
            # >>> prep = spyrit.core.prep.Rescale(1)
            # >>> sigma = torch.ones(64, 64)
            # >>> tikho = TikhoNet(acqu, prep, sigma)
            # >>> x = torch.rand(10, 1, 64, 64)
            # >>> y = acqu(x)
            # >>> z = tikho.reconstruct_pinv(y)
            # >>> print(z.shape)
            # torch.Size([10, 1, 64, 64])
        """
        # covariance of measurements BEFORE preprocessing
        cov_meas = self.prep.sigma(y)
        cov_meas = torch.diag_embed(cov_meas)

        y = self.prep(y)
        y = self.tikho(y, cov_meas)
        return y

    # def reconstruct_expe(self, x):
    #     r"""Reconstruction (measurement-to-image mapping) for experimental data.

    #     Args:
    #         :attr:`x` (torch.tensor): Raw measurement vectors with shape :math:`(B,C,M)`.

    #     Output:
    #         torch.tensor: Reconstructed images with shape :math:`(B,C,H,W)`
    #     """
    #     # Preprocessing
    #     cov_meas = self.prep.sigma_expe(x)
    #     # x, norm = self.prep.forward_expe(x, self.acqu.meas_op, (-2,-1)) # shape: [*, M]

    #     # Alternative where the mean is computed on each row
    #     x, norm = self.prep.forward_expe(x, self.acqu.meas_op)  # shape: [*, M]

    #     # covariance of measurements
    #     cov_meas = cov_meas / norm**2
    #     cov_meas = torch.diag_embed(cov_meas)

    #     # measurements to image domain processing
    #     x = self.tikho(x, cov_meas)
    #     # x = x.reshape(*x.shape[:-1], self.acqu.meas_op.h, self.acqu.meas_op.w)

    #     # Image domain denoising
    #     x = self.denoi(x)

    #     # Denormalization
    #     x = self.prep.denormalize_expe(x, norm, x.shape[-2], x.shape[-1])

    #     return x, norm


# =============================================================================
class LearnedPGD(nn.Module):
    r"""Learned Proximal Gradient Descent reconstruction network.

    Iterative algorithm that alternates between a gradient step and a proximal step,
    where the proximal operator is replaced by a learned denoiser. The update rule is given by

    .. math::

        x_{k+1} = \texttt{denoi}\left(x_k - \gamma \, H^T (Hx_k - m)\right)

    where :math:`x_k\in\mathbb{R}^N` is the current estimate, :math:`\gamma\in\mathbb{R}` is the step size, :math:`H\in\mathbb{R}^{M\times N}` is the forward model, and :math:`m\in\mathbb{R}^{M}` are the measurements.

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
            - If :attr:`meas_op.N` is available, the step size is estimated as
              :math:`\gamma=1/N` which is true for Hadamard operators.
            - If not, the step size is estimated by computing
              the Lipschitz constant as the largest singular value of the
              Hessian :math:`L=\lambda_{\max}(H^TH)`. If this fails,
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
        >>> from spyrit.core.meas import HadamSplit2d
        >>> from spyrit.core.prep import UnsplitRescale
        >>> from spyrit.core.recon import LearnedPGD
        >>> import torch

        >>> acqu = HadamSplit2d(32, M=400)
        >>> prep = UnsplitRescale()
        >>> recnet = LearnedPGD(acqu, prep)
        >>> x = torch.FloatTensor(10,1,32,32).uniform_(-1, 1)
        >>> z = recnet(x)
        >>> print(z.shape)
        torch.Size([10, 1, 32, 32])

        >>> y = torch.randn(10, 1, 800)
        >>> z = recnet.reconstruct(y)
        >>> print(z.shape)
        torch.Size([10, 1, 32, 32])
    """

    def __init__(
        self,
        acqu: meas.LinearSplit,
        prep: prep.UnsplitRescale,
        denoi=nn.Identity(),
        *,
        iter_stop=3,
        x0=0.5,  # image in [0,1]
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
            >>> from spyrit.core.meas import HadamSplit2d
            >>> from spyrit.core.prep import UnsplitRescale
            >>> from spyrit.core.recon import LearnedPGD
            >>> import torch

            >>> acqu = HadamSplit2d(32, M=400)
            >>> prep = UnsplitRescale()
            >>> recnet = LearnedPGD(acqu, prep)
            >>> x = torch.FloatTensor(10,1,32,32).uniform_(-1, 1)
            >>> z = recnet(x)
            >>> print(z.shape)
            torch.Size([10, 1, 32, 32])
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
            >>> from spyrit.core.meas import HadamSplit2d
            >>> from spyrit.core.prep import UnsplitRescale
            >>> from spyrit.core.recon import LearnedPGD
            >>> import torch

            >>> acqu = HadamSplit2d(32, M=400)
            >>> prep = UnsplitRescale()
            >>> recnet = LearnedPGD(acqu, prep)
            >>> x = torch.FloatTensor(10,1,32,32).uniform_(-1, 1)
            >>> z = recnet.acquire(x)
            >>> print(z.shape)
            torch.Size([10, 1, 800])
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
            >>> from spyrit.core.meas import HadamSplit2d
            >>> from spyrit.core.prep import UnsplitRescale
            >>> from spyrit.core.recon import LearnedPGD

            >>> import torch
            >>> acqu = HadamSplit2d(32, M=400)
            >>> prep = UnsplitRescale()
            >>> recnet = LearnedPGD(acqu, prep)
            >>> y = torch.randn(10, 1, 800)
            >>> z = recnet.reconstruct(y)
            >>> print(z.shape)
            torch.Size([10, 1, 32, 32])
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
        # if self.x0 != 0:
        # if self.x0 != 0.5:
        #     x = self.pinv(m)
        #     # if hasattr(self.acqu, "pinv"):
        #     #     x = self.acqu.pinv(m)

        #     # proximal step (prior)
        #     if isinstance(self.denoi, nn.ModuleList):
        #         x = self.denoi[0](x)
        #     else:
        #         x = self.denoi(x)

        #     if self.res_learn:
        #         z0 = x.detach().clone()
        # else:
        # zero init
        # x = torch.zeros((*x.shape[:-1], *self.acqu.meas_shape), device=x.device)
        # 0.5 init
        x = self.x0 * torch.ones(
            (*x.shape[:-1], *self.acqu.meas_shape), device=x.device
        )

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
