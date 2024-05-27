"""
Reconstruction methods and networks.
"""

import warnings
from typing import Union

import math
import torch

# import torchvision
import torch.nn as nn
import numpy as np

from spyrit.core.meas import Linear, DynamicLinear, HadamSplit
from spyrit.core.noise import NoNoise
from spyrit.core.prep import DirectPoisson, SplitPoisson

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")


# =============================================================================
class PseudoInverse(nn.Module):
    # =========================================================================
    r"""Moore-Penrose pseudoinverse.

    Considering linear measurements :math:`y = Hx`, where :math:`H` is the
    measurement matrix and :math:`x` is a vectorized image, it estimates
    :math:`x` from :math:`y` by computing :math:`\hat{x} = H^\dagger y`, where
    :math:`H` is the Moore-Penrose pseudo inverse of :math:`H`.

    Example:
        >>> H = torch.rand([400,32*32])
        >>> Perm = torch.rand([32*32,32*32])
        >>> meas_op =  HadamSplit(H, Perm, 32, 32)
        >>> y = torch.rand([85,400], dtype=torch.float)
        >>> pinv_op = PseudoInverse()
        >>> x = pinv_op(y, meas_op)
        >>> print(x.shape)
        torch.Size([85, 1024])
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.tensor,
        meas_op: Union[Linear, DynamicLinear],
        **kwargs,
    ) -> torch.tensor:
        r"""Computes pseudo-inverse of measurements.

        Args:
            :attr:`x`: Batch of measurement vectors.

            :attr:`meas_op`: Measurement operator. Any class that
            implements a :meth:`pinv` method can be used, e.g.,
            :class:`~spyrit.core.forwop.HadamSplit`.

        Shape:

            :attr:`x`: :math:`(*, M)`

            :attr:`meas_op`: not applicable

            :attr:`output`: :math:`(*, N)`

        Example:
            >>> H = torch.rand([400,32*32])
            >>> Perm = torch.rand([32*32,32*32])
            >>> meas_op =  HadamSplit(H, Perm, 32, 32)
            >>> y = torch.rand([85,400], dtype=torch.float)
            >>> pinv_op = PseudoInverse()
            >>> x = pinv_op(y, meas_op)
            >>> print(x.shape)
            torch.Size([85, 1024])
        """
        return meas_op.pinv(x, **kwargs)


# =============================================================================
class TikhonovMeasurementPriorDiag(nn.Module):
    # =========================================================================
    r"""
    Tikhonov regularisation with prior in the measurement domain.

    Considering linear measurements :math:`m = Hx \in\mathbb{R}^M`, where :math:`H = GF` is the measurement matrix and :math:`x\in\mathbb{R}^N` is a vectorized image, it estimates :math:`x` from :math:`m` by approximately minimizing

    .. math::
        \|m - GFx \|^2_{\Sigma^{-1}_\alpha} + \|F(x - x_0)\|^2_{\Sigma^{-1}}

    where :math:`x_0\in\mathbb{R}^N` is a mean image prior, :math:`\Sigma\in\mathbb{R}^{N\times N}` is a covariance prior, and :math:`\Sigma_\alpha\in\mathbb{R}^{M\times M}` is the measurement noise covariance. The matrix :math:`G\in\mathbb{R}^{M\times N}` is a subsampling matrix.

    .. note::
        The class is instantiated from :math:`\Sigma`, which represents the covariance of :math:`Fx`.

    Args:
        - :attr:`sigma`:  covariance prior with shape :math:`N` x :math:`N`
        - :attr:`M`: number of measurements :math:`M`


    Attributes:
        :attr:`comp`: The learnable completion layer initialized as
        :math:`\Sigma_1 \Sigma_{21}^{-1}`. This layer is a :class:`nn.Linear`

        :attr:`denoi`: The learnable denoising layer initialized from
        :math:`\Sigma_1`.

    Example:
        >>> sigma = torch.rand([32*32, 32*32])
        >>> recon_op = TikhonovMeasurementPriorDiag(sigma, 400)
    """

    def __init__(self, sigma: torch.tensor, M: int):
        super().__init__()

        N = sigma.shape[0]

        self.comp = nn.Linear(M, N - M, False)
        self.denoi = Denoise_layer(M)

        var_prior = sigma.diag()[:M]

        self.denoi.weight.data = torch.sqrt(var_prior)
        self.denoi.weight.data = self.denoi.weight.data.float()
        self.denoi.weight.requires_grad = False

        Sigma1 = sigma[:M, :M]
        Sigma21 = sigma[M:, :M]
        # W = Sigma21 @ torch.linalg.inv(Sigma1)
        W = torch.linalg.solve(Sigma1.T, Sigma21.T).T

        self.comp.weight.data = W
        self.comp.weight.data = self.comp.weight.data.float()
        self.comp.weight.requires_grad = False

    def forward(
        self, x: torch.tensor, x_0: torch.tensor, var: torch.tensor, meas_op: HadamSplit
    ) -> torch.tensor:
        r"""
        Computes the Tikhonov regularization with prior in the measurement domain.

        We approximate the solution as:

        .. math::
            \hat{x} = x_0 + F^{-1} \begin{bmatrix} m_1 \\ m_2\end{bmatrix}

        with :math:`m_1 = D_1(D_1 + \Sigma_\alpha)^{-1} (m - GF x_0)` and
        :math:`m_2 = \Sigma_1 \Sigma_{21}^{-1} m_1`, where
        :math:`\Sigma = \begin{bmatrix} \Sigma_1 & \Sigma_{21}^\top \\ \Sigma_{21} & \Sigma_2\end{bmatrix}`
        and  :math:`D_1 =\textrm{Diag}(\Sigma_1)`. Assuming the noise
        covariance :math:`\Sigma_\alpha` is diagonal, the matrix inversion
        involved in the computation of :math:`m_1` is straightforward.

        This is an approximation to the exact solution

        .. math::
            \hat{x} &= x_0 + F^{-1}\begin{bmatrix}\Sigma_1 \\ \Sigma_{21} \end{bmatrix}
                      [\Sigma_1 + \Sigma_\alpha]^{-1} (m - GF x_0)


        See Lemma B.0.5 of the PhD dissertation of A. Lorente Mur (2021):
        https://theses.hal.science/tel-03670825v1/file/these.pdf

        Args:
            - :attr:`x`: A batch of measurement vectors :math:`m`
            - :attr:`x_0`: A batch of prior images :math:`x_0`
            - :attr:`var`: A batch of measurement noise variances :math:`\Sigma_\alpha`
            - :attr:`meas_op`: A measurement operator that provides :math:`GF` and :math:`F^{-1}`

        Shape:
            - :attr:`x`: :math:`(*, M)`
            - :attr:`x_0`: :math:`(*, N)`
            - :attr:`var` :math:`(*, M)`
            - Output: :math:`(*, N)`

        Example:
            >>> B, H, M = 85, 32, 512
            >>> sigma = torch.rand([H**2, H**2])
            >>> recon_op = TikhonovMeasurementPriorDiag(sigma, M)
            >>> Ord = torch.ones((H,H))
            >> meas = HadamSplit(M, H, Ord)
            >>> y = torch.rand([B,M], dtype=torch.float)
            >>> x_0 = torch.zeros((B, H**2), dtype=torch.float)
            >>> var = torch.zeros((B, M), dtype=torch.float)
            >>> x = recon_op(y, x_0, var, meas)
            torch.Size([85, 1024])
        """
        x = x - meas_op.forward_H(x_0)
        y1 = torch.mul(self.denoi(var), x)
        y2 = self.comp(y1)

        y = torch.cat((y1, y2), -1)
        x = x_0 + meas_op.inverse(y)
        return x


# =============================================================================
class Denoise_layer(nn.Module):
    # =========================================================================
    r"""Wiener filter that assumes additive white Gaussian noise.

    .. math::
        y = \sigma_\text{prior}^2/(\sigma^2_\text{prior} + \sigma^2_\text{meas}) x,
        where :math:`\sigma^2_\text{prior}` is the variance prior and
        :math:`\sigma^2_\text{meas}` is the variance of the measurement,
        x is the input vector and y is the output vector.

    Args:
        :attr:`M` (int): size of incoming vector

    Shape:
        - Input: :math:`(*, M)`.
        - Output: :math:`(*, M)`.

    Attributes:
        :attr:`weight`:
        The learnable standard deviation prior :math:`\sigma_\text{prior}` of
        shape :math:`(M, 1)`. The values are initialized from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k = 1/M`.

        :attr:`in_features`:
        The number of input features equal to :math:`M`.

    Example:
        >>> m = Denoise_layer(30)
        >>> input = torch.randn(128, 30)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, M: int):
        super(Denoise_layer, self).__init__()
        self.in_features = M
        self.weight = nn.Parameter(torch.Tensor(M))
        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Resets the standard deviation prior :math:`\sigma_\text{prior}`.

        The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`,
        where :math:`k = 1/M`. They are stored in the :attr:`weight` attribute.
        """
        nn.init.uniform_(self.weight, 0, 2 / math.sqrt(self.in_features))

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        r"""
        Applies a transformation to the incoming data: :math:`y = A^2/(A^2+x)`.

        :math:`x` is the input tensor (see :attr:`inputs`) and :math:`A` is the
        standard deviation prior (see :attr:`self.weight`).

        Args:
            :attr:`inputs` (torch.tensor): input tensor :math:`x` of shape
            :math:`(N, *, in\_features)`

        Returns:
            torch.tensor: The transformed data :math:`y` of shape
            :math:`(N, in\_features)`

        Shape:

        """
        return self.tikho(inputs, self.weight)

    def extra_repr(self):
        return "in_features={}".format(self.in_features)

    @staticmethod
    def tikho(inputs: torch.tensor, weight: torch.tensor) -> torch.tensor:
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Applies a transformation to the incoming data: :math:`y = A^2/(A^2+x)`.

        :math:`x` is the input tensor (see :attr:`inputs`) and :math:`A` is the
        standard deviation prior (see :attr:`weight`).

        Args:
            :attr:`inputs` (torch.tensor): input tensor :math:`x` of shape
            :math:`(N, *, in\_features)`

            :attr:`weight` (torch.tensor): standard deviation prior :math:`A` of
            shape :math:`(in\_features)`

        Returns:
            torch.tensor: The transformed data :math:`y` of shape
            :math:`(N, in\_features)`

        Shape:
            - :attr:`inputs`: :math:`(N, *, in\_features)` where `*` means any number of
              additional dimensions - Variance of measurements
            - :attr:`weight`: :math:`(in\_features)` - corresponds to the standard deviation
              of our prior.
            - :attr:`output`: :math:`(N, in\_features)`
        """
        a = weight**2  # prefer to square it, because when learnt, it can go to the
        # negative, which we do not want to happen.
        # TO BE Potentially done : square inputs.
        b = a + inputs
        return a / b


# -----------------------------------------------------------------------------
# |                      RECONSTRUCTION NETWORKS                              |
# -----------------------------------------------------------------------------


# =============================================================================
class PinvNet(nn.Module):
    # =========================================================================
    r"""Pseudo inverse reconstruction network

    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`)

        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)

        :attr:`denoi` (optional): Image denoising operator
        (see :class:`~spyrit.core.nnet`).
        Default :class:`~spyrit.core.nnet.Identity`

    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(B,C,H,W)`
        corresponding to the batch size, number of channels, height, and width.

        :attr:`output`: Reconstructed images with shape :math:`(B,C,H,W)`
        corresponding to the batch size, number of channels, height, and width.

    Attributes:
        :attr:`Acq`: Acquisition operator initialized as :attr:`noise`

        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`

        :attr:`pinv`: Analytical reconstruction operator initialized as
        :class:`~spyrit.core.recon.PseudoInverse()`

        :attr:`Denoi`: Image denoising operator initialized as :attr:`denoi`


    Example:
        >>> B, C, H, M = 10, 1, 64, 64**2
        >>> Ord = torch.ones((H,H))
        >>> meas = HadamSplit(M, H, Ord)
        >>> noise = NoNoise(meas)
        >>> prep = SplitPoisson(1.0, M, H*H)
        >>> recnet = PinvNet(noise, prep)
        >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
        >>> z = recnet(x)
        >>> print(z.shape)
        >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
        torch.Size([10, 1, 64, 64])
        tensor(5.8912e-06)
    """

    def __init__(self, noise, prep, denoi=nn.Identity()):
        super().__init__()
        self.acqu = noise
        self.prep = prep
        self.pinv = PseudoInverse()
        self.denoi = denoi

    def forward(self, x):
        r"""Full pipeline of reconstrcution network

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
            >>> recnet = PinvNet(noise, prep)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet(x)
            >>> print(z.shape)
            >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
            torch.Size([10, 1, 64, 64])
            tensor(5.8912e-06)
        """
        b, c, _, _ = x.shape

        # Acquisition
        x = x.view(b * c, self.acqu.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        x = self.acqu(x)  # shape x = [b*c, 2*M]

        # Reconstruction
        x = self.reconstruct(x)  # shape x = [bc, 1, h,w]
        return x.view(b, c, self.acqu.meas_op.h, self.acqu.meas_op.w)

    def acquire(self, x):
        r"""Simulates data acquisition

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
        b, c, _, _ = x.shape
        # Acquisition
        x = x.view(b * c, self.acqu.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        return self.acqu(x)  # shape x = [b*c, 2*M]

    def meas2img(self, y):
        """Returns images from raw measurement vectors

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
        m = self.prep(y)
        m = torch.nn.functional.pad(m, (0, self.acqu.meas_op.N - self.acqu.meas_op.M))
        # z = m @ self.acqu.meas_op.get_Perm().T  # old way
        # new way, tested and working :
        z = self.acqu.meas_op.reindex(m, "cols", False)
        return z.view(-1, 1, self.acqu.meas_op.h, self.acqu.meas_op.w)

    def reconstruct(self, x):
        r"""Preprocesses, reconstructs, and denoises raw measurement vectors.

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
        # Denoise image-domain
        return self.denoi(self.reconstruct_pinv(x))

    def reconstruct_pinv(self, x):
        r"""Preprocesses and reconstructs raw measurement vectors.

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
            >>> z = recnet.reconstruct_pinv(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        # Measurement to image domain mapping
        bc, _ = x.shape

        # Preprocessing in the measurement domain
        x = self.prep(x)  # , self.acqu.meas_op) # shape x = [b*c, M]

        # measurements to image-domain processing
        x = self.pinv(x, self.acqu.meas_op)  # shape x = [b*c,N]

        # Image-domain denoising
        x = x.view(
            bc, 1, self.acqu.meas_op.h, self.acqu.meas_op.w
        )  # shape x = [b*c,1,h,w]
        return x

    def reconstruct_expe(self, x):
        r"""Reconstruction step of a reconstruction network

        Same as :meth:`reconstruct` reconstruct except that:

        1. The preprocessing step estimates the image intensity for normalization

        2. The output images are "denormalized", i.e., have units of photon counts

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: :math:`(BC,2M)`

            :attr:`output`: :math:`(BC,1,H,W)`
        """
        # x of shape [b*c, 2M]
        bc, _ = x.shape

        # Preprocessing
        x, N0_est = self.prep.forward_expe(x, self.acqu.meas_op)  # shape x = [b*c, M]
        print(N0_est)

        # measurements to image domain processing
        x = self.pinv(x, self.acqu.meas_op)  # shape x = [b*c,N]

        # Image domain denoising
        x = x.view(
            bc, 1, self.acqu.meas_op.h, self.acqu.meas_op.w
        )  # shape x = [b*c,1,h,w]
        x = self.denoi(x)  # shape x = [b*c,1,h,w]
        print(x.max())

        # Denormalization
        x = self.prep.denormalize_expe(
            x, N0_est, self.acqu.meas_op.h, self.acqu.meas_op.w
        )
        return x


# =============================================================================
class DCNet(nn.Module):
    # =========================================================================
    r"""Denoised completion reconstruction network.

    This is a four step reconstruction method:

    #. Denoising in the measurement domain.
    #. Estimation of the missing measurements from the denoised ones.
    #. Image-domain mapping.
    #. (Learned) Denoising in the image domain.

    The first three steps corresponds to Tikhonov regularisation. Typically, only the last step involves learnable parameters.

    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`)

        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)

        :attr:`sigma`: Covariance prior (for details, see the
        :class:`~spyrit.core.recon.TikhonovMeasurementPriorDiag()` class)

        :attr:`denoi` (optional): Image denoising operator
        (see :class:`~spyrit.core.nnet`).
        Default :class:`~spyrit.core.nnet.Identity`

    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(B,C,H,W)`

        :attr:`output`: Reconstructed images with shape :math:`(B,C,H,W)`

    Attributes:
        :attr:`Acq`: Acquisition operator initialized as :attr:`noise`

        :attr:`PreP`: Preprocessing operator initialized as :attr:`prep`

        :attr:`DC_Layer`: Data consistency layer initialized as :attr:`tikho`

        :attr:`Denoi`: Image denoising operator initialized as :attr:`denoi`


    Example:
        >>> B, C, H, M = 10, 1, 64, 64**2
        >>> Ord = torch.ones((H,H))
        >>> meas = HadamSplit(M, H, Ord)
        >>> noise = NoNoise(meas)
        >>> prep = SplitPoisson(1.0, M, H*H)
        >>> sigma = torch.rand([H**2, H**2])
        >>> recnet = DCNet(noise,prep,sigma)
        >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
        >>> z = recnet(x)
        >>> print(z.shape)
        torch.Size([10, 1, 64, 64])
    """

    def __init__(
        self,
        noise: NoNoise,
        prep: Union[DirectPoisson, SplitPoisson],
        sigma: torch.tensor,
        denoi=nn.Identity(),
    ):
        super().__init__()
        self.Acq = noise
        self.prep = prep
        self.denoi = denoi
        sigma = sigma.to(torch.float32)

        # old way
        # Perm = noise.meas_op.get_Perm().cpu().T #.numpy()
        # sigma = Perm @ sigma @ Perm.T

        # new way
        # Ord = noise.meas_op.Ord
        # Perm = torch.from_numpy(samp.Permutation_Matrix(noise.meas_op.Ord)).to(torch.float32)
        # sigma = samp.sort_by_significance(sigma, Ord, 'rows', True)
        # sigma = samp.sort_by_significance(sigma, Ord, 'cols', False)
        sigma = noise.reindex(sigma, "rows", False)
        sigma = noise.reindex(sigma, "cols", True)
        sigma_perm = sigma

        # save in tikho
        self.tikho = TikhonovMeasurementPriorDiag(sigma_perm, noise.meas_op.M)

    def forward(self, x):
        r"""Full pipeline of the reconstruction network

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
            >>> sigma = torch.rand([H**2, H**2])
            >>> recnet = DCNet(noise,prep,sigma)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """

        b, c, _, _ = x.shape

        # Acquisition
        x = x.view(b * c, self.Acq.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        x = self.Acq(x)  # shape x = [b*c, 2*M]

        # Reconstruction
        x = self.reconstruct(x)  # shape x = [bc, 1, h,w]
        x = x.view(b, c, self.Acq.meas_op.h, self.Acq.meas_op.w)

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
            >>> sigma = torch.rand([H**2, H**2])
            >>> recnet = DCNet(noise,prep,sigma)
            >>> x = torch.FloatTensor(B,C,H,H).uniform_(-1, 1)
            >>> z = recnet.acquire(x)
            >>> print(z.shape)
            torch.Size([10, 8192])
        """

        b, c, _, _ = x.shape

        # Acquisition
        x = x.view(b * c, self.Acq.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        x = self.Acq(x)  # shape x = [b*c, 2*M]

        return x

    def reconstruct(self, x):
        r"""Reconstruction step of a reconstruction network

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: raw measurement vectors with shape :math:`(BC,2M)`

            :attr:`output`: reconstructed images with shape :math:`(BC,1,H,W)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> sigma = torch.rand([H**2, H**2])
            >>> recnet = DCNet(noise,prep,sigma)
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        # x of shape [b*c, 2M]
        bc, _ = x.shape

        # Preprocessing
        var_noi = self.prep.sigma(x)
        x = self.prep(x)  # shape x = [b*c, M]

        # measurements to image domain processing
        x_0 = torch.zeros((bc, self.Acq.meas_op.N), device=x.device)
        x = self.tikho(x, x_0, var_noi, self.Acq.meas_op)
        x = x.view(
            bc, 1, self.Acq.meas_op.h, self.Acq.meas_op.w
        )  # shape x = [b*c,1,h,w]

        # Image domain denoising
        x = self.denoi(x)

        return x

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
        # x of shape [b*c, 2M]
        bc, _ = x.shape

        # Preprocessing expe
        var_noi = self.prep.sigma_expe(x)
        x, N0_est = self.prep.forward_expe(x, self.Acq.meas_op)  # x <- x/N0_est
        x = x / self.prep.gain
        norm = self.prep.gain * N0_est

        # variance of preprocessed measurements
        var_noi = torch.div(
            var_noi, (norm.view(-1, 1).expand(bc, self.Acq.meas_op.M)) ** 2
        )

        # measurements to image domain processing
        x_0 = torch.zeros((bc, self.Acq.meas_op.N), device=x.device)
        x = self.tikho(x, x_0, var_noi, self.Acq.meas_op)
        x = x.view(
            bc, 1, self.Acq.meas_op.h, self.Acq.meas_op.w
        )  # shape x = [b*c,1,h,w]

        # Image domain denoising
        x = self.denoi(x)  # shape x = [b*c,1,h,w]

        # Denormalization
        x = self.prep.denormalize_expe(x, norm, self.Acq.meas_op.h, self.Acq.meas_op.w)

        return x


# =============================================================================
class DCDRUNet(DCNet):
    # =========================================================================
    r"""Denoised completion reconstruction network based on DRUNet wich concatenates a
        noise level map to the input

    .. math:


    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`)

        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)

        :attr:`sigma`: UPDATE!! Tikhonov reconstruction operator of type
        :class:`~spyrit.core.recon.TikhonovMeasurementPriorDiag()`

        :attr:`denoi` (optional): Image denoising operator
        (see :class:`~spyrit.core.nnet`).
        Default :class:`~spyrit.core.nnet.Identity`

        :attr:`noise_level` (optional): Noise level in the range [0, 255], default is noise_level=5


    Input / Output:
        :attr:`input`: Ground-truth images with concatenated noise level map with
         shape :math:`(B,C+1,H,W)`

        :attr:`output`: Reconstructed images with shape :math:`(B,C,H,W)`

    Attributes:
        :attr:`Acq`: Acquisition operator initialized as :attr:`noise`

        :attr:`PreP`: Preprocessing operator initialized as :attr:`prep`

        :attr:`DC_Layer`: Data consistency layer initialized as :attr:`tikho`

        :attr:`Denoi`: Image (DRUNet architecture type) denoising operator
        initialized as :attr:`denoi`


    Example:
        >>> B, C, H, M = 10, 1, 64, 64**2
        >>> Ord = torch.ones((H,H))
        >>> meas = HadamSplit(M, H, Ord)
        >>> noise = NoNoise(meas)
        >>> prep = SplitPoisson(1.0, M, H*H)
        >>> sigma = torch.rand([H**2, H**2])
        >>> n_channels = 1                   # 1 for grayscale image
        >>> model_drunet_path = './spyrit/drunet/model_zoo/drunet_gray.pth'
        >>> denoi_drunet = drunet(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
            downsample_mode="strideconv", upsample_mode="convtranspose")
        >>> recnet = DCDRUNet(noise,prep,sigma,denoi_drunet)
        >>> z = recnet(x)
        >>> print(z.shape)
        torch.Size([10, 1, 64, 64])
    """

    def __init__(self, noise, prep, sigma, denoi=nn.Identity(), noise_level=5):
        super().__init__(noise, prep, sigma, denoi)
        self.register_buffer("noise_level", torch.FloatTensor([noise_level / 255.0]))

    def reconstruct(self, x):
        r"""Reconstruction step of a reconstruction network

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: raw measurement vectors with shape :math:`(BC,2M)`

            :attr:`output`: reconstructed images with shape :math:`(BC,1,H,W)`

        Example:
            >>> B, C, H, M = 10, 1, 64, 64**2
            >>> Ord = torch.ones((H,H))
            >>> meas = HadamSplit(M, H, Ord)
            >>> noise = NoNoise(meas)
            >>> prep = SplitPoisson(1.0, M, H*H)
            >>> sigma = torch.rand([H**2, H**2])
            >>> n_channels = 1                   # 1 for grayscale image
            >>> model_drunet_path = './spyrit/drunet/model_zoo/drunet_gray.pth'
            >>> denoi_drunet = drunet(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                downsample_mode="strideconv", upsample_mode="convtranspose")
            >>> recnet = DCDRUNet(noise,prep,sigma,denoi_drunet)
            >>> x = torch.rand((B*C,2*M), dtype=torch.float)
            >>> z = recnet.reconstruct(x)
            >>> print(z.shape)
            torch.Size([10, 1, 64, 64])
        """
        # x of shape [b*c, 2M]

        bc, _ = x.shape

        # Preprocessing
        var_noi = self.prep.sigma(x)
        x = self.prep(x)  # shape x = [b*c, M]

        # measurements to image domain processing
        x_0 = torch.zeros((bc, self.Acq.meas_op.N), device=x.device)
        x = self.tikho(x, x_0, var_noi, self.Acq.meas_op)
        x = x.view(
            bc, 1, self.Acq.meas_op.h, self.Acq.meas_op.w
        )  # shape x = [b*c,1,h,w]

        # Image domain denoising
        x = self.concat_noise_map(x)
        x = self.denoi(x)

        return x

    def concat_noise_map(self, x):
        r"""Concatenation of noise level map to reconstructed images

        Args:
            :attr:`x`: reconstructed images from the reconstruction layer

        Shape:
            :attr:`x`: reconstructed images with shape :math:`(BC,1,H,W)`

            :attr:`output`: reconstructed images with concatenated noise level map with shape :math:`(BC,2,H,W)`
        """

        b, c, h, w = x.shape
        x = 0.5 * (x + 1)
        x = torch.cat((x, self.noise_level.expand(b, 1, h, w)), dim=1)
        return x

    def set_noise_level(self, noise_level):
        r"""Reset noise level value

        Args:
            :attr:`noise_level`: noise level value in the range [0, 255]

        Shape:
            :attr:`noise_level`: float value noise level :math:`(1)`

            :attr:`output`: noise level tensor with shape :math:`(1)`
        """
        self.noise_level = torch.FloatTensor([noise_level / 255.0])


# =============================================================================
class PositiveParameters(nn.Module):
    # ==========================================================================
    def __init__(self, size, val_min=1e-6):
        super(PositiveParameters, self).__init__()
        self.val_min = torch.tensor(val_min)
        self.params = nn.Parameter(
            torch.abs(val_min * torch.ones(size, 1)), requires_grad=True
        )

    def forward(self):
        return torch.abs(self.params)


# =============================================================================
class PositiveMonoIncreaseParameters(PositiveParameters):
    # =========================================================================
    def __init__(self, size, val_min=0.000001):
        super().__init__(size, val_min)

    def forward(self):
        # cumsum in opposite order
        return super().forward().cumsum(dim=0).flip(dims=[0])


# =============================================================================
class UPGD(PinvNet):
    # =========================================================================
    def __init__(
        self,
        noise,
        prep,
        denoi=nn.Identity(),
        num_iter=6,
        lamb=1e-5,
        lamb_min=1e-6,
        split=False,
    ):
        super(UPGD, self).__init__(noise, prep, denoi)
        self.num_iter = num_iter
        self.lamb = lamb
        self.lamb_min = lamb_min
        # Set a trainable tensor for the regularization parameter with dimension num_iter
        # and constrained to be positive with clamp(min=0.0, max=None)
        self.lambs = PositiveMonoIncreaseParameters(
            num_iter, lamb_min
        )  # shape lambs = [num_iter,1]
        # self.noise = noise
        self.split = split

    def reconstruct(self, x):
        r"""Reconstruction step of a reconstruction network

        Same as :meth:`reconstruct` reconstruct except that:

            1. The regularization parameter is trainable

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: :math:`(BC,2M)`

            :attr:`output`: :math:`(BC,1,H,W)`
        """

        # Measurement operator
        # if self.split:
        #    meas = super().Acq.meas_op
        # else:
        # meas = self.Acq.meas_op
        meas = self.acqu.meas_op

        # x of shape [b*c, 2M]
        bc, _ = x.shape

        # First estimate: Pseudo inverse
        # Preprocessing in the measurement domain
        x = self.prep(x)  # [5, 1024]

        # Save measurements
        m = x.clone()  # [5, 1024]

        # measurements to image domain processing
        x = self.pinv(x, self.acqu.meas_op)  # [5, 4096]         # shape x = [b*c,N]
        # x = x.view(bc,1,self.acqu.meas_op.h, self.acqu.meas_op.w)   # shape x = [b*c,1,h,w]

        # Unroll network
        # Ensure step size is positive and monotonically decreasing and larger than self.lamb!
        lambs = self.lambs()
        for n in range(self.num_iter):
            # Projection onto the measurement space
            proj = self.acqu.meas_op.forward_H(x)  # [5, 1024]

            # Residual
            res = proj - m  # [5, 1024]

            # Gradient step
            x = x + lambs[n] * self.acqu.meas_op.H_adjoint(res)  # [5, 4096]

            # Denoising step
            x = x.view(
                bc, 1, self.acqu.meas_op.h, self.acqu.meas_op.w
            )  # [5, 1, 64, 64]
            x = self.denoi(x)
            x = x.view(bc, self.acqu.meas_op.N)  # [5, 4096]
        return x
