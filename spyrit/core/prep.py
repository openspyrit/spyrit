"""
Preprocessing operators applying affine transformations to the measurements.

There are two classes in this module: :class:`DirectPoisson` and
:class:`SplitPoisson`. The first one is used for direct measurements (i.e.
without splitting the measurement matrix in its positive and negative parts),
while the second one is used for split measurements.
"""

from typing import Union, Tuple

import torch
import torch.nn as nn

from spyrit.core.meas import LinearSplit, HadamSplit  # , Linear


# =============================================================================
class DirectPoisson(nn.Module):
    r"""
    Preprocess the raw data acquired with a direct measurement operator assuming
    Poisson noise. It also compensates for the affine transformation applied
    to the images to get positive intensities.

    It computes :math:`m = \frac{2}{\alpha}y - H1` and the variance
    :math:`\sigma^2 = 4\frac{y}{\alpha^{2}}`, where :math:`y = Hx` are obtained
    using a direct linear measurement operator (see :mod:`spyrit.core.Linear`),
    :math:`\alpha` is the image intensity, and 1 is the all-ones vector.

    Args:
        :attr:`alpha`: maximun image intensity :math:`\alpha` (in counts)

        :attr:`meas_op`: measurement operator (see :mod:`~spyrit.core.meas`)


    Example:
        >>> H = torch.rand([400,32*32])
        >>> meas_op =  Linear(H)
        >>> prep_op = DirectPoisson(1.0, meas_op)

    """

    def __init__(self, alpha: float, meas_op):
        super().__init__()
        self.alpha = alpha
        self.meas_op = meas_op

        self.M = meas_op.M
        self.N = meas_op.N
        self.h = meas_op.h
        self.w = meas_op.w

        self.max = nn.MaxPool2d((self.h, self.w))
        # self.register_buffer("H_ones", meas_op(torch.ones((1, self.N))))

    # generate H_ones on the fly as it is memmory intensive and easy to compute
    # ?? Why does it returns float64 ??
    @property
    def H_ones(self):
        return self.meas_op.H.sum(dim=-1).to(self.device)

    @property
    def device(self):
        return self.meas_op.device

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Preprocess measurements to compensate for the affine image normalization

        It computes :math:`\frac{2}{\alpha}x - H1`, where H1 represents the
        all-ones vector.

        Args:
            :attr:`x`: batch of measurement vectors

        Shape:
            x: :math:`(B, M)` where :math:`B` is the batch dimension

            meas_op: the number of measurements :attr:`meas_op.M` should match
            :math:`M`.

            Output: :math:`(B, M)`

        Example:
            >>> x = torch.rand([10,400], dtype=torch.float)
            >>> H = torch.rand([400,32*32])
            >>> meas_op =  Linear(H)
            >>> prep_op = DirectPoisson(1.0, meas_op)
            >>> m = prep_op(x)
            >>> print(m.shape)
            torch.Size([10, 400])
        """
        # normalize
        # H_ones = self.H_ones.expand(x.shape[0], self.M)
        x = 2 * x / self.alpha - self.H_ones.to(x.dtype).expand(x.shape)
        return x

    def sigma(self, x: torch.tensor) -> torch.tensor:
        r"""Estimates the variance of the preprocessed measurements

        The variance is estimated as :math:`\frac{4}{\alpha^2} x`

        Args:
            :attr:`x`: batch of measurement vectors

        Shape:
            :attr:`x`: :math:`(B,M)` where :math:`B` is the batch dimension

            Output: :math:`(B, M)`

        Example:
            >>> x = torch.rand([10,400], dtype=torch.float)
            >>> v = prep_op.sigma(x)
            >>> print(v.shape)
            torch.Size([10, 400])

        """
        # *4 to account for the image normalized [-1,1] -> [0,1]
        return 4 * x / (self.alpha**2)

    def denormalize_expe(
        self, x: torch.tensor, beta: torch.tensor, h: int = None, w: int = None
    ) -> torch.tensor:
        r"""Denormalize images from the range [-1;1] to the range [0; :math:`\beta`]

        It computes :math:`m = \frac{\beta}{2}(x+1)`, where
        :math:`\beta` is the normalization factor, that can be different for each
        image in the batch.

        Args:
            - :attr:`x` (torch.tensor): Batch of images
            - :attr:`beta` (torch.tensor): Normalization factor. It should have
            the same shape as the batch.
            - :attr:`h` (int, optional): Image height. If None, it is
            deduced from the shape of :attr:`x`. Defaults to None.
            - :attr:`w` (int): Image width. If None, it is deduced from the
            shape of :attr:`x`. Defaults to None.

        Shape:
            - :attr:`x`: :math:`(*, h, w)` where :math:`*` indicates any batch
            dimensions
            - :attr:`beta`: :math:`(*)` or :math:`(1)` if the same for all
            images
            - :attr:`h`: int
            - :attr:`w`: int
            - Output: :math:`(*, h, w)`

        Example:
            >>> x = torch.rand([10, 1, 32,32], dtype=torch.float)
            >>> beta = 9*torch.rand([10])
            >>> y = split_op.denormalize_expe(x, beta, 32, 32)
            >>> print(y.shape)
            torch.Size([10, 1, 32, 32])
        """
        if h is None:
            h = x.shape[-2]
        if w is None:
            w = x.shape[-1]

        if beta.numel() == 1:
            beta = beta.expand(x.shape)
        else:
            # Denormalization
            beta = beta.reshape(*beta.shape, 1, 1)
            beta = beta.expand((*beta.shape[:-2], h, w))

        return (x + 1) / 2 * beta

    def unsplit(self, x: torch.tensor, mode: str = "diff") -> torch.tensor:
        """Unsplits measurements by combining odd and even indices.

        The parameter `mode` can be either 'diff' or 'sum'. The first one
        computes the difference between the even and odd indices, while the
        second one computes the sum.

        Args:
            x (torch.tensor): Measurements, can have any shape.

            mode (str): 'diff' or 'sum'. If 'diff', the difference between the
            even and odd indices is computed. If 'sum', the sum is computed.
            Defaults to 'diff'.

        Returns:
            torch.tensor: The input tensor with the even and odd indices
            of the last dimension combined (either by difference or sum).
        """
        if mode == "diff":
            return x[..., 0::2] - x[..., 1::2]
        elif mode == "sum":
            return x[..., 0::2] + x[..., 1::2]
        else:
            raise ValueError("mode should be either 'diff' or 'sum'")


# =============================================================================
class SplitPoisson(DirectPoisson):
    r"""
    Preprocess the raw data acquired with a split measurement operator assuming
    Poisson noise.  It also compensates for the affine transformation applied
    to the images to get positive intensities.

    It computes :math:`m = \frac{y_{+}-y_{-}}{\alpha} - H1` and the variance
    :math:`var = \frac{2(y_{+} + y_{-})}{\alpha^{2}}`, where
    :math:`y_{+} = H_{+}x` and :math:`y_{-} = H_{-}x` are obtained using a
    split measurement operator (see :mod:`spyrit.core.LinearSplit`),
    :math:`\alpha` is the image intensity, and 1 is the all-ones vector.

    Args:
        alpha (float): maximun image intensity :math:`\alpha` (in counts)

        :attr:`meas_op`: measurement operator (see :mod:`~spyrit.core.meas`)


    Example:
        >>> H = torch.rand([400,32*32])
        >>> meas_op =  LinearSplit(H)
        >>> split_op = SplitPoisson(10, meas_op)

    Example 2:
        >>> Perm = torch.rand([32,32])
        >>> meas_op = HadamSplit(400, 32,  Perm)
        >>> split_op = SplitPoisson(10, meas_op)

    """

    def __init__(self, alpha: float, meas_op):
        super().__init__(alpha, meas_op)

    @property
    def even_index(self):
        return range(0, 2 * self.M, 2)

    @property
    def odd_index(self):
        return range(1, 2 * self.M, 2)

    # @property
    # def H_ones(self):
    #     return self.unsplit(super().H_ones, mode="diff")

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""
        Preprocess to compensates for image normalization and splitting of the
        measurement operator.

        It computes :math:`\frac{x[0::2]-x[1::2]}{\alpha} - H1`

        Args:
            :attr:`x`: batch of measurement vectors

        Shape:
            x: :math:`(*, 2M)` where :math:`*` indicates one or more dimensions

            meas_op: the number of measurements :attr:`meas_op.M` should match
            :math:`M`.

            Output: :math:`(*, M)`

        Example:
            >>> x = torch.rand([10,2*400], dtype=torch.float)
            >>> H = torch.rand([400,32*32])
            >>> meas_op =  LinearSplit(H)
            >>> split_op = SplitPoisson(10, meas_op)
            >>> m = split_op(x)
            >>> print(m.shape)
            torch.Size([10, 400])

        Example 2:
            >>> x = torch.rand([10,2*400], dtype=torch.float)
            >>> Perm = torch.rand([32,32])
            >>> meas_op = HadamSplit(400, 32,  Perm)
            >>> split_op = SplitPoisson(10, meas_op)
            >>> m = split_op(x)
            >>> print(m.shape)
            torch.Size([10, 400])
        """
        # s = x.shape[:-1] + torch.Size([self.M])  # torch.Size([*,M])
        # H_ones = self.H_ones.expand(s)
        return super().forward(self.unsplit(x, mode="diff"))

    def forward_expe(
        self, x: torch.tensor, meas_op: Union[LinearSplit, HadamSplit]
    ) -> Tuple[torch.tensor, torch.tensor]:
        r"""Preprocess to compensate for image normalization and splitting of the
        measurement operator.

        It computes :math:`m = \frac{x[0::2]-x[1::2]}{\alpha}`, where
        :math:`\alpha = \max H^\dagger (x[0::2]-x[1::2])`.

        Contrary to :meth:`~forward`, the image intensity :math:`\alpha`
        is estimated from the pseudoinverse of the unsplit measurements. This
        method is typically called for the reconstruction of experimental
        measurements, while :meth:`~forward` is called in simulations.

        The method returns a tuple containing both :math:`m` and :math:`\alpha`

        Args:
            :attr:`x`: batch of measurement vectors

            :attr:`meas_op`: measurement operator (required to estimate
            :math:`\alpha`)

            Output (:math:`m`, :math:`\alpha`): preprocess measurement and estimated
            intensities.

        Shape:
            x: :math:`(B, 2M)` where :math:`B` is the batch dimension

            meas_op: the number of measurements :attr:`meas_op.M` should match
            :math:`M`.

            :math:`m`: :math:`(B, M)`

            :math:`\alpha`: :math:`(B)`

        Example:
            >>> x = torch.rand([10,2*400], dtype=torch.float)
            >>> Perm = torch.rand([32,32])
            >>> meas_op = HadamSplit(400, 32,  Perm)
            >>> split_op = SplitPoisson(10, meas_op)
            >>> m, alpha = split_op.forward_expe(x, meas_op)
            >>> print(m.shape)
            >>> print(alpha.shape)
            torch.Size([10, 400])
            torch.Size([10])
        """
        x = self.unsplit(x, mode="diff")

        # estimate alpha
        x_pinv = meas_op.pinv(x)
        alpha = self.max(x_pinv).squeeze(-1)  # shape is now (b, c, 1)

        # normalize
        alpha = alpha.expand(x.shape)
        x = torch.div(x, alpha)
        x = 2 * x - self.H_ones.expand(x.shape)

        alpha = alpha[..., 0]  # shape is (b, c)

        return x, alpha

    def sigma(self, x: torch.tensor) -> torch.tensor:
        r"""Estimates the variance of the preprocessed measurements

        The variance is estimated as :math:`\frac{4}{\alpha^2} (x[0::2]+x[1::2])`

        Args:
            :attr:`x`: batch of images in the Hadamard domain

        Shape:
            - Input: :math:`(*,2*M)` :math:`*` indicates one or more dimensions
            - Output: :math:`(*, M)`

        Example:
            >>> x = torch.rand([10,2*400], dtype=torch.float)
            >>> v = split_op.sigma(x)
            >>> print(v.shape)
            torch.Size([10, 400])

        """
        return super().sigma(self.unsplit(x, mode="sum"))

    def set_expe(self, gain=1.0, mudark=0.0, sigdark=0.0, nbin=1.0):
        r"""
        Sets experimental parameters of the sensor

        Args:
            - :attr:`gain` (float): gain (in count/electron)
            - :attr:`mudark` (float): average dark current (in counts)
            - :attr:`sigdark` (float): standard deviation or dark current (in counts)
            - :attr:`nbin` (float): number of raw bin in each spectral channel (if input x results from the sommation/binning of the raw data)

        Example:
            >>> split_op.set_expe(gain=1.6)
            >>> print(split_op.gain)
            1.6
        """
        self.gain = gain
        self.mudark = mudark
        self.sigdark = sigdark
        self.nbin = nbin

    def sigma_expe(self, x: torch.tensor) -> torch.tensor:
        r"""
        Estimates the variance of the measurements that are compensated for
        splitting but **NOT** for image normalization


        Args:
            :attr:`x`: Batch of images in the Hadamard domain.

        Shape:
            Input: :math:`(B,2*M)` where :math:`B` is the batch dimension

            Output: :math:`(B, M)`

        Example:
            >>> x = torch.rand([10,2*32*32], dtype=torch.float)
            >>> split_op.set_expe(gain=1.6)
            >>> v = split_op.sigma_expe(x)
            >>> print(v.shape)
            torch.Size([10, 400])
        """
        x = self.unsplit(x, mode="sum")
        x = (
            self.gain * (x - 2 * self.nbin * self.mudark)
            + 2 * self.nbin * self.sigdark**2
        )
        x = 4 * x  # to get the cov of an image in [-1,1], not in [0,1]

        return x

    def sigma_from_image(
        self, x: torch.tensor, meas_op: Union[LinearSplit, HadamSplit]
    ) -> torch.tensor:
        r"""
        Estimates the variance of the preprocessed measurements corresponding
        to images through a measurement operator

        The variance is estimated as
        :math:`\frac{4}{\alpha} \{(Px)[0::2] + (Px)[1::2]\}`

        Args:
            :attr:`x`: Batch of images

            :attr:`meas_op`: Measurement operator

        Shape:
            :attr:`x`: :math:`(*,N)`

            :attr:`meas_op`: An operator such that :attr:`meas_op.N` :math:`=N`
            and :attr:`meas_op.M` :math:`=M`

            Output: :math:`(*, M)`

        Example:
            >>> x = torch.rand([10,2*400], dtype=torch.float)
            >>> Perm = torch.rand([32,32])
            >>> meas_op = HadamSplit(400, 32,  Perm)
            >>> split_op = SplitPoisson(10, meas_op)
            >>> v = split_op.sigma_from_image(x, meas_op)
            >>> print(v.shape)
            torch.Size([10, 400])

        """
        x = meas_op(x)
        x = self.unsplit(x, mode="sum")
        x = 4 * x / self.alpha  # here alpha should not be squared
        return x
