# ==================================================================================
class Preprocess_shift_poisson(nn.Module):  # header needs to be updated!
    # ==================================================================================
    r"""Preprocess the measurements acquired using shifted patterns corrupted
    by Poisson noise

        Computes:
        m = (2 m_shift - m_offset)/N_0
        var = 4*Diag(m_shift + m_offset)/alpha**2
        Warning: dark measurement is assumed to be the 0-th entry of raw measurements

        Args:
            - :math:`alpha`: noise level
            - :math:`M`: number of measurements
            - :math:`N`: number of image pixels

        Shape:
            - Input1: scalar
            - Input2: scalar
            - Input3: scalar

        Example:
            >>> PSP = Preprocess_shift_poisson(9, 400, 32*32)
    """

    def __init__(self, alpha, M, N):
        super().__init__()
        self.alpha = alpha
        self.N = N
        self.M = M

    def forward(self, x: torch.tensor, meas_op: Linear) -> torch.tensor:
        r"""

        Warning:
            - The offset measurement is the 0-th entry of the raw measurements.

        Args:
            - :math:`x`: Batch of images in Hadamard domain shifted by 1
            - :math:`meas_op`: Forward_operator

        Shape:
            - Input: :math:`(b*c, M+1)`
            - Output: :math:`(b*c, M)`

        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Forward_operator(Hsub)
            >>> x = torch.tensor(np.random.random([10, 400+1]), dtype=torch.float)
            >>> y_PSP = PSP(x, FO)
            >>> print(y_PSP.shape)
            torch.Size([10, 400])

        """
        y = self.offset(x)
        x = 2 * x[:, 1:] - y.expand(
            x.shape[0], self.M
        )  # Warning: dark measurement is the 0-th entry
        x = x / self.alpha
        x = 2 * x - meas_op.H(
            torch.ones(x.shape[0], self.N).to(x.device)
        )  # to shift images in [-1,1]^N
        return x

    def sigma(self, x):
        r"""
        Args:
            - :math:`x`: Batch of images in Hadamard domain shifted by 1

        Shape:
            - Input: :math:`(b*c, M+1)`

        Example:
            >>> x = torch.tensor(np.random.random([10, 400+1]), dtype=torch.float)
            >>> sigma_PSP = PSP.sigma(x)
            >>> print(sigma_PSP.shape)
            torch.Size([10, 400])
        """
        # input x is a set of measurement vectors with shape (b*c, M+1)
        # output is a set of measurement vectors with shape (b*c,M)
        y = self.offset(x)
        x = 4 * x[:, 1:] + y.expand(x.shape[0], self.M)
        x = x / (self.alpha**2)
        x = 4 * x  # to shift images in [-1,1]^N
        return x

    def cov(self, x):  # return a full matrix ? It is such that Diag(a) + b
        return x

    def sigma_from_image(self, x, meas_op):  # should check this!
        # input x is a set of images with shape (b*c, N)
        # input meas_op is a Forward_operator
        x = meas_op.H(x)
        y = self.offset(x)
        x = x[:, 1:] + y.expand(x.shape[0], self.M)
        x = x / (self.alpha)  # here the alpha contribution is not squared.
        return x

    def offset(self, x):
        r"""Get offset component from bach of shifted images.

        Args:
            - :math:`x`: Batch of shifted images

        Shape:
            - Input: :math:`(bc, M+1)`
            - Output: :math:`(bc, 1)`

        Example:
            >>> x = torch.tensor(np.random.random([10, 400+1]), dtype=torch.float)
            >>> y = PSP.offset(x)
            >>> print(y.shape)
            torch.Size([10, 1])

        """
        y = x[:, 0, None]
        return y


# ==================================================================================
class Preprocess_pos_poisson(nn.Module):  # header needs to be updated!
    # ==================================================================================
    r"""Preprocess the measurements acquired using positive (shifted) patterns
    corrupted by Poisson noise

    The output value of the layer with input size :math:`(B*C, M)` can be
    described as:

    .. math::
        \text{out}((B*C)_i, M_j}) = 2*\text{input}((B*C)_i, M_j}) -
        \sum_{k = 1}^{M-1} \text{input}((B*C)_i, M_k})

    The output size of the layer is :math:`(B*C, M)`, which is the imput size


        Warning:
            dark measurement is assumed to be the 0-th entry of raw measurements

        Args:
            - :math:`alpha`: noise level
            - :math:`M`: number of measurements
            - :math:`N`: number of image pixels

        Shape:
            - Input1: scalar
            - Input2: scalar
            - Input3: scalar

        Example:
            >>> PPP = Preprocess_pos_poisson(9, 400, 32*32)

    """

    def __init__(self, alpha, M, N):
        super().__init__()
        self.alpha = alpha
        self.N = N
        self.M = M

    def forward(self, x: torch.tensor, meas_op: Linear) -> torch.tensor:
        r"""
        Args:
            - :math:`x`: noise level
            - :math:`meas_op`: Forward_operator

        Shape:
            - Input1: :math:`(bc, M)`
            - Input2: None
            - Output: :math:`(bc, M)`

        Example:
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> meas_op = Forward_operator(Hsub)
            >>> x = torch.tensor(np.random.random([10, 400]), dtype=torch.float)
            >>> y = PPP(x, meas_op)
            torch.Size([10, 400])

        """
        y = self.offset(x)
        x = 2 * x - y.expand(-1, self.M)
        x = x / self.alpha
        x = 2 * x - meas_op.H(
            torch.ones(x.shape[0], self.N).to(x.device)
        )  # to shift images in [-1,1]^N
        return x

    def offset(self, x):
        r"""Get offset component from bach of shifted images.

        Args:
            - :math:`x`: Batch of shifted images

        Shape:
            - Input: :math:`(bc, M)`
            - Output: :math:`(bc, 1)`

        Example:
            >>> x = torch.tensor(np.random.random([10, 400]), dtype=torch.float)
            >>> y = PPP.offset(x)
            >>> print(y.shape)
            torch.Size([10, 1])

        """
        y = 2 / (self.M - 2) * x[:, 1:].sum(dim=1, keepdim=True)
        return y
