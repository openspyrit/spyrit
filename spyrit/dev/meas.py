# ==================================================================================
class Linear_shift(Linear):
    # ==================================================================================
    r"""Linear with shifted pattern matrix of size :math:`(M+1,N)` and :math:`Perm` matrix of size :math:`(N,N)`.

    Args:
        - Hsub: subsampled Hadamard matrix
        - Perm: Permuation matrix

    Shape:
        - Input1: :math:`(M, N)`
        - Input2: :math:`(N, N)`

    Example:
        >>> Hsub = np.array(np.random.random([400,32*32]))
        >>> Perm = np.array(np.random.random([32*32,32*32]))
        >>> FO_Shift = Linear_shift(Hsub, Perm)

    """

    def __init__(self, Hsub, Perm):
        super().__init__(Hsub)

        # Todo: Use index rather than permutation (see misc.walsh_hadamard)
        self.Perm = nn.Linear(self.N, self.N, False)
        self.Perm.weight.data = torch.from_numpy(Perm.T)
        self.Perm.weight.data = self.Perm.weight.data.float()
        self.Perm.weight.requires_grad = False

        H_shift = torch.cat((torch.ones((1, self.N)), (self.Hsub.weight.data + 1) / 2))

        self.H_shift = nn.Linear(self.N, self.M + 1, False)
        self.H_shift.weight.data = H_shift  # include the all-one pattern
        self.H_shift.weight.data = self.H_shift.weight.data.float()  # keep ?
        self.H_shift.weight.requires_grad = False

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Applies Linear transform such that :math:`y = \begin{bmatrix}{1}\\{H_{sub}}\end{bmatrix}x`.

        Args:
            :math:`x`: batch of images.

        Shape:
            - Input: :math:`(b*c, N)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`N` the number of pixels in the image.
            - Output: :math:`(b*c, M+1)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`M+1` the number of measurements + 1.

        Example:
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = FO_Shift(x)
            >>> print(y.shape)
            torch.Size([10, 401])
        """
        # input x is a set of images with shape (b*c, N)
        # output input is a set of measurement vector with shape (b*c, M+1)
        x = self.H_shift(x)
        return x

        # x_shift = super().forward(x) - x_dark.expand(x.shape[0],self.M) # (H-1/2)x


# ==================================================================================
class Linear_pos(Linear):
    # ==================================================================================
    r"""Linear with Permutation Matrix :math:`Perm` of size :math:`(N,N)`.

    Args:
        - Hsub: subsampled Hadamard matrix
        - Perm: Permuation matrix

    Shape:
        - Input1: :math:`(M, N)`
        - Input2: :math:`(N, N)`

    Example:
        >>> Hsub = np.array(np.random.random([400,32*32]))
        >>> Perm = np.array(np.random.random([32*32,32*32]))
        >>> meas_op_pos = Linear_pos(Hsub, Perm)
    """

    def __init__(self, Hsub, Perm):
        super().__init__(Hsub)

        # Todo: Use index rather than permutation (see misc.walsh_hadamard)
        self.Perm = nn.Linear(self.N, self.N, False)
        self.Perm.weight.data = torch.from_numpy(Perm.T)
        self.Perm.weight.data = self.Perm.weight.data.float()
        self.Perm.weight.requires_grad = False

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Computes :math:`y` according to :math:`y=0.5(H_{sub}x+\sum_{j=1}^{N}x_{j})` where :math:`j` is the pixel (column) index of :math:`x`.

        Args:
            :math:`x`: Batch of images.

        Shape:
            - Input: :math:`(b*c, N)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`N` the number of pixels in the image.
            - Output: :math:`(b*c, M)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`M` the number of measurements.

        Example:
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = meas_op_pos(x)
            >>> print(y.shape)
            torch.Size([100, 400])
        """
        # input x is a set of images with shape (b*c, N)
        # output is a set of measurement vectors with shape (b*c, M)

        # compute 1/2(H+1)x = 1/2 HX + 1/2 1x
        x = super().forward(x) + x.sum(dim=1, keepdim=True).expand(-1, self.M)
        x *= 0.5

        return x


# ==================================================================================
class Linear_shift_had(Linear_shift):
    # ==================================================================================
    r"""Linear_shift operator with inverse method.

    Args:
        - Hsub: subsampled Hadamard matrix
        - Perm: Permuation matrix

    Shape:
        - Input1: :math:`(M, N)`
        - Input2: :math:`(N, N)`.

    Example:
        >>> Hsub = np.array(np.random.random([400,32*32]))
        >>> Perm = np.array(np.random.random([32*32,32*32]))
        >>> FO_Shift_Had = Linear_shift_had(Hsub, Perm)
    """

    def __init__(self, Hsub, Perm):
        super().__init__(Hsub, Perm)

    def inverse(self, x: torch.tensor, n: Union[None, int] = None) -> torch.tensor:
        r"""Inverse transform such that :math:`x = \frac{1}{N}H_{sub}y`.

        Args:
            :math:`x`: Batch of completed measurements.

        Shape:
            - Input: :math:`(b*c, N)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`N` the number of measurements.
            - Output: :math:`(b*c, N)` with :math:`b` the batch size, :math:`c` the number of channels, and :math:`N` the number of reconstructed. pixels.

        Example:
            >>> x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> x_reconstruct = FO_Shift_Had.inverse(y_pad)
            >>> print(x_reconstruct.shape)
            torch.Size([10, 1024])
        """
        # rearrange the terms + inverse transform
        # maybe needs to be initialised with a permutation matrix as well!
        # Permutation matrix may be sparsified when sparse tensors are no longer in
        # beta (as of pytorch 1.11, it is still in beta).

        # --> Use index rather than permutation (see misc.walsh_hadamard)

        # input x is a set of **measurements** with shape (b*c, N)
        # output is a set of **images** with shape (b*c, N)
        bc, N = x.shape
        x = self.Perm(x)

        if n is None:
            n = int(np.sqrt(N))

        # Inverse transform
        x = x.reshape(bc, 1, n, n)
        x = (
            1 / self.N * walsh2_torch(x)
        )  # todo: initialize with 1D transform to speed up
        x = x.reshape(bc, N)
        return x
