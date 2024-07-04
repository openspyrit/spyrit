import torch
import torch.nn as nn
import numpy as np
import math


# ==================================================================================
class PseudoInverseStore(nn.Module):
    # ==================================================================================
    r"""Moore-Penrose Pseudoinverse

    Considering linear measurements :math:`y = Hx`, where :math:`H` is the
    measurement matrix and :math:`x` is a vectorized image, it estimates
    :math:`x` from :math:`y` by computing :math:`\hat{x} = H^\dagger y`, where
    :math:`H^\dagger` is the Moore-Penrose pseudo inverse of :math:`H`.

    Args:
        :attr:`meas_op`: Measurement operator that defines :math:`H`. Any class
        that implements a :meth:`get_H` method can be used, e.g.,
        :class:`~spyrit.core.forwop.Linear`.

        :attr:`reg` (optional): Regularization parameter (cutoff for small
        singular values, see :mod:`numpy.linal.pinv`).

        :attr:`learn` (optional): Option to learn the pseudo inverse matrix. By
        default the pseudo inverse is frozen during training.

    Attributes:
        :attr:`H_pinv`: The learnable pseudo inverse matrix of shape
        :math:`(N,M)` initialized as :math:`H^\dagger`.

    .. note::
        Contrary to :class:`~spyrit.core.recon.PseudoInverse`, the pseudoinverse
        matrix is stored and therefore learnable

    Example:
        >>> H = np.random.rand(400,32*32)
        >>> meas_op = Linear(H)
        >>> recon_op = PseudoInverseStore(meas_op)
    """

    def __init__(self, meas_op: Linear, reg: float = 1e-15, learn: bool = False):
        H = meas_op.get_H()
        M, N = H.shape
        H_pinv = np.linalg.pinv(H, rcond=reg)

        super().__init__()
        self.H_pinv = nn.Linear(M, N, False)
        self.H_pinv.weight.data = torch.from_numpy(H_pinv).float()
        self.H_pinv.weight.requires_grad = learn

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Compute pseudo-inverse of measurements.

        Args:
            - :attr:`x`: Batch of measurement vectors.

        Shape:
            - :attr:`x`: :math:`(*, M)`
            - :attr:`output`: :math:`(*, N)`

        Example:
            >>> H = np.random.rand(400,32*32)
            >>> meas_op = Linear(H)
            >>> recon_op = PseudoInverseStore(meas_op)
            >>> x = torch.rand([10,400], dtype=torch.float)
            >>> y = recon_op(x)
            >>> print(y.shape)
            torch.Size([10, 1024])
        """
        x = self.H_pinv(x)
        return x


# ==================================================================================
class PseudoInverseStore2(nn.Module):
    # ==================================================================================
    r"""Moore-Penrose Pseudoinverse

    Considering linear measurements :math:`y = Hx`, where :math:`H` is the
    measurement matrix and :math:`x` is a vectorized image, it estimates
    :math:`x` from :math:`y` by computing :math:`\hat{x} = H^\dagger y`, where
    :math:`H^\dagger` is the Moore-Penrose pseudo inverse of :math:`H`.

    Args:
        :attr:`meas_op`: Measurement operator that defines :math:`H`. Any class
        that implements a :meth:`get_H` method can be used, e.g.,
        :class:`~spyrit.core.forwop.LinearSplit`.

        :attr:`reg` (optional): Regularization parameter (cutoff for small
        singular values, see :mod:`numpy.linal.pinv`).

        :attr:`learn` (optional): Option to learn the pseudo inverse matrix. By
        default the pseudo inverse is frozen during training.

    Attributes:
        :attr:`H_pinv`: The learnable pseudo inverse matrix of shape
        :math:`(N,M)` initialized as :math:`H^\dagger`.

    .. note::
        Contrary to :class:`~spyrit.core.recon.PseudoInverse`, the pseudoinverse
        matrix is stored and therefore learnable

    Example 1:
        >>> H = np.random.rand(24,64)
        >>> meas_op = LinearSplit(H)
        >>> recon_op = PseudoInverseStore2(meas_op)

    Example 2:
        >>> M = 63
        >>> N = 64
        >>> B = 1
        >>> H = walsh_matrix(N)
        >>> meas_op = LinearSplit(H)
        >>> recon_op = PseudoInverseStore2(meas_op)

    """

    def __init__(self, meas_op: LinearSplit, reg: float = 1e-15, learn: bool = False):
        H = meas_op.get_H()
        M, N = H.shape
        H_pinv = np.linalg.pinv(H, rcond=reg)

        super().__init__()

        self.H_pinv = nn.Linear(M, N, False)
        self.H_pinv.weight.data = torch.from_numpy(H_pinv).float()
        self.H_pinv.weight.requires_grad = learn

    def forward(self, x: torch.tensor) -> torch.tensor:
        r"""Compute pseudo-inverse of measurements.

        Args:
            - :attr:`x`: Batch of measurement vectors.

        Shape:
            - :attr:`x`: :math:`(*, M)`
            - :attr:`output`: :math:`(*, N)`

        Example 1:
            >>> H = np.random.rand(24,64)
            >>> meas_op = LinearSplit(H)
            >>> recon_op = PseudoInverseStore2(meas_op)
            >>> x = torch.rand([10,24,92], dtype=torch.float)
            >>> y = recon_op(x)
            >>> print(y.shape)
            torch.Size([10, 64, 92])

        Example 2:
            >>> M = 63
            >>> N = 64
            >>> B = 1
            >>> H = walsh_matrix(N)
            >>> meas_op = LinearSplit(H)
            >>> noise_op = NoNoise(meas_op)
            >>> split_op = SplitRowPoisson(1.0, M, 92) # splitrowpoisson has been removed !!
            >>> recon_op = PseudoInverseStore2(meas_op)
            >>> x = torch.FloatTensor(B,N,92).uniform_(-1, 1)
            >>> y = noise_op(x)
            >>> m = split_op(y, meas_op)
            >>> z = recon_op(m)
            >>> print(z.shape)
            >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
            torch.Size([1, 64, 92])
            tensor(0.1338)
        """
        x = torch.transpose(x, 1, 2)  # swap last two dimensions
        x = self.H_pinv(x)
        x = torch.transpose(x, 1, 2)  # swap last two dimensions
        return x


# ==================================================================================
class learned_measurement_to_image(nn.Module):
    # ==================================================================================
    r"""Measurement to image.

    Args:
        - :math:`N`: number of pixels
        - :math:`M`: number of measurements

    Shape:
        - Input1: scalar
        - Input2: scalar

    Example:
        >>> Meas_to_Img = learned_measurement_to_image(32*32, 400)

    """

    def __init__(self, N, M):
        super().__init__()
        # FO = Forward Operator
        self.FC = nn.Linear(M, N, True)  # FC - fully connected

    def forward(self, x: torch.tensor, FO: Linear = None) -> torch.tensor:
        r"""Measurement to image.

        Args:
            - :math:`x`: Batch of measurements
            - :math:`FO`: Linear

        Shape:
            - Input1: :math:`(b*c, M)`
            - Input2: non-applicable
            - Output: :math:`(b*c, N)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> y = Meas_to_Img(x, FO)
            >>> print(y.shape)
            torch.Size([10, 1024])

        """
        # input (b*c, M)
        # output (b*c, N)
        x = self.FC(x)
        return x


# ==================================================================================
class gradient_step(nn.Module):
    # ==================================================================================
    r"""Gradient step

    Args:
        - :math:`mu`: Mean ?

    Shape:
        - Input: scalar

    Example:
        >>> GS = gradient_step()

    """

    def __init__(self, mu=0.1):
        super().__init__()
        # FO = Forward Operator
        # -- Pseudo-inverse to determine levels of noise.
        self.mu = nn.Parameter(
            torch.tensor([mu], requires_grad=True)
        )  # need device maybe?
        # if user wishes to keep mu constant, then he can change requires gard to false

    def forward(
        self, x: torch.tensor, x_0: torch.tensor, FO: HadamSplit
    ) -> torch.tensor:
        r"""Gradient step

        Args:
            - :math:`x`: measurement vector
            - :math:`x_{0}`: previous estimate
            - :math:`FO`: Linear

        Shape:
            - Input1: :math:`(b*c, M)`
            - Input2: :math:`(b*c, N)`
            - Output: :math:`(b*c, N)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> x_0 = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = GS(x, x_0, FO)
            >>> print(y.shape)
            torch.Size([10, 1024])

        """
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        # z = x_0 - mu*A^T(A*x_0-x)
        x = FO.Forward_op(x_0) - x
        x = x_0 - self.mu * FO.adjoint(x)
        return x


# ==================================================================================
class Tikhonov_cg(nn.Module):
    # ==================================================================================
    r"""
    Args:
        - None

    Example:
        >>> TIK = Tikhonov_cg()

    """

    def __init__(self, n_iter=5, mu=0.1, eps=1e-6):
        super().__init__()
        # FO = Forward Operator - Works for ANY forward operator
        self.n_iter = n_iter
        self.mu = nn.Parameter(
            torch.tensor([float(mu)], requires_grad=True)
        )  # need device maybe?
        # if user wishes to keep mu constant, then he can change requires gard to false
        self.eps = eps
        # self.FO = FO

    def A(self, x: torch.tensor, FO: Linear) -> torch.tensor:
        r"""
        Args:
            - :math:`x`: Batch of measurements
            - :math:`FO`: Linear

        Shape:
            - Input1: :math:`(b*c, M)`
            - Output: :math:`(b*c, M)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> y = TIK.A(x)
            >>> print(y.shape)
            torch.Size([10, 400])

        """
        return FO.Forward_op(FO.adjoint(x)) + self.mu * x

    def CG(
        self, y: torch.tensor, FO: Linear, shape: tuple, device: torch.device
    ) -> torch.tensor:
        r"""
        Args:
            - :math:`y`: Batch of measurements
            - :math:`FO`: Linear
            - :math:`shape`: tensor shape
            - device: torch device (cpu or gpu)

        Shape:
            - Input1: :math:`(b*c, M)`
            - Output: :math:`(b*c, M)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> y = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            >>> x = TIK.CG(y, FO, y.shape, device)
            >>> print(x.shape)
            torch.Size([10, 400])

        """
        x = torch.zeros(shape).to(device)
        r = y - self.A(x, FO)
        c = r.clone()
        kold = torch.sum(r * r)
        a = torch.ones((1))
        for i in range(self.n_iter):
            if (
                a > self.eps
            ):  # Necessary to avoid numerical issues with a = 0 -> a = NaN
                Ac = self.A(c, FO)
                cAc = torch.sum(c * Ac)
                a = kold / cAc
                x += a * c
                r -= a * Ac
                k = torch.sum(r * r)
                b = k / kold
                c = r + b * c
                kold = k
        return x

    def forward(self, x: torch.tensor, x_0: torch.tensor, FO: Linear) -> torch.tensor:
        r"""

        Args:
            - :math:`x`: measurement vector
            - :math:` x_{0}`: previous estimate
            - :math:`FO`: Linear

        Shape:
            - Input1: :math:`(bc, M)`
            - Input2: :math:`(bc, N)`
            - Output: :math:`(bc, N)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> x_0 = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = TIK(x, x_0, FO)
            >>> print(y.shape)
            torch.Size([10, 1024])

        """
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        # n_step steps of Conjugate gradient to solve \|Ax-b\|^2 + mu \|x - x_0\|^2
        # FO could be inside the class

        y = x - FO.Forward_op(x_0)
        x = self.CG(y, FO, x.shape, x.device)
        x = x_0 + FO.adjoint(x)
        return x


#
#    def forward(self, x, x_0):
#        # x - input (b*c, M) - measurement vector
#        # x_0 - input (b*c, N) - previous estimate
#        # z - output (b*c, N)
#        # n_step steps of Conjugate gradient to solve \|Ax-b\|^2 + mu \|x - x_0\|^2
#
#        y = x-self.FO.Forward_op(x_0);
#        x = self.CG(y, x.shape, x.device);
#        x = x_0 + self.FO.adjoint(x)
#        return x
#


# ==================================================================================
class Tikhonov_solve(nn.Module):
    # ==================================================================================
    r"""
    Args:
        - :math:`\mu`: noise level

    Shape:
        - Input: scalar

    Example:
        >>> TIK_solve = Tikhonov_solve(mu=0.1)
    """

    def __init__(self, mu=0.1):
        super().__init__()
        # FO = Forward Operator - Needs to be matrix-storing
        # -- Pseudo-inverse to determine levels of noise.
        self.mu = nn.Parameter(
            torch.tensor([float(mu)], requires_grad=True)
        )  # need device maybe?

    def solve(self, x: torch.tensor, FO: Linear) -> torch.tensor:
        r"""
        Agrs:
            - :math:`x`: measurement vector
            - :math:`FO`: Linear

        Shape:
            - Input1: :math:`(bc, M)`
            - Ouput: :math:`(bc, M)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> y = TIK_solve.solve(x, FO)
            >>> print(y.shape)
            torch.Size([10, 400])
        """
        A = FO.Mat() @ torch.transpose(FO.Mat(), 0, 1) + self.mu * torch.eye(FO.M)
        # Can precompute H@H.T to save time!
        A = A.reshape(1, FO.M, FO.M)
        # Instead of reshaping A, reshape x in the batch-final dimension
        # A = A.repeat(x.shape[0],1, 1); # Not optimal in terms of memory
        A = A.expand(x.shape[0], -1, -1)
        # Not optimal in terms of memory
        x = torch.linalg.solve(A, x)
        return x

    def forward(self, x: torch.tensor, x_0: torch.tensor, FO: Linear) -> torch.tensor:
        r"""
        Args:
            - :math:`x`: measurement vector
            - :math:`x_{0}`: previous estimate
            - :math:`FO`: Linear

        Shape:
            - Input1: :math:`(bc, M)`
            - Input2: :math:`(bc, N)`
            - Output: :math:`(bc, N)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> x_0 = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = TIK_solve(x, x_0, FO)
            >>> print(y.shape)
            torch.Size([10, 1024])

        """
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)

        # uses torch.linalg.solve [As of Pytorch 1.9 autograd supports solve!!]
        x = x - FO.Forward_op(x_0)
        x = self.solve(x, FO)
        x = x_0 + FO.adjoint(x)
        return x


# ==================================================================================
class Orthogonal_Tikhonov(nn.Module):
    # ==================================================================================
    r"""
    Args:
        - :math:`\mu`: noise level

    Shape:
        - Input1: scalar

    Example:
        >>> Orth_TIK = Orthogonal_Tikhonov(mu = 0.1)

    """

    def __init__(self, mu=0.1):
        super().__init__()
        # FO = Forward Operator
        # -- Pseudo-inverse to determine levels of noise.
        self.mu = nn.Parameter(
            torch.tensor([float(mu)], requires_grad=True)
        )  # need device maybe?

    def forward(self, x, x_0, FO):
        r"""
        Args:
            - :math:`x`: measurement vector
            - :math:`x_{0}`: previous estimate

        Shape:
            - Input1: :math:`(bc, M)`
            - Input2: :math:`(bc, N)`
            - Output: :math:`(bc, N)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> x_0 = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> y = Orth_TIK(x, x_0, FO)
            >>> print(y.shape)
            torch.Size([10, 1024])

        """
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)

        x = x - FO.Forward_op(x_0)
        x = x * (1 / (FO.N + self.mu))
        # for hadamard, otherwise, line above
        x = FO.adjoint(x) + x_0
        return x


# ==================================================================================
class Generalised_Tikhonov_cg(nn.Module):  # not inheriting from Tikhonov_cg because
    #                           the of the way var is called in CG
    # ==================================================================================
    r"""
    Args:
        - :math:`\sigma_{prior}`: covariance matix
        - :math:`n_{iter}`: number of iterations
        - :math:`\eps`: convergence error

    Shape:
        - Input1: :math:`(N, N)`
        - Input2: scalar
        - Input3: scalar

    Example:

        >>> Sigma_prior = np.array(np.random.random([1024,1024]))
        >>> General_TIK = Generalised_Tikhonov_cg(Sigma_prior, n_iter = 6, eps = 1e-6)

    """

    def __init__(self, Sigma_prior: np.array, n_iter=6, eps=1e-6):
        super().__init__()
        # FO = Forward Operator - Works for ANY forward operator
        # if user wishes to keep mu constant, then he can change requires gard to false
        self.n_iter = n_iter

        self.Sigma_prior = nn.Linear(Sigma_prior.shape[1], Sigma_prior.shape[0], False)
        self.Sigma_prior.weight.data = torch.from_numpy(Sigma_prior)
        self.Sigma_prior.weight.data = self.Sigma_prior.weight.data.float()
        self.Sigma_prior.weight.requires_grad = False
        self.eps = eps

    def A(self, x: torch.tensor, var: torch.tensor, FO: Linear) -> torch.tensor:
        r"""
        Args:
            - :math:`x`: measurement vector
            - :math:`var`: noise variance
            - :math:`FO`: Linear

        Shape:
            - Input1: :math:`(bc, M)`
            - Input2: :math:`(bc, M)`
            - Output: :math:`(bc, M)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> var = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> y = General_TIK.A(x, var, FO)
            print(y.shape)
            torch.Size([10, 400])
        """
        return FO.Forward_op(self.Sigma_prior(FO.adjoint(x))) + torch.mul(x, var)
        # the first part can be precomputed for optimisation

    def CG(
        self,
        y: torch.tensor,
        var: torch.tensor,
        FO: Linear,
        shape: tuple,
        device: torch.device,
    ) -> torch.tensor:
        r"""
        Args:
            - :math:`y`: measurement vector
            - :math:`var`: measurement variance
            - :math:`FO`: Linear
            - shape: measurement vector shape
            - device: cpu or gpu

        Shape:
            - Input1: :math:`(bc, M)`
            - Input2: :math:`(bc, M)`
            - Output: :math:`(bc, M)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> y = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> var = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            >>> x_out = General_TIK.CG(y, var, FO, y.shape, device)
            >>> print(x_out.shape)
            torch.Size([10, 400])

        """
        x = torch.zeros(shape).to(device)
        r = y - self.A(x, var, FO)
        c = r.clone()
        kold = torch.sum(r * r)
        a = torch.ones((1))
        for i in range(self.n_iter):
            if a > self.eps:
                Ac = self.A(c, var, FO)
                cAc = torch.sum(c * Ac)
                a = kold / cAc
                x += a * c
                r -= a * Ac
                k = torch.sum(r * r)
                b = k / kold
                c = r + b * c
                kold = k
        return x

    def forward(
        self, x: torch.tensor, x_0: torch.tensor, var_noise: torch.tensor, FO: Linear
    ) -> torch.tensor:
        r"""
        Args:
            - :math:`x`: measurement vector
            - :math:`x_{0}`: previous estimate
            - :math:`var_noise`: estimated variance of noise
            - :math:`FO`: Linear

        Shape:
            - Input1: :math:`(bc, M)`
            - Input2: :math:`(bc, N)`
            - Input3: :math:`(bc, M)`
            - Output: :math:`(bc, N)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> x_0 = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> var_noise = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> y = General_TIK(x, x_0, var_noise, FO)
            >>> print(y.shape)
            torch.Size([10, 1024])
        """
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # var_noise - input (b*c, M) - estimated variance of noise
        # z - output (b*c, N)
        # n_step steps of Conjugate gradient to solve
        # \|Ax-b\|^2_{sigma_prior^-1} + \|x - x_0\|^2_{var_noise^-1}
        y = x - FO.Forward_op(x_0)
        x = self.CG(y, var_noise, FO, x.shape, x.device)
        x = x_0 + self.Sigma_prior(FO.adjoint(x))
        return x


# ==================================================================================
class Generalised_Tikhonov_solve(nn.Module):
    # ==================================================================================
    r"""
    Args:
        - :math:`\sigma_{prior}`

    Shape:
        - Input: :math:`(N, N)`

    Example:
        >>> Sigma_prior = Sigma_prior = np.array(np.random.random([32*32, 32*32]))
        >>> General_TIK_solve = Generalised_Tikhonov_solve(Sigma_prior)
    """

    def __init__(self, Sigma_prior: np.array) -> nn.Parameter:
        super().__init__()
        # FO = Forward Operator - Needs to be matrix-storing
        # -- Pseudo-inverse to determine levels of noise.
        self.Sigma_prior = nn.Parameter(
            torch.from_numpy(Sigma_prior.astype("float32")), requires_grad=True
        )

    def solve(self, x: torch.tensor, var: torch.tensor, FO: Linear) -> torch.tensor:
        r"""
        Args:
            - :math:`x`: measurement vector
            - :math:`var`: measurement variance
            - :math:`FO`: Forwrad_operator

        Shape:
            - Input1: :math:`(bc, M)`
            - Input2: :math:`(bc, M)`
            - Output: :math:`(bc, M)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> var = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> y = General_TIK_solve.solve(x, var, FO)
            >>> print(y.shape)
            torch.Size([10, 400])
        """
        A = FO.Mat() @ self.Sigma_prior @ torch.transpose(FO.Mat(), 0, 1)
        A = A.reshape(1, FO.M, FO.M)
        # A = A.repeat(x.shape[0],1,1);# this could be precomputed maybe
        # A += torch.diag_embed(var);
        A = A.expand(x.shape[0], -1, -1) + torch.diag_embed(var)
        x = torch.linalg.solve(A, x)
        return x

    def forward(
        self, x: torch.tensor, x_0: torch.tensor, var_noise: torch.tensor, FO: Linear
    ) -> torch.tensor:
        r"""
        Args:
            - :math:`x`: measurement vector
            - :math:`x_{0}`: previous estimate
            - :math:`var_noise`: estimated variance of noise
            - :math:`FO`: Forward_oprator

        Shape:
            - Input1: :math:`(bc, M)`
            - Input2: :math:`(bc, N)`
            - Input3: :math:`(bc, M)`
            - Output: :math:`(bc, N)`

        Example:
            >>> from spyrit.core.forwop import Linear
            >>> Hsub = np.array(np.random.random([400,32*32]))
            >>> FO = Linear(Hsub)
            >>> x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> x_0 = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
            >>> var_noise = torch.tensor(np.random.random([10,400]), dtype=torch.float)
            >>> y = General_TIK_solve(x, x_0, var_noise, FO)
            >>> print(y.shape)
            torch.Size([10, 1024])

        """
        # x - input (b*c, M) - measurement vector
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)

        # uses torch.linalg.solve [As of Pytorch 1.9 autograd supports solve!!]
        # torch linal solve uses (I believe the LU decomposition of matrix A to
        # solve the linear system.

        x = x - FO.Forward_op(x_0)
        x = self.solve(x, var_noise, FO)
        x = x_0 + torch.matmul(self.Sigma_prior, FO.adjoint(x).T).T
        return x


# ===========================================================================================
class List_Generalized_Orthogonal_Tikhonov(nn.Module):
    # ===========================================================================================
    r"""
    Args:
        - :math:`\sigma_{prior_list}`: list of ?
        - :math:`M` : number of measurements
        - :math:`N`: number of pixels per image

    Shape:
        - Input1: :math:`(N, N)`
        - Input2: scalar
        - Input3: scalar

    Example:
        >>> sigma_prior_list = list(np.random.random([1,400]))
        >>> List_General_Orth_TIK = List_Generalized_Orthogonal_Tikhonov(sigma_prior_list, 400, 32*32)
    """

    def __init__(self, sigma_prior_list, M, N, n_comp=None, n_denoi=None):
        super().__init__()
        # FO = Forward Operator - needs foward operator with defined inverse transform
        # -- Pseudo-inverse to determine levels of noise.

        if n_denoi is None:
            n_denoi = len(sigma_prior_list)
        self.n_denoi = n_denoi

        if n_comp is None:
            n_comp = len(sigma_prior_list)
        self.n_comp = n_comp

        comp_list = []
        for i in range(self.n_comp):
            comp_list.append(nn.Linear(M, N - M, False))

            index = min(i, len(sigma_prior_list) - 1)
            Sigma1 = sigma_prior_list[index][:M, :M]
            Sigma21 = sigma_prior_list[index][M:, :M]

            W = Sigma21 @ np.linalg.inv(Sigma1)

            comp_list[i].weight.data = torch.from_numpy(W)
            comp_list[i].weight.data = comp_list[i].weight.data.float()
            comp_list[i].weight.requires_grad = False

        self.comp_list = nn.ModuleList(comp_list)

        denoise_list = []
        for i in range(self.n_denoi):
            denoise_list.append(Denoise_layer(M))

            index = min(i, len(sigma_prior_list) - 1)

            diag_index = np.diag_indices(N)
            var_prior = sigma_prior_list[index][diag_index]
            var_prior = var_prior[:M]

            denoise_list[i].weight.data = torch.from_numpy(np.sqrt(var_prior))
            denoise_list[i].weight.data = denoise_list[i].weight.data.float()
            denoise_list[i].weight.requires_grad = True
        self.denoise_list = nn.ModuleList(denoise_list)

    def forward(self, x, x_0, var, FO, iterate):
        # x - input (b*c, M) - measurement vector
        # var - input (b*c, M) - measurement variance
        # x_0 - input (b*c, N) - previous estimate
        # z - output (b*c, N)
        #

        i = min(iterate, self.n_denoi - 1)
        j = min(iterate, self.n_comp - 1)

        x = x - FO.Forward_op(x_0)
        y1 = torch.mul(self.denoise_list[i](var), x)
        y2 = self.comp_list[j](y1)

        y = torch.cat((y1, y2), -1)
        x = x_0 + FO.inverse(y)
        return x


#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  RECONSTRUCTION NETWORKS
#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%=============================================================================
class PinvStoreNet(nn.Module):
    # =============================================================================
    r"""Pseudo inverse reconstruction network

    .. math:


    Args:
        :attr:`noise`: Acquisition operator (see :class:`~spyrit.core.noise`)

        :attr:`prep`: Preprocessing operator (see :class:`~spyrit.core.prep`)

        :attr:`denoi` (optional): Image denoising operator
        (see :class:`~spyrit.core.nnet`).
        Default :class:`~spyrit.core.nnet.Identity`

    Input / Output:
        :attr:`input`: Ground-truth images with shape :math:`(B,C,H,W)`

        :attr:`output`: Reconstructed images with shape :math:`(B,C,H,W)`

    Attributes:
        :attr:`acqu`: Acquisition operator initialized as :attr:`noise`

        :attr:`prep`: Preprocessing operator initialized as :attr:`prep`

        :attr:`pinv`: Analytical reconstruction operator initialized as
        :class:`~spyrit.core.recon.PseudoInverse()`

        :attr:`Denoi`: Image denoising operator initialized as :attr:`denoi`

    Example:
        >>> H = np.random.random([M, h**2])
        >>> meas =  Linear(H)
        >>> meas.h, meas.w = h, h
        >>> noise = NoNoise(meas)
        >>> prep = DirectPoisson(1.0, meas)
        >>> pinv_net = PinvStoreNet(noise, prep)
        >>> print(z.shape)
        >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
        torch.Size([85, 1, 32, 32])
        tensor(0.0003)
    """

    def __init__(self, noise, prep, reg: float = 1e-15, denoi=nn.Identity()):
        super().__init__()
        self.acqu = noise
        self.prep = prep
        self.pinv = PseudoInverseStore(noise.meas_op, reg)
        self.denoi = denoi

    def forward(self, x):
        r"""Full pipeline of reconstrcution network

        Args:
            :attr:`x`: ground-truth images

        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`

            :attr:`output`: reconstructed images with shape :math:`(B,C,H,W)`

        Example:
            >>> H = np.random.random([M, h**2])
            >>> meas =  Linear(H)
            >>> meas.h, meas.w = h, h
            >>> noise = NoNoise(meas)
            >>> prep = DirectPoisson(1.0, meas)
            >>> pinv_net = PinvStoreNet(noise, prep)
            >>> z = pinv_net(x)
            >>> print(z.shape)
            >>> print(torch.linalg.norm(x - z)/torch.linalg.norm(x))
            torch.Size([85, 1, 32, 32])
            tensor(0.6429)
        """

        b, c, _, _ = x.shape

        # Acquisition
        x = x.reshape(b * c, self.acqu.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        x = self.acqu(x)  # shape x = [b*c, 2*M]

        # Reconstruction
        x = self.reconstruct(x)  # shape x = [bc, 1, h,w]
        x = x.reshape(b, c, self.acqu.meas_op.h, self.acqu.meas_op.w)

        return x

    def acquire(self, x):
        r"""Simulate data acquisition

        Args:
            :attr:`x`: ground-truth images

        Shape:
            :attr:`x`: ground-truth images with shape :math:`(B,C,H,W)`

            :attr:`output`: measurement vectors with shape :math:`(BC,M)`

        Example:
            >>> H = np.random.random([M, h**2])
            >>> meas =  Linear(H)
            >>> meas.h, meas.w = h, h
            >>> noise = NoNoise(meas)
            >>> prep = DirectPoisson(1.0, meas)
            >>> pinv_net = PinvStoreNet(noise, prep)
            >>> x = torch.FloatTensor(B,C,h,h).uniform_(-1, 1)
            >>> z = pinv_net.acquire(x)
            >>> print(z.shape)
            torch.Size([85, 600])
        """

        b, c, _, _ = x.shape

        # Acquisition
        x = x.reshape(b * c, self.acqu.meas_op.N)  # shape x = [b*c,h*w] = [b*c,N]
        x = self.acqu(x)  # shape x = [b*c, 2*M]

        return x

    def reconstruct(self, x):
        r"""Reconstruction step of a reconstruction network

        Args:
            :attr:`x`: raw measurement vectors

        Shape:
            :attr:`x`: :math:`(BC,M)`

            :attr:`output`: :math:`(BC,1,H,W)`

        Example:
            >>> H = np.random.random([M, h**2])
            >>> meas =  Linear(H)
            >>> meas.h, meas.w = h, h
            >>> noise = NoNoise(meas)
            >>> prep = DirectPoisson(1.0, meas)
            >>> pinv_net = PinvStoreNet(noise, prep)
            >>> x = torch.rand((B*C,M), dtype=torch.float)
            >>> z = pinv_net.reconstruct(x)
            >>> print(z.shape)
            torch.Size([85, 1, 32, 32])
        """
        # Measurement to image domain mapping
        bc, _ = x.shape

        # Preprocessing in the measurement domain
        x = self.prep(x)  # shape x = [b*c, M]

        # measurements to image-domain processing
        x = self.pinv(x)  # shape x = [b*c,N]

        # Image-domain denoising
        x = x.reshape(
            bc, 1, self.acqu.meas_op.h, self.acqu.meas_op.w
        )  # shape x = [b*c,1,h,w]
        x = self.denoi(x)

        return x


# ===========================================================================================
class MoDL(nn.Module):
    # ===========================================================================================
    def __init__(self, Acq, PreP, DC_layer, Denoi, n_iter):
        super().__init__()
        self.Acq = Acq
        self.PreP = PreP
        self.DC_layer = DC_layer
        # must be a non-generalized Tikhonov
        self.Denoi = Denoi
        self.n_iter = n_iter

    def forward(self, x):
        # x - of shape [b,c,h,w]
        b, c, h, w = x.shape

        # Acquisition
        x = x.reshape(b * c, h * w)
        # shape x = [b*c,h*w] = [b*c,N]
        x_0 = torch.zeros_like(x).to(x.device)
        x = self.Acq(x)
        #  shape x = [b*c, 2*M]

        # Preprocessing
        m = self.PreP(x, self.Acq.FO)
        # shape x = [b*c, M]

        # Data consistency layer
        # measurements to the image domain
        for i in range(self.n_iter):
            x = self.DC_layer(m, x_0, self.Acq.FO)
            # shape x = [b*c, N]
            # Image-to-image mapping via convolutional networks
            # Image domain denoising
            x = x.reshape(b * c, 1, h, w)
            x = self.Denoi(x)
            # shape stays the same
            x = x.reshape(b * c, h * w)
            # shape x = [b*c,h*w] = [b*c,N]
            x_0 = x
        x = self.DC_layer(m, x_0, self.Acq.FO)
        # shape x = [b*c, N]
        x = x.reshape(b, c, h, w)
        return x

    def forward_mmse(self, x):
        # x - of shape [b,c,h,w]
        b, c, h, w = x.shape

        # Acquisition
        x = x.reshape(b * c, h * w)
        # shape x = [b*c,h*w] = [b*c,N]
        x_0 = torch.zeros_like(x).to(x.device)
        x = self.Acq(x)
        #  shape x = [b*c, 2*M]

        # Preprocessing
        x = self.PreP(x, self.Acq.FO)
        # shape x = [b*c, M]

        # Data consistency layer
        # measurements to the image domain
        x = self.DC_layer(x, x_0, self.Acq.FO)
        # shape x = [b*c, N]

        # Image-to-image mapping via convolutional networks
        # Image domain denoising
        x = x.reshape(b * c, 1, h, w)
        return x

    def reconstruct(self, x, h=None, w=None):
        # x - of shape [b,c, 2M]
        b, c, M2 = x.shape

        if h is None:
            h = int(np.sqrt(self.Acq.FO.N))

        if w is None:
            w = int(np.sqrt(self.Acq.FO.N))

        x = x.reshape(b * c, M2)
        x_0 = torch.zeros((b * c, self.Acq.FO.N)).to(x.device)

        # Preprocessing
        sigma_noi = self.PreP.sigma(x)
        m = self.PreP(x, self.Acq.FO)
        # shape x = [b*c, M]
        # Data consistency layer
        # measurements to the image domain
        for i in range(self.n_iter):
            x = self.DC_layer(m, x_0, self.Acq.FO)
            # shape x = [b*c, N]
            # Image-to-image mapping via convolutional networks
            # Image domain denoising
            x = x.reshape(b * c, 1, h, w)
            x = self.Denoi(x)
            # shape stays the same
            x = x.reshape(b * c, h * w)
            # shape x = [b*c,h*w] = [b*c,N]
            x_0 = x

        x = self.DC_layer(m, x_0, self.Acq.FO)
        # shape x = [b*c, N]
        x = x.reshape(b, c, h, w)
        return x


# ===========================================================================================
class EM_net(nn.Module):
    # ===========================================================================================
    def __init__(self, Acq, PreP, DC_layer, Denoi, n_iter, est_var=True):
        super().__init__()
        self.Acq = Acq
        self.PreP = PreP
        self.DC_layer = DC_layer
        # must be a tikhonov-list
        self.Denoi = Denoi
        # Must be a denoi-list
        self.n_iter = n_iter

    def forward(self, x):
        # x - of shape [b,c,h,w]
        b, c, h, w = x.shape

        # Acquisition
        x = x.reshape(b * c, h * w)
        # shape x = [b*c,h*w] = [b*c,N]
        x_0 = torch.zeros_like(x).to(x.device)
        x = self.Acq(x)
        #  shape x = [b*c, 2*M]

        # Preprocessing
        var_noi = self.PreP.sigma(x)
        m = self.PreP(x, self.Acq.FO)
        # shape x = [b*c, M]

        # Data consistency layer
        # measurements to the image domain
        for i in range(self.n_iter):
            if self.est_var:
                var_noi = self.PreP.sigma_from_image(x, self.Acq.FO)
            x = self.DC_layer(m, x_0, self.Acq.FO, iterate=i)
            # shape x = [b*c, N]
            # Image-to-image mapping via convolutional networks
            # Image domain denoising
            x = x.reshape(b * c, 1, h, w)
            x = self.Denoi(x, iterate=i)
            # shape stays the same
            x = x.reshape(b * c, h * w)
            # shape x = [b*c,h*w] = [b*c,N]
            x_0 = x
        x = x.reshape(b, c, h, w)
        return x


## ==================================================================================
# class Tikhonov_cg_test(nn.Module):
## ==================================================================================
#    def __init__(self, FO, n_iter = 5, mu = 0.1, eps = 1e-6):
#        super().__init__()
#        # FO = Forward Operator - Works for ANY forward operator
#        self.n_iter = n_iter;
#        self.mu = nn.Parameter(torch.tensor([float(mu)], requires_grad=True)) #need device maybe?
#        # if user wishes to keep mu constant, then he can change requires gard to false
#        self.eps = eps;
#        self.FO = FO
#
#    def A(self,x, FO):
#        return FO.Forward_op(FO.adjoint(x)) + self.mu*x
#
#    def CG(self, y, FO, shape, device):
#        x = torch.zeros(shape).to(device);
#        r = y - self.A(x, FO);
#        c = r.clone()
#        kold = torch.sum(r * r)
#        a = torch.ones((1));
#        for i in range(self.n_iter):
#            if a>self.eps : # Necessary to avoid numerical issues with a = 0 -> a = NaN
#                Ac = self.A(c, FO)
#                cAc =  torch.sum(c * Ac)
#                a =  kold / cAc
#                x += a * c
#                r -= a * Ac
#                k = torch.sum(r * r)
#                b = k / kold
#                c = r + b * c
#                kold = k
#        return x
#
#    def forward(self, x, x_0, FO):
#        # x - input (b*c, M) - measurement vector
#        # x_0 - input (b*c, N) - previous estimate
#        # z - output (b*c, N)
#        # n_step steps of Conjugate gradient to solve \|Ax-b\|^2 + mu \|x - x_0\|^2
#        y = x-FO.Forward_op(x_0);
#        print(id(FO))
#        print(FO.Hsub.weight.data.data_ptr())
#        x = self.CG(y, FO, x.shape, x.device);
#        x = x_0 + FO.adjoint(x)
#        return x
#


#
## ==================================================================================
# class Split_Forward_operator_pylops(Split_Forward_operator):
## ==================================================================================
## Pylops compatible split forward operator
#    def __init__(self, Hsub, device = "cpu"):
#        # [H^+, H^-]
#        super().__init__(Hsub)
#        self.Op = LinearOperator(aslinearoperator(Hsub), device = device, dtype = torch.float32)
#

# DOES NOT WORK YET BECAUSE MatrixLinearOperator are not subscriptable
#
## ==================================================================================
# class Tikhonov_cg_pylops(nn.Module):
## ==================================================================================
#    def __init__(self, FO, n_iter = 5, mu = 0.1):
#        super().__init__()
#        # FO = Forward Operator - Needs to be pylops compatible!!
#        #-- Pseudo-inverse to determine levels of noise.
#        # Not sure about the support of learnable mu!!! (to be tested)
#        self.FO = FO;
#        self.mu = mu;
#        self.n_iter = n_iter
#
#    def A(self):
#        print(type(self.FO.Op))
#        # self.FO.Op.H NOT WORKING FOR NOW - I believe it's a bug, but here it isa
#        #
#        #File ~/.conda/envs/spyrit-env/lib/python3.8/site-packages/pylops_gpu/LinearOperator.py:336, in LinearOperator._adjoint(self)
#        #    334 def _adjoint(self):
#        #    335     """Default implementation of _adjoint; defers to rmatvec."""
#        #--> 336     shape = (self.shape[1], self.shape[0])
#        #    337     return _CustomLinearOperator(shape, matvec=self.rmatvec,
#        #    338                                  rmatvec=self.matvec,
#        #    339                                  dtype=self.dtype, explicit=self.explicit,
#        #    340                                  device=self.device, tocpu=self.tocpu,
#        #    341                                  togpu=self.togpu)
#        #
#        #TypeError: 'MatrixLinearOperator' object is not subscriptable
#        # Potentially needs to be improved
#        return self.FO.Op*self.FO.Op.T + self.mu*Diagonal(torch.ones(self.FO.M).to(self.FO.OP.device))
#
#    def forward(self, x, x_0):
#        # x - input (b*c, M) - measurement vector
#        # x_0 - input (b*c, N) - previous estimate
#        # z - output (b*c, N)
#
#        # Conjugate gradient to solve \|Ax-b\|^2 + mu \|x - x_0\|^2
#        #y = self.FO.Forward_op(x_0)-x;
#        #x,_ = cg(self.A(), y, niter = self.n_iter) #see pylops gpu conjugate gradient
#        #x = x_0 + self.FO.adjoint(x)
#        x = NormalEquationsInversion(Op = self.FO.Op, Regs = None, data = x, \
#                epsI = self.mu, x0 = x_0, device = self.FO.Op.device, \
#                **dict(niter = self.n_iter))
#        return x
#
# DOES NOT WORK YET BECAUSE MatrixLinearOperator are not subscriptable
#
## ==================================================================================
# class Generalised_Tikhonov_cg_pylops(nn.Module):
## ==================================================================================
#    def __init__(self, FO, Sigma_prior, n_steps):
#        super().__init__()
#        # FO = Forward Operator - pylops compatible! Does not allow to
#        # optimise the matrices Sigma_prior yet
#        self.FO = FO;
#        self.Sigma_prior = LinearOperator(aslinearoperator(Sigma_prior), self.FO.OP.device, dtype = self.FO.OP.dtype)
#
#    def A(self, var):
#        return self.FO.OP*self.Sigma_prior*self.FO.OP.H + Diagonal(var.to(self.FO.OP.device));
#
#    def forward(self, x, x_0, var_noise):
#        # x - input (b*c, M) - measurement vector
#        # x_0 - input (b*c, N) - previous estimate
#        # z - output (b*c, N)
#
#        # Conjugate gradient to solve \|Ax-b\|^2_Var_noise + \|x - x_0\|^2_Sigma_prior
#        y = self.FO.Forward_op(x_0)-x;
#        x,_ = cg(self.A(var_noise), y, niter = self.n_iter)
#        x = x_0 + self.Sigma_prior(self.FO.adjoint(x)) # to check that cast works well!!!
#        return x
#
