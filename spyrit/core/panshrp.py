import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

from spyrit.core.meas import DynamicLinearSplit, Linear, LinearSplit, HadamSplit2d, DynamicLinear
from spyrit.core.noise import Poisson
from spyrit.core.prep import Unsplit
import spyrit.core.torch as spytorch

from spyrit.misc.dual_arm import Calibration
from spyrit.misc.metrics import CC, SAM, RMSE

# from panshrp.utilities import prox_l2, prox_indic, prox_h1_2d, prox_h1_3d, prox_h1_2d_w, prox_h1_lr   # I put the functions below in the file, it will be moved in the future.
# from panshrp.utilities import derivative_spat, derivative_spec
# from panshrp.utilities import gkern, convolution, adjust_kernel

from spyrit.external.vca import vca

# from panshrp.visualize import hs_to_rgb, tensor2img
from spyrit.misc.disp import tensor2img, hs_to_rgb

import warnings



# ==================== PROXIMAL OPERATORS AS FUNCTIONS ====================
# TODO: move them somewhere else in the future

def prox_l2(u, tau, eta_tikho, mean):
    return (u + tau * eta_tikho * mean) / (1 + tau * eta_tikho)

def prox_indic(u):
    return u.clamp(0, 1)


def prox_h1_2d(u, tau, eta_3):
    height, width, n_wav = u.shape

    xi_x = torch.fft.fftfreq(width, dtype=u.dtype, device=u.device) * torch.pi * 2  # include 2pi of fourier transform inside the frequency component
    xi_y = torch.fft.fftfreq(height, dtype=u.dtype, device=u.device) * torch.pi * 2
    XI_X, XI_Y = torch.meshgrid(xi_x, xi_y, indexing="xy")
    XI_2 = XI_X**2 + XI_Y**2
    XI_2 = XI_2.reshape(height, width, 1) 

    u_complex = u.to(torch.complex64)  
    u_hat = torch.fft.fft2(u_complex, dim=(0, 1))

    denom = 1 + tau * eta_3 * XI_2
    f = torch.fft.ifft2(u_hat / denom, dim=(0, 1)).real

    return f

def prox_h1_2d_w(f: torch.Tensor, w: torch.Tensor, tau: float, w_infty: float,
                 w_c: float = None, u0: torch.Tensor = None,
                 tol: float = 1e-6, max_iter: int = 200):
    r"""Prox of :math:`\frac{\eta_3}{2} \| \nabla \cdot \|_2^2` in the metric
    :math:`W^{-1}`, i.e. the solution of

    .. math:: (I + \tau \, \mathrm{diag}(w) \, A) \, u = f, \qquad A = -\Delta

    with `tau` already carrying eta_3 and A the spectral Laplacian (periodic
    boundary conditions). The system is diagonal in no single basis (diag(w) is
    diagonal in space, A in Fourier), so it is solved by a frozen-coefficient
    splitting M = P - N with

        P = I + tau * w_c * A          (diagonal in Fourier, one FFT pair to invert)
        N = tau * diag(w_c - w) * A

    iterated as :math:`u_{k+1} = P^{-1}(f + N u_k)`. The iteration contracts at a
    rate well approximated by
    :math:`\min(\tau w_c \xi^2_{\max} / (1 + \tau w_c \xi^2_{\max}), \ 1 - w_{\min}/w_c)`
    with :math:`\xi^2_{\max} = 2\pi^2`, so it is the regularization strength
    `tau` that drives the cost, not the contrast of the weights.

    Convergence requires w_c > max(w) / 2. The default w_c = w_infty = max(w)
    keeps the splitting regular (w_c - w >= 0). Smaller admissible values lower
    the asymptotic rate but were not measured to be faster in practice, since
    the iteration stops well before the asymptotic regime: keep the default
    unless you have profiled the alternative on your own data.

    Note: the prox is inexact, which formally breaks the FISTA guarantee unless
    the errors are summable. `tol` is kept well below the outer stopping
    threshold so the prox error stays negligible; tightening it much further
    would require float64, as the FFTs run in complex64.

    Args:
        f (torch.Tensor): right-hand side, of shape (height, width, n_wav).
        w (torch.Tensor): positive weights, of shape (height, width, n_wav).
        tau (float): step size already multiplied by eta_3.
        w_infty (float): upper bound max(w).
        w_c (float): frozen coefficient of the preconditioner. Defaults to w_infty.
        u0 (torch.Tensor): warm start, typically the previous outer iterate.
        tol (float): relative increment at which the inner iteration stops.
        max_iter (int): cap on the number of inner iterations.
    """
    height, width, n_wav = f.shape

    xi_x = torch.fft.fftfreq(width, dtype=f.dtype, device=f.device) * torch.pi * 2  # include 2pi of fourier transform inside the frequency component
    xi_y = torch.fft.fftfreq(height, dtype=f.dtype, device=f.device) * torch.pi * 2
    XI_X, XI_Y = torch.meshgrid(xi_x, xi_y, indexing="xy")
    XI_2 = XI_X**2 + XI_Y**2
    XI_2 = XI_2.reshape(height, width, 1)

    if w_c is None:
        w_c = w_infty

    f_complex = f.to(torch.complex64)
    defect = w_c - w
    denom = 1 + tau * w_c * XI_2

    u = f_complex.clone() if u0 is None else u0.to(torch.complex64)

    for _ in range(max_iter):
        inner = defect * torch.fft.ifft2(XI_2 * torch.fft.fft2(u, dim=(0, 1)), dim=(0, 1))
        u_new = torch.fft.ifft2(torch.fft.fft2(f_complex + tau * inner, dim=(0, 1)) / denom, dim=(0, 1))

        incr = torch.linalg.norm(u_new - u) / torch.linalg.norm(u_new).clamp_min(1e-12)
        u = u_new

        if incr < tol:
            break
    else:
        warnings.warn(f"prox_h1_2d_w did not reach tol={tol} in {max_iter} iterations "
                      f"(relative increment = {incr:.2e}). Consider lowering tau * eta_3.")

    return u.real


def prox_h1_lr(f: torch.Tensor, U: torch.Tensor, tau: float, u_infty: float = 1.0,
               u0: torch.Tensor = None, tol: float = 1e-6, max_iter: int = 200):
    r"""Prox of :math:`\frac{\eta_3}{2} \sum_m \| \nabla b_m \|_2^2` in the metric V, for the
    coefficient maps of a low-rank model. This is the counterpart of `prox_h1_2d_w` on the
    abundance maps, and it solves

    .. math:: (I + \tau \, U \, L) \, v = f, \qquad L = -\Delta

    with `tau` already carrying eta_3, L the spectral Laplacian acting map by map (periodic
    boundary conditions) and U = V^-1 the inverse metric, block diagonal with one rank x rank
    block per pixel. Multiplying through by V gives the symmetric form (V + tau L) v = V f.

    The operator mixes two bases just like in the image domain -- L is diagonal in Fourier,
    U is diagonal in space -- so it is solved by the same frozen-coefficient splitting:

        M = I + tau * u_infty * L      (diagonal in Fourier, one FFT pair: this is prox_h1_2d)
        N = M - A = tau * (u_infty * I - U) * L

    iterated as :math:`v_{k+1} = M^{-1} (f + N v_k)`.

    u_infty = 1 is both admissible and optimal when U comes from `compute_lr_metric`, which
    normalises it so that lambda_max(U) = 1: the identity then dominates every block in the
    Loewner order, and since the rate bound grows with u_infty, no larger value helps. Do NOT
    derive u_infty from, say, the infinity norm of each block: for a symmetric matrix that
    overestimates lambda_max and only slows the iteration down.

    Note that the defect N carries L, so it vanishes at zero frequency and the rate stays
    bounded by tau*||L|| / (1 + tau*||L||) with ||L|| = 2*pi^2, independently of how badly
    conditioned the metric is (measured: 0.43 at kappa(V) = 1e1, 0.49 at 1e6). Splitting the
    symmetric form instead, with M = V, would put the metric alone in the defect and make the
    rate degrade as 1 - 1/kappa(V), or diverge outright once the frozen block no longer
    dominates.

    Args:
        f (torch.Tensor): right-hand side, of shape (height, width, rank).
        U (torch.Tensor): inverse metric, of shape (height * width, rank, rank), with the
            pixels in the same row-major order as `f`.
        tau (float): step size already multiplied by eta_3.
        u_infty (float): frozen coefficient, an upper bound on lambda_max(U).
        u0 (torch.Tensor): warm start, typically the previous outer iterate.
        tol (float): relative increment at which the iteration stops.
        max_iter (int): cap on the number of iterations.
    """
    height, width, rank = f.shape

    xi_x = torch.fft.fftfreq(width, dtype=f.dtype, device=f.device) * torch.pi * 2  # include 2pi of fourier transform inside the frequency component
    xi_y = torch.fft.fftfreq(height, dtype=f.dtype, device=f.device) * torch.pi * 2
    XI_X, XI_Y = torch.meshgrid(xi_x, xi_y, indexing="xy")
    XI_2 = XI_X**2 + XI_Y**2
    XI_2 = XI_2.reshape(height, width, 1)

    denom = 1 + tau * u_infty * XI_2  # M in Fourier

    v = f.clone() if u0 is None else u0.to(dtype=f.dtype, device=f.device)

    for _ in range(max_iter):
        # L v, map by map
        Lv = torch.fft.ifft2(XI_2 * torch.fft.fft2(v.to(torch.complex64), dim=(0, 1)), dim=(0, 1)).real

        # N v / tau = (u_infty * I - U) (L v), one rank x rank product per pixel
        Lv_flat = Lv.reshape(-1, rank)
        Nv = u_infty * Lv_flat - torch.bmm(U, Lv_flat.unsqueeze(2)).squeeze(2)

        rhs = (f + tau * Nv.reshape(height, width, rank)).to(torch.complex64)
        v_new = torch.fft.ifft2(torch.fft.fft2(rhs, dim=(0, 1)) / denom, dim=(0, 1)).real

        incr = torch.linalg.norm(v_new - v) / torch.linalg.norm(v_new).clamp_min(1e-12)
        v = v_new

        if incr < tol:
            break
    else:
        warnings.warn(f"prox_h1_lr did not reach tol={tol} in {max_iter} iterations "
                      f"(relative increment = {incr:.2e}). Consider lowering tau * eta_3.")

    return v


def prox_h1_3d(u, tau, eta_3, eta_4):
    # prox de \tau * (eta_3 * || \nabla_{x, y} \cdot ||_2^2 + eta_4 * || \partial_{\lambda} \cdot ||_2^2)
    height, width, n_wav = u.shape

    xi_x = torch.fft.fftfreq(width, dtype=u.dtype, device=u.device) * torch.pi * 2
    xi_y = torch.fft.fftfreq(height, dtype=u.dtype, device=u.device) * torch.pi * 2
    xi_wav = torch.fft.fftfreq(n_wav, dtype=u.dtype, device=u.device) * torch.pi * 2
    
    XI_X, XI_Y, XI_WAV = torch.meshgrid(xi_x, xi_y, xi_wav, indexing="xy")
    
    XI_2_XY = XI_X ** 2 + XI_Y ** 2 
    XI_2_WAV = XI_WAV ** 2
    XI_2_XY, XI_2_WAV = XI_2_XY.reshape(height, width, n_wav), XI_2_WAV.reshape(height, width, n_wav) 

    u_complex = u.to(torch.complex64)  
    u_hat = torch.fft.fftn(u_complex, dim=(0, 1, 2))
    
    num = u_hat
    denom = 1 + tau * (eta_3 * XI_2_XY + eta_4 * XI_2_WAV)
    f = torch.fft.ifftn(num / denom, dim=(0, 1, 2)).real

    return f

## derivative operator for losses computations
def derivative_spat(img, p = 1):
    b,  n_wav, height, width = img.shape

    img_reshaped = img.squeeze(0).moveaxis(0, -1)
    xi_x = torch.fft.fftfreq(width, dtype=img.dtype, device=img.device) * torch.pi * 2
    xi_y = torch.fft.fftfreq(height, dtype=img.dtype, device=img.device) * torch.pi * 2
    XI_X, XI_Y = torch.meshgrid(xi_x, xi_y, indexing="xy")

    img_complex = img_reshaped.to(torch.complex64)  
    img_hat = torch.fft.fft2(img_complex, dim=(0, 1))

    deriv_x = torch.fft.ifft2((1j * XI_X.unsqueeze(-1)) ** p * img_hat, dim=(0, 1)).real
    deriv_y = torch.fft.ifft2((1j * XI_Y.unsqueeze(-1)) ** p * img_hat, dim=(0, 1)).real

    deriv_x = deriv_x.unsqueeze(0).moveaxis(-1, 1)
    deriv_y = deriv_y.unsqueeze(0).moveaxis(-1, 1)

    deriv_xy = torch.stack((deriv_x, deriv_y))

    return deriv_xy

def derivative_spec(img, p = 1):
    b,  n_wav, height, width = img.shape

    img_reshaped = img.squeeze(0).moveaxis(0, -1)
    xi_wav = torch.fft.fftfreq(n_wav, dtype=img.dtype, device=img_reshaped.device) * torch.pi * 2

    img_complex = img_reshaped.to(torch.complex64)  
    img_hat = torch.fft.fft(img_complex, dim=2)

    deriv_spec = torch.fft.ifft((1j * xi_wav.unsqueeze(0).unsqueeze(0)) ** p * img_hat, dim=2).real

    deriv_spec = deriv_spec.unsqueeze(0).moveaxis(-1, 1)

    return deriv_spec


# ==================== END OF PROXIMAL OPERATORS ====================



# ==================== UTILITIES 

def gkern(l: int = 5, sig: float = 1., dtype=torch.float32):
    ax = torch.linspace(-(l - 1) / 2., (l - 1) / 2., l, dtype=dtype)
    gauss = torch.exp(-0.5 * torch.square(ax) / sig ** 2)
    kernel = torch.outer(gauss, gauss)
    return kernel / torch.sum(kernel)


def convolution(img: torch.tensor, gamma: torch.tensor):
    r"""
    Performs the convolution of img with kernel gamma
    :param img: shape (height, width, n_wav)
    :return: convolved img of shape (N_hat, n_wav)
    """

    height, width, n_wav = img.shape
    N_hat = height * width

    img = img.moveaxis(-1, 0).unsqueeze(0)  # shape (1, n_wav, height, width)

    gamma = gamma.unsqueeze(0).unsqueeze(0)
    kernel = gamma.repeat(n_wav, 1, 1, 1)

    pad_h = kernel.shape[-2] // 2
    pad_w = kernel.shape[-1] // 2
    img_pad = nn.functional.pad(img, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    res = nn.functional.conv2d(img_pad, kernel, padding=0, groups=n_wav)
    # res = nn.functional.conv2d(img, kernel, padding='same', groups=n_wav)

    res = torch.moveaxis(res, 1, -1)
    res = res.squeeze()
    res = res.reshape((N_hat, -1))

    return res

def adjust_kernel(kernel: torch.tensor):
    ks_h, ks_w = kernel.shape
    ks_h_adj, ks_w_adj = ks_h, ks_w

    if ks_h % 2 == 0:
        ks_h_adj += 1

    if ks_w % 2 == 0:
        ks_w_adj += 1

    gamma = torch.zeros((ks_h_adj, ks_w_adj), dtype=kernel.dtype, device=kernel.device)
    pad_h, pad_w = ks_h_adj - ks_h, ks_w_adj - ks_w
    gamma[pad_h:, pad_w:] = kernel.clone().detach()

    gamma_adj = torch.flip(gamma, dims=[0, 1])

    return gamma, gamma_adj





# =================== PANSHARPENING PGD CLASS ====================


class Pan_PGD(nn.Module):
    r""" Class for pansharpening in the image domain.
    """

    def __init__(self, meas_op, homography: torch.Tensor = torch.eye(3), R: torch.Tensor = None, spectrum = None, verbose: bool = False):
        super().__init__()

        self.dtype = homography.dtype
        self.device = homography.device

        self.meas_op = meas_op
        self.prep_op = Unsplit() if isinstance(meas_op, LinearSplit) or isinstance(meas_op, DynamicLinearSplit) else torch.nn.Identity()  # manage rescaling in place as there a places I don't 

        self.homography = homography.to(dtype=self.dtype, device=self.device)
        self.homography_inv = torch.linalg.inv(self.homography)

        self.R = R
        
        self.spectrum = spectrum

        self.verbose = verbose


        if isinstance(meas_op, DynamicLinearSplit):
            self.fwd = meas_op.measure_H_dyn
            self.adj = meas_op.adjoint_H_dyn
        elif isinstance(meas_op, DynamicLinear):
            self.fwd = meas_op.measure_H_dyn
            self.adj = meas_op.adjoint
        elif isinstance(meas_op, LinearSplit):
            self.fwd = meas_op.measure_H
            self.adj = meas_op.adjoint_H
        elif isinstance(meas_op, Linear):
            self.fwd = meas_op.measure
            self.adj = meas_op.adjoint
        elif isinstance(meas_op, torch.nn.Identity):
            self.fwd = torch.nn.Identity()
            self.adj = torch.nn.Identity()
        

    def _preprocess(self, HS: torch.Tensor):
        HS = self.prep_op(HS)
       
        if isinstance(self.meas_op.noise_model, Poisson):
            alpha = self.meas_op.noise_model.alpha
            HS = HS / alpha

        return HS
    

    def _init_shapes(self, HS, P):
        # shape variables
        if isinstance(self.meas_op, torch.nn.Identity):
            self.bs, self.n_wav, self.n, self.m = HS.shape
            self.M = self.n * self.m
        else:
            self.bs, self.n_wav, self.M = HS.shape
            self.n, self.m = self.meas_op.meas_shape

        if isinstance(self.meas_op, DynamicLinear):
            self.l, _ = self.meas_op.img_shape
        else:
            self.l = self.n
        self.L = self.l ** 2
        
        _, _, self.height, self.width = P.shape

        self.N = self.n * self.m
        self.N_hat = self.height * self.width

        self.im_shape = self.height, self.width, self.n_wav
        self.mat_shape = self.N_hat, self.n_wav

        if self.n_wav != 3 and self.spectrum is None:
            warnings.warn("No spectrum specified with hyperspectral data, \
                            be extremely careful as visualization errors might occur")    
            
    def display_image(self, X: torch.Tensor, X_im_shape: tuple, k: int = 0, step_show: int = None):
        if step_show is not None and k % step_show == 0:
            clear_output(wait=True)
            if self.n_wav != 3:
                plt.imshow(tensor2img(hs_to_rgb(X.reshape(X_im_shape).unsqueeze(0).moveaxis(-1, 1), self.spectrum)))
            else:
                plt.imshow(tensor2img(X.reshape(X_im_shape).unsqueeze(0).moveaxis(-1, 1)))

            plt.title('Itération %d' % k)
            plt.pause(0.1)

    def build_psf(self):
        ks_h, ks_w = int(self.height / self.l), int(self.width / self.l)
        up_factor = ks_h * ks_w

        gamma = torch.ones((ks_h, ks_w), dtype=self.dtype, device=self.device)
        gamma = gamma / up_factor
        gamma, gamma_adj = adjust_kernel(gamma)

        if self.verbose:
            print("kernel size = (", ks_h, ", ", ks_w, ")")

        return gamma, gamma_adj           
    

    def build_cov_mat(self, HS):
        """
        Build the covariance matrix of the noise for the data fidelity term of the optimization problem from data

        Args:
            HS (torch.Tensor): Hyperspectral measurement of shape (..., 2 M)

        """
        if isinstance(self.meas_op, torch.nn.Identity):
            raise ValueError("Can't use variability on noise without a measurement operator.")
        
        elif isinstance(self.meas_op, LinearSplit) or isinstance(self.meas_op, DynamicLinearSplit):
            if isinstance(self.meas_op.noise_model, torch.nn.Identity):
                Sigma_alpha = None
                warnings.warn("No noise model detected on the measurement operator, switching to var_noise = False.")
            elif isinstance(self.meas_op.noise_model, Poisson):
                Sigma_alpha = (HS[..., ::2] + HS[..., 1::2]).squeeze(0) / self.meas_op.noise_model.alpha  # error in v2, it should not be squared

                # var_img is legacy argument
                var_img = False
                if var_img: 
                    Sigma = self.meas_op.order.ravel().unsqueeze(0) # unscaled

                    I_lambdas = Sigma_alpha.mean(axis=1)
                    Sigma = Sigma.repeat(I_lambdas.shape[0], 1) * I_lambdas.unsqueeze(1) #/ I_lambdas.sum()  # scaled for every lambda
                else:
                    Sigma = None
            else:
                raise ValueError(f"Variability for the {type(self.meas_op.noise_model)} noise model has not yet been implemented.")
            
        else:
            raise ValueError(f"Can't extract covariance matrix from {type(self.meas_op)} measurement operator.")

        return Sigma_alpha
  
    def estim_relative_response(self, M_HS: torch.Tensor, P: torch.Tensor, calib_op: Calibration, gamma: torch.Tensor):
        """
        Estimate the relative response of the CMOS camera with respect to the hyperspectral camera.

        Args:
            M_HS (torch.tensor): Low-resolution hyperspectral measurement from SPC of size K \times \Lambda.

        Returns:
            r_estim (torch.tensor): Estimated relative response.
        """

        p = P.flatten()
        
        # M of shape (K, n_wav), p of shape N_hat
        p_down = calib_op.vfast_prod_G_vec(convolution(p.reshape((self.height, self.width, 1)), gamma))
        # if isinstance(self.meas_op, HadamSplit2d):
        p_down = self.fwd(p_down.moveaxis(-1, 0).reshape((1, 1, self.l, self.l)))
        p_down = p_down.squeeze()

        # Create finite difference matrix D and compute D^T D and add identity for Sobolev's H1 regularization
        D = torch.diag(torch.ones(self.n_wav - 1, dtype=self.dtype, device=self.device), 1) - torch.diag(torch.ones(self.n_wav, dtype=self.dtype, device=self.device))
        D[-1, -1] = 0  # remove last element (neumann boundary conditions)
        DtD = torch.eye(self.n_wav, dtype=self.dtype, device=self.device) + D.T @ D

        eta = 1.2e3   # approximately determined for CIE spectrum with grid search on different alphas
        if isinstance(self.meas_op, nn.Identity):
            eta /= self.N

        MtM = M_HS.T @ M_HS
        MtM_reg = MtM + eta * DtD
        r_estim = torch.linalg.solve(MtM_reg, M_HS.T @ p_down)

        return r_estim.unsqueeze(1)
    
    def compute_metric(self, weighted: bool = False, X_interp: torch.Tensor = None, eta_w_3: float = 100, eta_w_4: float = 10,
                       eps_w: float = 1e-5):
        r"""Build the weights W of the variable-metric PGD from a smoothed prior on the colors.

        `eps_w` is the floor of W, hence its condition number (kappa(W) ~ 1 / eps_w). It plays
        opposite roles in the two weighted regularizations:

        - 'h1': the metric cancels in the optimality condition, so W does not change the
          minimizer. The effect comes entirely from the ill-conditioning slowing down the
          coordinates where W is small, and is therefore an early-stopping effect: keep
          eps_w small (1e-5) and fix n_iter explicitly.
        - 'h1-adv-diff': W enters the regularizer itself, so it does change the minimizer and
          no ill-conditioning is needed. A small eps_w only makes b = grad(W) / W blow up and
          the explicit advection step unstable: use eps_w ~ 1e-2 there.
        """
        if weighted:
            epsilon = eps_w
            X_priori = prox_h1_3d(X_interp.reshape(self.im_shape), tau=1, eta_3=eta_w_3, eta_4=eta_w_4)
            X_priori = X_priori.moveaxis(-1, 0).unsqueeze(0)
            # X_priori = X_priori / self.R.T ?
            X_priori = prox_indic(X_priori)

            X_priori_across_lambda, _ = torch.max(X_priori, dim=1, keepdim=True)
            # X_priori_across_lambda = torch.sum(X_priori, dim=1, keepdim=True)

            W = (X_priori + epsilon) / (X_priori_across_lambda + epsilon)
            return W.squeeze().moveaxis(0, -1).reshape((self.N_hat, self.n_wav))
        else:
            return None

    def set_stepsize_pgd(self, tau, eta_1, eta_2, Sigma_alpha, R):
        if tau is None:
            norm1 = 1.  #  1 = max sing val de G (sans chgt pov). prendre en compte gamma aussi (c'est fait car \|GAMMA\|_2 < sum(kernel) = 1)
            if isinstance(self.meas_op, HadamSplit2d):
                norm1 *= self.meas_op.N
            elif isinstance(self.meas_op, DynamicLinear):
                norm1 *= self.max_eig_Hdyn ** 2
            norm2 = torch.linalg.norm(R @ R.T, dtype=self.dtype, ord=2)

            eps = 1e-6  # avoid divide by 0
            if Sigma_alpha is not None: 
                Sigma_alpha_pinv = torch.where(torch.abs(Sigma_alpha) > eps, 1 / Sigma_alpha, torch.zeros_like(Sigma_alpha))
                norm1 *= Sigma_alpha_pinv.max()

            beta = eta_1 * norm1 + eta_2 * norm2
            
            tau = float(1 / beta) - 1e-5

            if self.verbose:
                print(f"beta = {eta_1} * {norm1} + {eta_2} * {norm2} = {beta}")
                print(f"tau = {tau}")

        return tau


    def pan_pgd(self, exp_input: dict, X0: torch.Tensor, type_reg: str = 'h1', eta_1: float = 1., eta_2: float = 1., eta_3: float = 0., eta_4: float = 0., 
                n_iter: int = 250, tau: float = 1e-3, step_show: int = None, compute_loss: bool = False, X_ref: torch.tensor = None):
        """
        Perform pansharpening optimization process.

        Args:
            M_HS (torch.tensor): Low-resolution hyperspectral image.
            P (torch.tensor): High-resolution panchromatic image.
            R (torch.tensor): Relative spectral response between the two cameras.
            type_reg (str): Type of regularization ('h1', 'h1-3d', 'semi_sobolev', etc.).
            weighted (str): Use weights from an ad hoc a priori spatial distribution of colors ('interp', 'spc_reg').
            spc_spec_prior (bool): Whether to use the spectral prior from the hyperspectral camera.
            eta_1, eta_2, eta_3, eta_4 (float): Regularization parameters.
            n_iter (int): Number of iterations.
            tau (float): Step size for the gradient descent.
            order (int): Order of the interpolation in the calibration operator.
            var_noise (bool): Whether to account for noise variability.
            step_show (int): Step interval for displaying optimization progress.

        Returns:
            X (torch.tensor): Current estimate of the high-resolution image.
            metadata (dict): Dictionary containing the computed losses and metrics.
        """

        HS = exp_input['HS']
        P = exp_input['P']
        R = exp_input['R']
        calib_op = exp_input['calib_op']
        gamma = exp_input['psf']
        gamma_adj = exp_input['psf_adj']
        Sigma_alpha = exp_input['noise_variance']
        W = exp_input['W']

        losses = {'loss_1': np.zeros(n_iter), 'loss_2': np.zeros(n_iter),
                  'loss_3': np.zeros(n_iter), 'loss_4': np.zeros(n_iter)}
        
        metrics = {'CC': np.zeros(n_iter), 'SAM': np.zeros(n_iter), 'RMSE': np.zeros(n_iter)}
        
        X = X0.clone().detach()

        var_noise = Sigma_alpha is not None
        weighted = W is not None

        if var_noise:
            epsilon = 1e-6
            Sigma_alpha_pinv = torch.where(torch.abs(Sigma_alpha) > epsilon, 1 / Sigma_alpha, torch.zeros_like(Sigma_alpha))

        with torch.no_grad():
            if weighted:
                # advection field b = grad(log W), used by 'h1-adv-diff'
                W_img = W.reshape(self.im_shape).unsqueeze(0).moveaxis(-1, 1)  # (1, n_wav, height, width)
                d_log_W = derivative_spat(W_img) / W_img  # probably more stable than taking the log

                # The advection step is explicit while the diffusion is implicit, so the
                # per-mode amplification is |1 - i tau eta_3 (b.xi)| / (1 + tau eta_3 |xi|^2).
                # It stays below 1 (worst case xi -> 0) only while tau * eta_3 * |b|^2 <= 2.
                # b is grad(W) / W, so it blows up wherever W is close to its floor epsilon:
                # clip it to the stability bound rather than let the scheme diverge.
                if type_reg == 'h1-adv-diff' and tau * eta_3 > 0:
                    b_max = float((2 / (tau * eta_3)) ** 0.5)
                    b_norm = (d_log_W[0] ** 2 + d_log_W[1] ** 2).sqrt()
                    n_clipped = int((b_norm > b_max).sum())
                    if n_clipped > 0:
                        d_log_W = d_log_W * (b_max / b_norm.clamp_min(b_max))
                        warnings.warn(
                            f"h1-adv-diff: |grad log W| exceeds the stability bound {b_max:.2f} on "
                            f"{100 * n_clipped / b_norm.numel():.1f}% of the coordinates (max was "
                            f"{b_norm.max():.1f}); the advection field was clipped. The bound comes "
                            f"from a frozen-coefficient analysis and ignores the variation of b, so "
                            f"clipping alone does NOT guarantee stability -- at eps_w = 1e-5 the "
                            f"scheme still diverges. Raise eps_w (1e-2 works) rather than rely on it.")

                # used for iterative h1 computation
                W_im = W.reshape(self.im_shape)
                W_infty = W.max().item()
                prox_warm_start = None  # previous output of prox_h1_2d_w, reused as a warm start


            for k in range(n_iter):
                X_old = X.clone().detach()            
                if k > 0: # skip first iteration to get initial loss
                    if isinstance(self.meas_op, torch.nn.Identity): 
                        down = calib_op.vfast_prod_G_vec(convolution(X.reshape(self.im_shape), gamma))
                        up = convolution(calib_op.vfast_prod_Gt_vec(down - HS).reshape(self.im_shape), gamma_adj)
                    else:
                        down = calib_op.vfast_prod_G_vec(convolution(X.reshape(self.im_shape), gamma))
                        down_meas = self.fwd(down.moveaxis(-1, 0).reshape((self.bs, self.n_wav, self.l, self.l)))
                        # down_meas = self.prep_op(down_meas).squeeze()
                        down_meas = down_meas.squeeze()
                        residu = down_meas - HS.moveaxis(-1, 0)
                        if var_noise:
                            # residu /= Sigma_alpha
                            residu *= Sigma_alpha_pinv
                        down_img = self.adj(residu).moveaxis(0, -1)
                        up = convolution(calib_op.vfast_prod_Gt_vec(down_img).reshape(self.im_shape), gamma_adj)
                    
                    if weighted:
                        X = X - tau * W * (eta_1 * up + eta_2 * (X @ R - P) @ R.T)  # change of metric

                    else:
                        X = X - tau * (eta_1 * up + eta_2 * (X @ R - P) @ R.T)

                    if type_reg == 'h1':
                        if weighted:
                            # todo: think about how not to define frequency grid at each call...
                            prox_warm_start = prox_h1_2d_w(X.reshape(self.im_shape), W_im, tau * eta_3, W_infty, u0=prox_warm_start)
                            X = prox_warm_start.reshape(self.mat_shape)
                        else:
                            X = prox_h1_2d(X.reshape(self.im_shape), tau, eta_3).reshape(self.mat_shape)

                    elif type_reg == 'h1-3d':
                        X = prox_h1_3d(X.reshape(self.im_shape), tau, eta_3, eta_4).reshape(self.mat_shape)

                    elif type_reg == 'indic':
                        X = prox_indic(X)

                    # add inertia (FISTA, Chambolle-Dossal with a=3)
                    if k > 0:
                        alpha_k = (k - 1) / (k + 3)
                        X = X + alpha_k * (X - X_old)

                self.display_image(X, self.im_shape, k, step_show)

                if compute_loss:
                    # Data fidelity term
                    l1 = calib_op.vfast_prod_G_vec(convolution(X.reshape(self.im_shape), gamma))
                    l1 = self.fwd(l1.moveaxis(-1, 0).reshape((self.bs, self.n_wav, self.l, self.l)))
                    l1 = l1.squeeze(0).moveaxis(0, -1)

                    if var_noise:
                        l1 = (l1 - HS) * Sigma_alpha_pinv.moveaxis(0, -1) ** 0.5
                    else:
                        l1 = l1 - HS

                    l1 = torch.linalg.norm(l1).cpu() ** 2
                    losses["loss_1"][k] = 0.5 * eta_1 * l1

                    # Panchromatic fidelity term
                    l2 = torch.linalg.norm(X @ R - P).cpu() ** 2
                    losses["loss_2"][k] = 0.5 * eta_2 * l2

                    l = 0.5 * (eta_1 * l1 + eta_2 * l2)

                    # Regularization terms
                    X_torch = X.reshape(self.im_shape).unsqueeze(0).moveaxis(-1, 1)

                    if "h1" in type_reg:
                        d_spat = derivative_spat(X_torch)

                        l3 = torch.linalg.norm(d_spat.flatten(), ord=2).cpu() ** 2
                        losses["loss_3"][k] = 0.5 * eta_3 * l3
                        l += 0.5 * eta_3 * l3

                        if type_reg == "h1-3d":
                            d_spec = derivative_spec(X_torch)

                            l4 = torch.linalg.norm(d_spec.flatten(), ord=2).cpu() ** 2
                            losses["loss_4"][k] = 0.5 * eta_4 * l4
                            l += 0.5 * eta_4 * l4
                        
                # compute metrics during iterations
                if X_ref is not None:
                    metrics['CC'][k] = CC(X.reshape(self.im_shape), X_ref)
                    metrics['SAM'][k] = SAM(X.reshape(self.im_shape), X_ref)
                    metrics['RMSE'][k] = RMSE(X.reshape(self.im_shape), X_ref)

                if torch.linalg.norm(X - X_old) / torch.linalg.norm(X_old) < 1e-5 and k > 0:
                    print("Convergence reached at iteration %d" % k)
                    break

            plt.show()

            metadata = {'loss': losses, 'metrics': metrics}

            X_torch = X.reshape(self.im_shape).unsqueeze(0).moveaxis(-1, 1)

            if self.verbose and weighted:
                print('weights range in [%.2f, %.2f]' % (W.min(), W.max()))

            if k == n_iter - 1:
                print("Maximum number of iterations reached, convergence not guaranteed. Residual norm = %.5f" % torch.linalg.norm(X - X_old))

        return X_torch, metadata


    def forward(self, HS, P, type_reg: str = 'h1', eta_3: int = 1e-1, eta_4: int = 0, order: int = 1, var_noise: bool = False, weighted = False, 
                tau: float = None, n_iter: int = 500, step_show: int = None, compute_loss: bool = False, X_ref: torch.tensor = None,
                eta_w_3: float = 10, eta_w_4: float = 0, eps_w: float = 1e-5):
        """
        Proximal gradient descent for pansharpening. We minimize _[Maitre]:

        .. math::
            \arg \min_{f \in mathbb{R}^{\hat{N} \Lambda} \frac{\eta_1}{2} \| m - A_1 f \|_{\Gamma^{-1}}^2 + \frac{\eta_2}{2} \| p - A_2 f \|_2^2 + r_{\eta_3, \eta_4}(f)

        where 
        :math:`m \in \mathbb{R}^{M \Lambda}` are the hyperspectral measurements whose shape is (bs, n_wav, n, m) for an Identity measurement operator and (bs, n_wav, M) otherwise,  
        :math:`A_1 \in \mathbb{R}^{M \Lambda \times \hat{N} \Lambda}` is the hyperspectral forward operator,
        :math:`\Gamma` is the covariance matrix of the noise in the hyperspectral measurement,
        :math:`p \in \mathbb{R}^{\hat{N}}` is the high-resolution panchromatic image,
        :math:`A_2 \in \mathbb{R}^{\hat{N} \times \hat{N} \Lambda}` is the panchromatic forward operator, and
        :math:`r_{\eta_3, \eta_4}(f)` is a regularization term that can be chosen based on the desired properties of the solution.

        The two data fidelity terms are normalized by the spectral norm of the corresponding forward operator, i.e. :math:`\eta_1 \approx 1 / \| A_1 \|^2_2` and :math:`\eta_2 = 1 / \| A_2 \|^2_2`.

        .. warning::
            TODO: Improve the documentation
        
        Args: 
            HS (torch.tensor): Low-resolution hyperspectral measurement, of shape .
            P (torch.tensor): High-resolution panchromatic image.
            type_reg (str): Type of regularization ('h1', 'h1-3d', 'semi_sobolev', etc.).
            weighted (str): Use weights from an ad hoc a priori spatial distribution of colors ('interp', 'spc_reg').
            spc_spec_prior (bool): Whether to use the spectral prior from the hyperspectral camera.
            eta_1, eta_2, eta_3, eta_4 (float): Regularization parameters.
            n_iter (int): Number of iterations.
            tau (float): Step size for the gradient descent.
            order (int): Order of the interpolation in the calibration operator.
            var_noise (bool): Whether to account for noise variability.
            step_show (int): Step interval for displaying optimization progress.
        """

        # Compute cov matrix on unsplit data
        Sigma_alpha = self.build_cov_mat(HS) if var_noise else None

        HS = self._preprocess(HS)
        self._init_shapes(HS, P)

        P = P.squeeze().reshape((self.N_hat, 1))
        HS = HS.squeeze().moveaxis(0, -1).reshape((self.M, self.n_wav))

        gamma, gamma_adj = self.build_psf()
        up_factor = self.N_hat // self.N
        
        calib_op = Calibration(order=order, spc_shape=(self.l, self.l), cmos_shape=(self.height, self.width), homography=self.homography)

        if self.R is None:
            r_estim = self.estim_relative_response(HS, P, calib_op, gamma)
        else:
            r_estim = self.R
        
        # Initial guess
        if isinstance(self.meas_op, torch.nn.Identity):
            X_interp = up_factor * convolution(calib_op.vfast_prod_Gt_vec(HS).reshape(self.im_shape), gamma_adj)
        else:
            if isinstance(self.meas_op, HadamSplit2d):
                f_HS = self.meas_op.fast_pinv(HS.moveaxis(-1, 0), vectorize=True).moveaxis(0, -1)
            elif isinstance(self.meas_op, DynamicLinear):
                # send to cpu for faster linalg (on a single img)
                H_dyn = self.meas_op.H_dyn.to('cpu')

                s = torch.linalg.svdvals(H_dyn)
                self.max_eig_Hdyn = s[0].item()

                Dx, Dy = spytorch.neumann_boundary((self.l, self.l))
                D2 = Dx.T @ Dx + Dy.T @ Dy

                eta = 1e-1  # doesnt need to much tuning, as it is just for the initial guess (simulé recuit)
                f_HS = torch.linalg.solve(H_dyn.T @ H_dyn + eta * self.max_eig_Hdyn ** 2 * D2, H_dyn.T @ HS.to('cpu')).reshape((self.L, self.n_wav)).to(self.device)

            X_interp = up_factor * convolution(calib_op.vfast_prod_Gt_vec(f_HS).reshape(self.im_shape), gamma_adj)

        # Prior image (used with the weighted PGD to obtain a map of the color distribution)
        W = self.compute_metric(weighted=weighted, X_interp=X_interp, eta_w_3=eta_w_3, eta_w_4=eta_w_4, eps_w=eps_w)

        self.display_image(X_interp, self.im_shape, k=0, step_show=step_show)

        if isinstance(self.meas_op, HadamSplit2d):
            eta_1 = 1 / self.N 
        elif isinstance(self.meas_op, DynamicLinear):
            eta_1 = 1 / self.max_eig_Hdyn ** 2
        else:
            eta_1 = 1
        eta_2 = 1 / float(torch.sum(r_estim ** 2))

        if var_noise:
            Sigma_alpha_pinv = torch.where(torch.abs(Sigma_alpha) > 1e-6, 1 / Sigma_alpha, torch.zeros_like(Sigma_alpha))
            eta_1 /= Sigma_alpha_pinv.max()

        tau = self.set_stepsize_pgd(tau, eta_1, eta_2, Sigma_alpha, r_estim)

        exp_input = {
            'HS': HS, 'P': P, 'R': r_estim, 'calib_op': calib_op, 'psf': gamma, 'psf_adj': gamma_adj, 'noise_variance': Sigma_alpha, 'W': W
        }

        X, metadata = self.pan_pgd(exp_input, X0=X_interp, type_reg=type_reg, eta_1=eta_1, eta_2=eta_2, eta_3=eta_3, eta_4=eta_4,
                                   n_iter=n_iter, tau=tau, step_show=step_show, compute_loss=compute_loss, X_ref=X_ref)

        metadata['exp_input'] = exp_input
        metadata['W'] = W if weighted else None
        
        return X, metadata


class Pan_PGD_LR(Pan_PGD):
    r""" 
    PanPGD class but with a low-rank assumption.  TODO: Improve the documentation.
    """

    def __init__(self, meas_op, rank: int, fact: str = 'vca', homography: torch.Tensor = torch.eye(3), R: torch.Tensor = None, spectrum = None, verbose: bool = False):
        super().__init__(
            meas_op,
            homography,
            R,
            spectrum,
            verbose
        )

        self.rank = rank
        self.fact = fact


    def _init_shapes(self, HS, P):
        # shape variables
        super()._init_shapes(HS, P)
            
        self.lr_im_shape = self.height, self.width, self.rank
        self.lr_mat_shape = self.N_hat, self.rank


    def factorize(self, X, rank, fact: str = 'vca'):
        # X of shape (n_pixels, n_wav)
        if fact == 'svd':
            U, SV, Vh = torch.linalg.svd(X, full_matrices=False)

            if rank is None:
                rcond = 0.085
                threshold = rcond * SV[0]
                rank = torch.searchsorted(-SV, -threshold, right=False).item()  # negative because function only works with tensors sorted in ascending order
            
            # Truncate SVD
            A0 = U[:, :rank] @ torch.diag(SV[:rank]) # materials maps
            S = Vh[:rank, :]    # spectral signatures

            # A0 = U[:, :rank] # materials maps
            # S = torch.diag(SV[:rank]) @ Vh[:rank, :]

            S_l2_norm = torch.linalg.norm(S, ord=2)
            S /= S_l2_norm
            A0 *= S_l2_norm


        elif fact == 'pca':
            # /!\ PCA might not be a good idea because I assume X = AS; not X = X_mean + AS...
            n_pix, n_wav = X.shape
            X_mean = X.mean(dim=0, keepdim=True)
            X_pca = X - X_mean
            Cov_pca = (X_pca.T @ X_pca) / (n_pix - 1)

            # U, SV, Vh = torch.linalg.svd(Cov_pca, full_matrices=False)
            L, Q = torch.linalg.eigh(Cov_pca)  # sorted in ascending order
            idx = torch.argsort(L, descending=True)[:rank]

            S = Q.T[idx, :] # spectra
            # S = Vh[:rank, :] * torch.sqrt(SV[:rank]).unsqueeze(1)

            S = S / torch.linalg.norm(S, ord=2)

            A0 = X @ torch.linalg.pinv(S)

        elif fact == 'vca':
            max_vol = 0
            vol = torch.zeros((1, 20), dtype=self.dtype, device=self.device)

            for idx_vca in range(20):
                Ae, _, _ = vca(X.T, rank, True, verbose='off')
                vol[0, idx_vca] = torch.abs(torch.det(Ae.T @ Ae))
                if vol[0, idx_vca] > max_vol:
                    S = Ae.T
                    max_vol = vol[0, idx_vca]

            # orthonormalize S to avoid scaling issues with the gradient step (which is not scale invariant)
            # U, SV, Vh = torch.linalg.svd(S, full_matrices=False)
            # S = Vh

            S /= torch.linalg.norm(S, ord=2)
            A0 = X @ torch.linalg.pinv(S)

        return A0, S


    def compute_lr_metric(self, W, S, eps=1e-6, rcond=1e-10):
        """
        Build the variable metric on the coefficient maps induced by the image-domain metric.

        The inner product on hyperspectral cubes carries over to the coefficient maps as
        <u, v>_{W^-1} = <a, b>_V with V = L^T W^-1 L and L = S^T (x) I_N_hat. Since W is
        diagonal and L is a Kronecker product, V is block diagonal: it reduces to one M x M
        Gram matrix V(x) = S diag(1 / W(x, .)) S^T per pixel. That is what makes the metric
        affordable at all -- N_hat small dense blocks instead of one N_hat * M square matrix.

        Each block is then inverted through its eigendecomposition rather than with
        torch.linalg.inv. V is symmetric positive definite, but its condition number has a
        heavy tail (up to ~1e7 on RGB data with a small eps_w), and inv() returns blocks
        asymmetric by up to ~1e-3, which stops U from being a metric at all -- the
        convergence proof needs it symmetric positive definite. eigh gives an exactly
        symmetric inverse, a genuine positivity check and the normalisation in one pass.
        Note that eigvalsh applied to the output of inv() is NOT such a check: it reads a
        single triangle, so on an asymmetric input it reports negative eigenvalues that the
        matrix does not have.

        Args:
            W (torch.Tensor): image-domain metric, of shape (N_hat, n_wav).
            S (torch.Tensor): spectral basis, of shape (M, n_wav).
            eps (float): floor added to W before inversion.
            rcond (float): relative floor on the eigenvalues of V, guarding near-singular blocks.

        Returns:
            U (torch.Tensor): inverse metric per pixel, of shape (N_hat, M, M), normalised so
                that lambda_max(U) = 1 and the step size computed without it stays valid.
            ev_U (torch.Tensor): eigenvalues of U before normalisation, ascending.
            C (torch.Tensor): the metric V itself, scaled consistently so that C = U^-1.
        """

        if W is None:
            return None, None, None

        # kappa(V) reaches ~1e7, which is at the edge of what float32 inversion survives
        inv_W = 1.0 / (W.double() + eps)
        S_d = S.double()

        # V(x) = S diag(1 / W(x, .)) S^T, one M x M block per pixel
        # (N, 1, n_wav) * (1, M, n_wav) -> (N, M, n_wav), then @ (n_wav, M) -> (N, M, M)
        C = torch.matmul(inv_W.unsqueeze(1) * S_d.unsqueeze(0), S_d.T)
        C = 0.5 * (C + C.transpose(1, 2))  # drop the round-off asymmetry of the product

        ev_C, Q = torch.linalg.eigh(C)

        floor = ev_C.amax(dim=1, keepdim=True) * rcond
        n_clamped = int((ev_C < floor).sum())
        if n_clamped > 0:
            ev_C = ev_C.clamp_min(floor)
            warnings.warn(f"compute_lr_metric: {100 * n_clamped / ev_C.numel():.2f}% of the "
                          f"eigenvalues of V fell below rcond * lambda_max and were clamped. "
                          f"The spectral basis S is nearly singular on those pixels.")

        ev_U = 1.0 / ev_C  # eigenvalues of U = V^-1 (descending, since ev_C is ascending)
        max_eig = ev_U.amax()

        # U = Q diag(ev_U / max_eig) Q^T, exactly symmetric by construction
        U = (Q * (ev_U / max_eig).unsqueeze(1)) @ Q.transpose(1, 2)
        C = C * max_eig  # keep C = U^-1

        return U.to(self.dtype), ev_U.flip(-1).to(self.dtype), C.to(self.dtype)
    

    def set_stepsize_pgd(self, tau, eta_1, eta_2, Sigma_alpha, R, S):
        if tau is None:
            norm1 = torch.linalg.norm(S, ord=2) ** 2
            if isinstance(self.meas_op, HadamSplit2d):
                norm1 *= self.meas_op.N
            elif isinstance(self.meas_op, DynamicLinear):
                norm1 *= self.max_eig_Hdyn ** 2
            norm2 = torch.linalg.norm(S @ R, ord=2) ** 2

            eps = 1e-6  # avoid divide by 0
            if Sigma_alpha is not None: 
                Sigma_alpha_pinv = torch.where(torch.abs(Sigma_alpha) > eps, 1 / Sigma_alpha, torch.zeros_like(Sigma_alpha))
                norm1 *= Sigma_alpha_pinv.max()

            beta = eta_1 * norm1 + eta_2 * norm2
            
            tau = float(1 / beta) - 1e-5  # 1 / beta rather than 2 / beta because of the FISTA acceleration (which allows for larger step sizes without diverging)

            if self.verbose:
                print(f"beta = {eta_1} * {norm1} + {eta_2} * {norm2} = {beta}")
                print(f"tau = {tau}")

        return tau


    def pan_pgd_lr(self, exp_input: dict, A0: torch.tensor, type_reg: str = 'h1', eta_1: float = 1, eta_2: float = 1, eta_3: float = 1e-1,
                    n_iter: int = 250, tau: float = 1e-3, step_show: int = None, compute_loss: bool = False, X_ref: torch.tensor = None):
        """
        Proximal gradient descent for pansharpening. We minimize 

        .. math::
            \arg \min_{f \in mathbb{R}^{\hat{N} \Lambda} \eta_1 \mathcal{D}_1(f) + \eta_2 \mathcal{D}_2(f) + \eta_3 \mathcal{R}(f)

        .. warning::
            TODO: Improve the documentation

        where 
        :math:`\mathcal{D}_1(f) = \frac{1}{2} \| m - M_1 f \|_{\Gamma^{-1}}^2`
        :math:`\mathcal{D}_2(f) = \frac{1}{2} \| p - M_2 f \|_2^2`

        """

        HS = exp_input['HS']
        P = exp_input['P']
        R = exp_input['R']
        calib_op = exp_input['calib_op']
        gamma = exp_input['psf']
        gamma_adj = exp_input['psf_adj']
        Sigma_alpha = exp_input['noise_variance']
        U = exp_input['U']
        C = exp_input['C']
        S = exp_input['spectral_basis']

        Sr = S @ R

        losses = {'loss_1': np.zeros(n_iter), 'loss_2': np.zeros(n_iter), 'loss_3': np.zeros(n_iter)}

        metrics = {'CC': np.zeros(n_iter), 'SAM': np.zeros(n_iter), 'RMSE': np.zeros(n_iter)}

        var_noise = Sigma_alpha is not None
        weighted = U is not None

        mean_0 = A0.mean()

        # we assume HS was preprocess
        with torch.no_grad():
            # Optim % A
            if weighted:
                C = C.reshape((self.height, self.width, self.rank ** 2)).unsqueeze(0).moveaxis(-1, 1)  # reshape to (r^2, h, w) to apply spatial derivative
                d_C = derivative_spat(C).squeeze(1).moveaxis(1, -1).reshape((2, self.N_hat, self.rank, self.rank))  # (2, N_hat, r, r)

                # U_tilde = U.reshape((self.height, self.width, self.rank ** 2)).unsqueeze(0).moveaxis(-1, 1)  # reshape to (r^2, h, w) to apply spatial derivative
                # d_U = derivative_spat(U_tilde).squeeze(1).moveaxis(1, -1).reshape((2, self.N_hat, self.rank, self.rank))  # (2, N_hat, r, r)

                prox_warm_start = None  # previous output of prox_h1_lr, reused as a warm start

            A = A0.clone().detach()
            for k in range(n_iter):
                A_old = A.clone().detach()            
                if k > 0: # skip first iteration to get initial loss
                    if isinstance(self.meas_op, torch.nn.Identity): 
                        down = calib_op.vfast_prod_G_vec(convolution(A.reshape(self.lr_im_shape), gamma)) @ S
                        up = convolution(calib_op.vfast_prod_Gt_vec(down - HS).reshape(self.lr_im_shape), gamma_adj) @ S.T
                    else:
                        down = calib_op.vfast_prod_G_vec(convolution(A.reshape(self.lr_im_shape), gamma))
                        down_meas = self.fwd(down.moveaxis(-1, 0).reshape((1, self.rank, self.l, self.l)))
                        # down_meas = self.prep_op(down_meas).squeeze(0)
                        down_meas = down_meas.squeeze(0)
                        down_meas = S.T @ down_meas    # down_meas @ S
                        residu = down_meas - HS.moveaxis(-1, 0)  # change shapes to be handled by spyrit with n_wav first
                        if var_noise:
                            residu /= Sigma_alpha #** 0.5
                        down_img = self.adj(residu).moveaxis(0, -1) @ S.T
                        up = convolution(calib_op.vfast_prod_Gt_vec(down_img).reshape(self.lr_im_shape), gamma_adj)


                    grad_df = eta_1 * up + eta_2 * (A @ Sr - P) @ Sr.T

                    if weighted:
                        # shapes: (N_hat, M, M) @ (N_hat, M, 1) -> (N_hat, M) : N_hat matrix vector products (M, M) @ (M, 1)
                        grad_df = torch.bmm(U, grad_df.unsqueeze(2)).squeeze(2)

                    A = A - tau * grad_df

                    if type_reg == 'h1':
                        if weighted:
                            # Prox in the new metric V, i.e. the solution of (I + tau eta_3 U L) v = A.
                            prox_warm_start = prox_h1_lr(A.reshape(self.lr_im_shape), U, tau * eta_3, u0=prox_warm_start)
                            A = prox_warm_start.reshape(self.lr_mat_shape)
                        else:
                            A = prox_h1_2d(A.reshape(self.lr_im_shape), tau, eta_3).reshape(self.lr_mat_shape)

                    elif type_reg == 'l2':
                        A = prox_l2(A.reshape(self.lr_im_shape), tau, eta_3, mean=mean_0).reshape(self.lr_mat_shape)

                    # add inertia (FISTA, Chambolle-Dossal with a=3)
                    if k > 0:
                        alpha_k = (k - 1) / (k + 3)
                        A = A + alpha_k * (A - A_old)

                if step_show is not None:
                    X = A @ S
                    self.display_image(X, self.im_shape, k, step_show)

                if compute_loss:
                    # Data fidelity term
                    l1 = calib_op.vfast_prod_G_vec(convolution(A.reshape((self.lr_im_shape)), gamma))
                    l1 = self.fwd(l1.moveaxis(-1, 0).reshape((self.bs, self.rank, self.l, self.l)))
                    l1 = l1.squeeze(0).moveaxis(0, -1)

                    if var_noise:
                        l1 = (l1 @ S - HS) / Sigma_alpha.moveaxis(0, -1)  ** 0.5
                    else:
                        l1 = l1 @ S - HS

                    l1 = torch.linalg.norm(l1).cpu() ** 2
                    losses["loss_1"][k] = 0.5 * eta_1 * l1

                    # Panchromatic fidelity term
                    l2 = torch.linalg.norm(A @ Sr - P).cpu() ** 2
                    losses["loss_2"][k] = 0.5 * eta_2 * l2

                    l = 0.5 * (eta_1 * l1 + eta_2 * l2)

                    # Regularization terms
                    A_torch = A.reshape(self.lr_im_shape).unsqueeze(0).moveaxis(-1, 1)

                    if type_reg in ["h1"]:
                        d_spat = derivative_spat(A_torch)

                        l3 = torch.linalg.norm(d_spat.flatten(), ord=2).cpu() ** 2
                        losses["loss_3"][k] = 0.5 * eta_3 * l3
                        l += 0.5 * eta_3 * l3

                    elif type_reg == "tv":
                        grad = derivative_spat(A_torch)
                        l3 = torch.linalg.norm(grad.flatten(), ord=1).cpu()
                        losses["loss_3"][k] = eta_3 * l3
                        l += eta_3 * l3

                # compute metrics during iterations
                if X_ref is not None:
                    X = A @ S
                    metrics['CC'][k] = CC(X.reshape(self.im_shape), X_ref)
                    metrics['SAM'][k] = SAM(X.reshape(self.im_shape), X_ref)
                    metrics['RMSE'][k] = RMSE(X.reshape(self.im_shape), X_ref)

                if torch.linalg.norm(A - A_old) < 1e-5 and k > 0:
                    print("Convergence reached at iteration %d" % k)
                    break

            plt.show()

            metadata = {'loss': losses, 'metrics': metrics, 'rank': self.rank, 'abundance': A, 'abundance_0': A0}

            X = A @ S
            X_torch = X.reshape(self.im_shape).unsqueeze(0).moveaxis(-1, 1)

            if k == n_iter - 1:
                print("Maximum number of iterations reached, convergence not guaranteed. Residual norm = %.5f" % torch.linalg.norm(A - A_old))

        return X_torch, metadata
        

    def forward(self, HS, P, type_reg: str = 'h1', eta_3: int = 1e-1, order: int = 1, var_noise: bool = False, weighted = False, 
        tau: float = None, n_iter: int = 500, step_show: int = None, compute_loss: bool = False, X_ref: torch.tensor = None, 
        eta_w_3: float = 100, eta_w_4: float = 10, eps_w: float = 1e-5):
        """
        Proximal gradient descent for pansharpening. We minimize 

        .. math::
            \arg \min_{a \in mathbb{R}^{\hat{N} M} \eta_1 \mathcal{D}_1(L a) + \eta_2 \mathcal{D}_2(L a) + \eta_3 \mathcal{R}_{x, y}(a)


        .. warning::
            TODO : Improve the documentation
        
        where 
        :math:`\mathcal{D}_1(L a) = \frac{1}{2} \| m - M_1 L a \|_{\Gamma^{-1}}^2`
        :math:`\mathcal{D}_2(L a) = \frac{1}{2} \| p - M_2 L a \|_2^2`.

        Args: TODO (definition of M_1, M_2, \mathcal{R}, \Gamma and W in the different cases)
        """

        # Compute cov matrix on unsplit data
        Sigma_alpha = self.build_cov_mat(HS) if var_noise else None

        HS = self._preprocess(HS)
        self._init_shapes(HS, P)

        P = P.squeeze().reshape((self.N_hat, 1))
        HS = HS.squeeze().moveaxis(0, -1).reshape((self.M, self.n_wav))

        gamma, gamma_adj = self.build_psf()
        up_factor = self.N_hat // self.L
        
        calib_op = Calibration(order=order, spc_shape=(self.l, self.l), cmos_shape=(self.height, self.width), homography=self.homography)

        if self.R is None:
            r_estim = self.estim_relative_response(HS, P, calib_op, gamma)
        else:
            r_estim = self.R
        
        # Initial guess
        if isinstance(self.meas_op, torch.nn.Identity):
            X_interp = up_factor * convolution(calib_op.vfast_prod_Gt_vec(HS).reshape(self.im_shape), gamma_adj)
        else:
            if isinstance(self.meas_op, HadamSplit2d):
                f_HS = self.meas_op.fast_pinv(HS.moveaxis(-1, 0), vectorize=True).moveaxis(0, -1)
            elif isinstance(self.meas_op, DynamicLinear):
                # send to cpu for faster linalg (on a single img)
                H_dyn = self.meas_op.H_dyn.to('cpu')

                s = torch.linalg.svdvals(H_dyn)
                self.max_eig_Hdyn = s[0].item()

                Dx, Dy = spytorch.neumann_boundary((self.l, self.l))
                D2 = Dx.T @ Dx + Dy.T @ Dy

                eta = 1e-1  # doesnt need to much tuning, as it is just for the initial guess (simulé recuit)
                f_HS = torch.linalg.solve(H_dyn.T @ H_dyn + eta * self.max_eig_Hdyn ** 2 * D2, H_dyn.T @ HS.to('cpu')).reshape((self.L, self.n_wav)).to(self.device)

            X_interp = up_factor * convolution(calib_op.vfast_prod_Gt_vec(f_HS).reshape(self.im_shape), gamma_adj)

        A0, S = self.factorize(X_interp, rank=self.rank, fact=self.fact)

        # Prior image (used with the weighted PGD to obtain a map of the color distribution)
        W = self.compute_metric(weighted=weighted, X_interp=X_interp, eta_w_3=eta_w_3, eta_w_4=eta_w_4, eps_w=eps_w)
        U, ev_U, C = self.compute_lr_metric(W, S)

        self.display_image(X_interp, self.im_shape, k=0, step_show=step_show)

        if isinstance(self.meas_op, HadamSplit2d):
            eta_1 = 1 / self.N 
        elif isinstance(self.meas_op, DynamicLinear):
            eta_1 = 1 / self.max_eig_Hdyn ** 2
        else:
            eta_1 = 1
        eta_2 = 1

        eta_1 /= torch.linalg.norm(S, ord=2) ** 2
        eta_2 /= torch.linalg.norm(S @ r_estim, ord=2) ** 2

        if var_noise:
            Sigma_alpha_pinv = torch.where(torch.abs(Sigma_alpha) > 1e-6, 1 / Sigma_alpha, torch.zeros_like(Sigma_alpha))
            eta_1 /= Sigma_alpha_pinv.max()

        tau = self.set_stepsize_pgd(tau, eta_1, eta_2, Sigma_alpha, r_estim, S)

        exp_input = {
            'HS': HS, 'P': P, 'R': r_estim, 'calib_op': calib_op, 'psf': gamma, 'psf_adj': gamma_adj,'spectral_basis': S, 'noise_variance': Sigma_alpha, 'U': U, 'C': C
        }

        X, metadata = self.pan_pgd_lr(exp_input, A0=A0, type_reg=type_reg, eta_1=eta_1, eta_2=eta_2, eta_3=eta_3,
                                n_iter=n_iter, tau=tau, step_show=step_show, compute_loss=compute_loss, X_ref=X_ref)

        metadata['exp_input'] = exp_input
        metadata['max_eig_U'] = ev_U if weighted else None
        metadata['W'] = W if weighted else None

        metadata['X_interp'] = X_interp

        return X, metadata
        
