# from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# from fht import *
from ..misc.pattern_choice import Hadamard, matrix2conv, split
from collections import OrderedDict
from pathlib import Path
import cv2
from scipy.stats import rankdata
from numpy import linalg as LA
from itertools import cycle;

from ..misc.disp import *
from ..misc.walsh_hadamard import *

#######################################################################
# 1. Determine the important Hadamard Coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Helps determining the statistical best
# Hadamard patterns for a given image size
#

def hadamard_opt_spc(M, root, nx, ny):
    msk = np.ones((nx, ny))
    had_mat = np.load(root + '{}x{}'.format(nx, ny) + '.npy');
    had_comp = np.reshape(rankdata(-had_mat, method='ordinal'), (nx, ny));
    msk[np.absolute(had_comp) > M] = 0;

    conv = Hadamard(msk);

    return conv


def Stat_had(dataloader, root):
    """
        Computes Mean Hadamard Image over the whole dataset +
        Covariance Matrix Amongst the coefficients
    """

    inputs, classes = next(iter(dataloader))
    inputs = inputs.cpu().detach().numpy();
    (batch_size, channels, nx, ny) = inputs.shape;
    tot_num = len(dataloader) * batch_size;

    Mean_had = np.zeros((nx, ny));
    for inputs, labels in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i, 0, :, :];
            h_img = fht2(img);
            Mean_had += h_img;
    Mean_had = Mean_had / tot_num;

    Cov_had = np.zeros((nx * ny, nx * ny));
    for inputs, labels in dataloader:
        inputs = inputs.cpu().detach().numpy();
        for i in range(inputs.shape[0]):
            img = inputs[i, 0, :, :];
            h_img = fht2(img);
            Norm_Variable = np.reshape(h_img - Mean_had, (nx * ny, 1));
            Cov_had += Norm_Variable * np.transpose(Norm_Variable);
    Cov_had = Cov_had / (tot_num - 1);

    np.save(root + 'Cov_{}x{}'.format(nx, ny) + '.npy', Cov_had)
    np.savetxt(root + 'Cov_{}x{}'.format(nx, ny) + '.txt', Cov_had)

    np.save(root + 'Average_{}x{}'.format(nx, ny) + '.npy', Mean_had)
    np.savetxt(root + 'Average_{}x{}'.format(nx, ny) + '.txt', Mean_had)
    cv2.imwrite(root + 'Average_{}x{}'.format(nx, ny) + '.png', Mean_had)  # Needs conversion to Uint8!
    return Mean_had, Cov_had


def stat_walsh(dataloader, device, root):
    # Get dimensions and estimate total number of images in the dataset
    inputs, classes = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape;
    tot_num = len(dataloader) * b;

    # 1. Mean

    # Init
    n = 0
    mean = torch.zeros((nx, ny), dtype=torch.float32)
    H = walsh_matrix(nx).astype(np.float32, copy=False)

    # Send to device (e.g., cuda)
    mean = mean.to(device)
    H = torch.from_numpy(H).to(device)

    # Accumulate sum over all images in dataset
    for inputs, _ in dataloader:
        inputs = inputs.to(device);
        trans = walsh2_torch(inputs, H)
        mean = mean.add(torch.sum(trans, 0))
        # print
        n = n + inputs.shape[0]
        print(f'Mean:  {n} / (less than) {tot_num} images', end='\n')
    print('', end='\n')

    # Normalize
    mean = mean / n;
    mean = torch.squeeze(mean)
    # torch.save(mean, root+'Average_{}x{}'.format(nx,ny)+'.pth')
    np.save(root / Path('Average_{}x{}'.format(nx, ny) + '.npy'), mean.cpu().detach().numpy())

    # 2. Covariance

    # Init
    n = 0
    cov = torch.zeros((nx * ny, nx * ny), dtype=torch.float32)
    cov = cov.to(device)

    # Accumulate (im - mu)*(im - mu)^T over all images in dataset
    for inputs, _ in dataloader:
        inputs = inputs.to(device);
        trans = walsh2_torch(inputs, H)
        trans = trans - mean.repeat(inputs.shape[0], 1, 1, 1)
        trans = trans.view(inputs.shape[0], nx * ny, 1)
        cov = torch.addbmm(cov, trans, trans.view(inputs.shape[0], 1, nx * ny))
        # print
        n += inputs.shape[0]
        print(f'Cov:  {n} / (less than) {tot_num} images', end='\n')
    print('', end='\n')

    # Normalize
    cov = cov / (n - 1);
    # torch.save(cov, root+'Cov_{}x{}'.format(nx,ny)+'.pth')
    np.save(root / Path('Cov_{}x{}'.format(nx, ny) + '.npy'), cov.cpu().detach().numpy())

    return mean, cov


def img2mask(Value_map, M):
    (nx, ny) = Value_map.shape;
    msk = np.ones((nx, ny));
    ranked_data = np.reshape(rankdata(-Value_map, method='ordinal'), (nx, ny));
    msk[np.absolute(ranked_data) > M] = 0;
    return msk


def Cov2Var(Cov):
    """
    Extracts Variance Matrix from Covarience Matrix
    """
    (Nx, Ny) = Cov.shape;
    diag_index = np.diag_indices(Nx);
    Var = Cov[diag_index];
    Var = np.reshape(Var, (int(np.sqrt(Nx)), int(np.sqrt(Nx))));
    return Var

def Permutation_Matrix_root(root):
    """
        Returns Permutaion Matrix For The Hadamard Coefficients that ranks
        The Coefficients according to the Matrix defined By root.
    """
    had_mat = np.load(root);
    (nx, ny) = had_mat.shape;
    Reorder = rankdata(-had_mat, method='ordinal');
    Columns = np.array(range(nx * ny));
    P = np.zeros((nx * ny, nx * ny));
    P[Reorder - 1, Columns] = 1;
    return P


def Permutation_Matrix(mat):
    """
        Returns permutation matrix from sampling map

    Args:
        mat (np.ndarray): A a n-by-n sampling map, where high value means high significance.

    Returns:
        P (np.ndarray): A n*n-by-n*n permutation matrix
    """
    (nx, ny) = mat.shape;
    Reorder = rankdata(-mat, method='ordinal');
    Columns = np.array(range(nx * ny));
    P = np.zeros((nx * ny, nx * ny));
    P[Reorder - 1, Columns] = 1;
    return P


def maximum_Variance_Pattern(Cov, H, M):
    """
        Returns the patterns corresponding to coefficient that have the maximun
        variance for a given image database
    """
    Var = Cov2Var(Cov)
    Perm = Permutation_Matrix(Var)
    Pmat = np.dot(Perm, H);
    Pmat = Pmat[:M, :];
    return Pmat, Perm


def Hadamard_Transform_Matrix(img_size):
    H = np.zeros((img_size ** 2, img_size ** 2))
    for i in range(img_size ** 2):
        base_function = np.zeros((img_size ** 2, 1));
        base_function[i] = 1;
        base_function = np.reshape(base_function, (img_size, img_size));
        hadamard_function = fht2(base_function);
        H[i, :] = np.reshape(hadamard_function, (1, img_size ** 2));
    return H


def Hadamard_stat_completion_matrices(Cov_had, Mean_had, CR):
    img_size, ny = Mean_had.shape;

    # choice of patterns
    Var = Cov2Var(Cov_had)
    P = Permutation_Matrix(Var)
    H = Hadamard_Transform_Matrix(img_size);

    Sigma = np.dot(P, np.dot(Cov_had, np.transpose(P)))
    mu = np.dot(P, np.reshape(Mean_had, (img_size ** 2, 1)))
    mu1 = mu[:CR];

    Sigma1 = Sigma[:CR, :CR]
    Sigma21 = Sigma[CR:, :CR]

    W_p = np.zeros((img_size ** 2, CR))
    W_p[:CR, :] = np.eye(CR);
    W_p[CR:, :] = np.dot(Sigma21, np.linalg.inv(Sigma1));

    W = np.dot(H, np.dot(np.transpose(P), W_p));
    b = np.dot(H, np.dot(np.transpose(P), mu));
    return W, b, mu1, P, H


def stat_denoising_matrices(P, Cov_had, NVMS, n, CR):
    Sigma = n ** 2 / 4 * np.dot(P, np.dot(Cov_had, np.transpose(P)))
    Sigma1 = Sigma[:CR, :CR]

    NVMS_inv = np.linalg.inv(Sigma1 + NVMS)
    Product0 = np.matmul(NVMS, NVMS_inv)
    Product1 = np.matmul(Sigma1, NVMS_inv)
    Product2 = np.matmul(Product1, np.eye(CR) + Product0)

    return NVMS_inv, Product1, Product2


def stat_completion_matrices(P, H, Cov_had, Mean_had, CR):
    img_size, ny = Mean_had.shape;

    Sigma = np.dot(P, np.dot(Cov_had, np.transpose(P)))
    mu = np.dot(P, np.reshape(Mean_had, (img_size ** 2, 1)))
    mu1 = mu[:CR];

    Sigma1 = Sigma[:CR, :CR]
    Sigma21 = Sigma[CR:, :CR]

    W_p = np.zeros((img_size ** 2, CR))
    W_p[:CR, :] = np.eye(CR);
    W_p[CR:, :] = np.dot(Sigma21, np.linalg.inv(Sigma1));

    W = np.dot(H, np.dot(np.transpose(P), W_p));
    b = np.dot(H, np.dot(np.transpose(P), mu));
    return W, b, mu1


def Hadamard_stat_completion_extract(img, CR, P, H):
    img_size, ny = img.shape;
    f = np.reshape(img, (img_size ** 2, 1))
    y = np.dot(P, np.dot(H, f))
    m = y[:CR];
    return m


def Hadamard_stat_completion(W, b, mu1, m):
    nxny, col = b.shape;
    img_size = int(round(np.sqrt(nxny)));
    f_star = b + np.dot(W, (m - mu1))
    img_rec = np.reshape(f_star, (img_size, img_size));
    return img_rec;


def Hadamard_stat_completion_comp(Cov, Mean, img, CR):
    img_size, ny = img.shape;
    Var = Cov2Var(Cov)
    P = Permutation_Matrix(Var)
    H = Hadamard_Transform_Matrix(img_size);

    Sigma = np.dot(P, np.dot(Cov, np.transpose(P)))
    mu = np.dot(P, np.reshape(Mean, (img_size ** 2, 1)))
    mu1 = mu[:CR];

    Sigma1 = Sigma[:CR, :CR]
    Sigma21 = Sigma[CR:, :CR]

    W_p = np.zeros((img_size ** 2, CR))
    W_p[:CR, :] = np.eye(CR);
    W_p[CR:, :] = np.dot(Sigma21, np.linalg.inv(Sigma1));

    W = np.dot(H, np.dot(np.transpose(P), W_p));
    b = np.dot(H, np.dot(np.transpose(P), mu));

    f = np.reshape(img, (img_size ** 2, 1))
    y = np.dot(P, np.dot(H, f))
    m = y[:CR];
    f_star = b + np.dot(W, (m - mu1))
    img_rec = np.reshape(f_star, (img_size, img_size));

    return img_rec;


###############################################################################
# 2. NEW Convolutional Neural Network
###############################################################################
# ==============================================================================
# A. NO NOISE
# ==============================================================================
class compNet(nn.Module):
    def __init__(self, n, M, Mean, Cov, variant=0, H=None, Ord=None):
        super(compNet, self).__init__()

        self.n = n;
        self.M = M;

        self.even_index = range(0, 2 * M, 2);
        self.uneven_index = range(1, 2 * M, 2);

        self.Cov = Cov

        # -- Hadamard patterns (full basis)
        if type(H) == type(None):
            H = Hadamard_Transform_Matrix(self.n)
        H = n * H;  # fht hadamard transform needs to be normalized

        self.H = H

        # -- Hadamard patterns (undersampled basis)
        if type(Ord) == type(None):
            Ord = Cov2Var(Cov)

        Perm = Permutation_Matrix(Ord)
        self.Perm = Perm
        Pmat = np.dot(Perm, H);
        Pmat = Pmat[:M, :];
        self.Pmat = Pmat
        Pconv = matrix2conv(Pmat);

        # -- Denoising parameters
        Sigma = np.dot(Perm, np.dot(Cov, np.transpose(Perm)));
        Sigma1 = n ** 2 / 4 * torch.Tensor(Sigma[:M, :M])
        # Sigma1 = torch.Tensor(Sigma[:M, :M])
        self.Sigma1 = Sigma1.view(1, M, M)
        diag_index = np.diag_indices(n ** 2);
        Sigma = Sigma[diag_index];
        Sigma = n ** 2 / 4 * Sigma[:M];  # (H = nH donc Cov = n**2 Cov)!
        # Sigma = Sigma[:M];
        Sigma = torch.Tensor(Sigma);
        self.sigma = Sigma.view(1, 1, M);
        self.T1 = self.Sigma1 - torch.diag_embed(self.sigma).view([1, self.M, self.M])

        P1 = np.zeros((n ** 2, 1));
        P1[0] = n ** 2;
        mean = n * np.reshape(Mean, (self.n ** 2, 1)) + P1;
        mu = (1 / 2) * np.dot(Perm, mean);
        # mu = np.dot(Perm, np.reshape(Mean, (n**2,1)))
        mu1 = torch.Tensor(mu[:M]);
        self.mu_1 = mu1.view(1, 1, M);

        # -- Measurement preprocessing
        self.Patt = Pconv;
        P, T = split(Pconv, 1);
        self.P = P;
        self.T = T;
        self.P.bias.requires_grad = False;
        self.P.weight.requires_grad = False;
        self.Patt.bias.requires_grad = False;
        self.Patt.weight.requires_grad = False;
        self.T.weight.requires_grad = False;
        self.T.weight.requires_grad = False;

        # -- Pseudo-inverse to determine levels of noise.
        Pinv = (1 / n ** 2) * np.transpose(Pmat);
        self.Pinv = nn.Linear(M, n ** 2, False)
        self.Pinv.weight.data = torch.from_numpy(Pinv);
        self.Pinv.weight.data = self.Pinv.weight.data.float();
        self.Pinv.weight.requires_grad = False;

        # -- Measurement to image domain
        if variant == 0:
            # --- Statistical Matrix completion (no mean)
            print("Measurement to image domain: statistical completion (no mean)")

            self.fc1 = nn.Linear(M, n ** 2, False)

            W, b, mu1 = stat_completion_matrices(Perm, H, Cov, Mean, M)
            W = (1 / n ** 2) * W;

            self.fc1.weight.data = torch.from_numpy(W);
            self.fc1.weight.data = self.fc1.weight.data.float();
            self.fc1.weight.requires_grad = False;

        if variant == 1:
            # --- Statistical Matrix completion
            print("Measurement to image domain: statistical completion")

            self.fc1 = nn.Linear(M, n ** 2)

            W, b, mu1 = stat_completion_matrices(Perm, H, Cov, Mean, M)
            W = (1 / n ** 2) * W;
            b = (1 / n ** 2) * b;
            b = b - np.dot(W, mu1);
            self.fc1.bias.data = torch.from_numpy(b[:, 0]);
            self.fc1.bias.data = self.fc1.bias.data.float();
            self.fc1.bias.requires_grad = False;
            self.fc1.weight.data = torch.from_numpy(W);
            self.fc1.weight.data = self.fc1.weight.data.float();
            self.fc1.weight.requires_grad = False;

        elif variant == 2:
            # --- Pseudo-inverse
            print("Measurement to image domain: pseudo inverse")

            self.fc1 = self.Pinv;

        elif variant == 3:
            # --- FC is learnt
            print("Measurement to image domain: free")

            self.fc1 = nn.Linear(M, n ** 2)

        # -- Image correction

        self.recon = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2))
        ]));

        # self.recon = UNet()

    def forward(self, x):
        b, c, h, w = x.shape;
        x = self.forward_acquire(x, b, c, h, w)
        x = self.forward_reconstruct(x, b, c, h, w)
        return x

    # --------------------------------------------------------------------------
    # Forward functions (with grad)
    # --------------------------------------------------------------------------
    def forward_acquire(self, x, b, c, h, w):
        # --Scale input image
        x = (x + 1) / 2;
        # --Acquisition
        x = x.view(b * c, 1, h, w);
        x = self.P(x);
        x = F.relu(x);
        x = x.view(b * c, 1, 2 * self.M);
        return x

    def forward_reconstruct(self, x, b, c, h, w):
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w)
        return x

    def forward_reconstruct_pinv(self, x, b, c, h, w):
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.pinv(x, b, c, h, w);
        return x

    def forward_reconstruct_mmse(self, x, b, c, h, w):
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        return x

    def forward_preprocess(self, x, b, c, h, w):
        # - Pre-processing (use batch norm to avoid division by N0 ?)
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b * c, 1, self.M));
        return x

    def forward_maptoimage(self, x, b, c, h, w):
        # --Projection to the image domain
        x = self.fc1(x);
        x = x.view(b * c, 1, h, w)
        return x

    def forward_postprocess(self, x, b, c, h, w):
        x = self.recon(x)
        x = x.view(b, c, h, w)
        return x

    def pinv(self, x, b, c, h, w):
        x = self.Pinv(x);
        x = x.view(b * c, 1, h, w)
        return x

    # --------------------------------------------------------------------------
    # Evaluation functions (no grad)
    # --------------------------------------------------------------------------
    def acquire(self, x, b, c, h, w):
        with torch.no_grad():
            b, c, h, w = x.shape
            x = self.forward_acquire(x, b, c, h, w)
        return x

    def evaluate_fcl(self, x):
        with torch.no_grad():
            b, c, h, w = x.shape
            x = self.forward_acquire(x, b, c, h, w)
            x = self.forward_maptoimage(x, b, c, h, w)
        return x

    def evaluate_Pinv(self, x):
        with torch.no_grad():
            b, c, h, w = x.shape
            x = self.forward_Pinv(x, b, c, h, w)
        return x

    def evaluate(self, x):
        with torch.no_grad():
            x = self.forward(x)
        return x

    def reconstruct(self, x, b, c, h, w):
        with torch.no_grad():
            x = self.forward_reconstruct(x, b, c, h, w)
        return x


# ==============================================================================
# B. NOISY MEASUREMENTS (NOISE LEVEL IS VARYING)
# ==============================================================================
class noiCompNet(compNet):
    def __init__(self, n, M, Mean, Cov, variant, N0=2500, sig=0.5, H=None, Ord=None):
        super().__init__(n, M, Mean, Cov, variant, H, Ord)
        self.N0 = N0;
        self.sig = sig;
        self.max = nn.MaxPool2d(kernel_size=n);
        print("Varying N0 = {:g} +/- {:g}".format(N0, sig * N0))

    def forward_acquire(self, x, b, c, h, w):
        # --Scale input image
        a = self.N0 * (1 + self.sig * (torch.rand(x.shape[0]) - 0.5)).to(x.device)
        # print('alpha in [{}--{}] photons'.format(min(a).item(), max(a).item()))
        x = a.view(-1, 1, 1, 1) * (x + 1) / 2;

        # --Acquisition
        x = x.view(b * c, 1, h, w);
        x = self.P(x);
        x = F.relu(x);  # x[:,:,1] = -1/N0 ????
        x = x.view(b * c, 1, 2 * self.M);  # x[:,:,1] < 0???

        # --Measurement noise (Gaussian approximation of Poisson)
        x = x + torch.sqrt(x) * torch.randn_like(x);
        return x

    def forward_preprocess(self, x, b, c, h, w):
        # -- Pre-processing(Recombining positive and negative values+normalisation)
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];
        x = x / self.N0;
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b * c, 1, self.M));

        return x

    def forward_reconstruct(self, x, b, c, h, w):
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w)
        return x

    def forward_maptomeasure(self, x, b, c, h, w):
        x = x.view(b * c, 1, h, w);
        x = self.P(x);
        x = F.relu(x);
        x = x.view(b * c, 1, 2 * self.M);
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index]

        return x

    def forward_reconstruct_expe(self, x, b, c, h, w):
        x, N0_est = self.forward_preprocess_expe(x, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w)
        return x

    def forward_reconstruct_pinv_expe(self, x, b, c, h, w):
        x = self.forward_preprocess_expe(x, b, c, h, w)
        x = self.pinv(x, b, c, h, w)
        # -- Faster alternative
        # x = x[:,:,self.even_index] - x[:,:,self.uneven_index]
        # x = self.pinv(x, b, c, h, w);
        # N0_est = self.max(x);
        # N0_est = N0_est.view(b*c,1,1,1);
        # N0_est = N0_est.repeat(1,1,h,w);
        # x = torch.div(x,N0_est);
        # x=2*x-1;
        return x

    def forward_preprocess_expe(self, x, b, c, h, w):

        x = x.view(b * c, 1, 2 * self.M)

        # -- Recombining positive and negative values
        x = x[:, :, self.even_index] - x[:, :, self.uneven_index];

        # -- Estimating and normalizing by N0 = K*alpha
        x_est = self.pinv(x, b, c, h, w);
        N0 = self.max(x_est)
        N0 = N0.view(b, c, 1)
        print(N0)

        # --
        N0_est = N0.repeat(1, 1, self.M)
        x = torch.div(x, N0_est)
        x = 2 * x - torch.reshape(self.Patt(torch.ones(b * c, 1, h, w).to(x.device)), (b, c, self.M))
        return x, N0

# =============================================================================================
# B. NOISY MEASUREMENTS (NOISE LEVEL IS VARYING) + denoising architecture with iterative method
# =============================================================================================


class DenoiCompNet_NVMS(noiCompNet):
    def __init__(self, n, M, Mean, Cov, NVMS, Niter=4, variant=0, denoi=0, N0=50, sig=0.0, H=None, Ord=None):
        super().__init__(n, M, Mean, Cov, variant, N0, sig, H, Ord)
        self.denoi = denoi
        self.Niter = Niter
        # -- Hadamard operator H1 (image push forward)
        self.H1 = nn.Linear(self.n ** 2, self.M, False)
        self.H1.weight.data = torch.from_numpy(self.Pmat)
        self.H1.weight.data = self.H1.weight.data.float()
        self.H1.weight.requires_grad = False

        if denoi == 0:
            self.fcP0 = nn.Linear(M, M, False)
            self.fcP1 = nn.Linear(M, M, False)
            self.fcP2 = nn.Linear(M, M, False)

            Pr0, Pr1, Pr2 = stat_denoising_matrices(self.Perm, Cov, NVMS, n, M)

            self.fcP0.weight.data = torch.from_numpy(Pr0);
            self.fcP0.weight.data = self.fcP0.weight.data.float();
            self.fcP0.weight.requires_grad = False;

            self.fcP1.weight.data = torch.from_numpy(Pr1);
            self.fcP1.weight.data = self.fcP1.weight.data.float();
            self.fcP1.weight.requires_grad = False;

            self.fcP2.weight.data = torch.from_numpy(Pr2);
            self.fcP2.weight.data = self.fcP2.weight.data.float();
            self.fcP2.weight.requires_grad = False;

            print("Iterative Neural Network with denoised measurements (by NVMS approximation ). For {} iterations".format(Niter))

            """
            This method computes the denoising of the raw data in the measurement
            domain, based on a precalculate a Noise Variance Matrix Stabilization (NVMS),
            which is a matrix that takes the mean of the variance of the noised measurements, 
            for a given photon level N0 on a batch of the STL-10 database. This method allows  
            to stabilize the signal dependent variance matrix in the denoising stage. 
            A first-order taylor development is taken also for tackle the matrix inversion.
            """
        elif denoi == 1:

            print("Iterative Neural Network with denoised measurements (by diagonal approximation ). For {} iterations".format(Niter))

        elif denoi == 2:

            print("Iterative Neural Network with denoised measurements (by full inverse product). For {} iterations".format(Niter))

    def forward(self, x):
        b, c, h, w = x.shape;
        x = self.forward_acquire(x, b, c, h, w)
        x = self.forward_reconstruct(x, b, c, h, w)
        return x

    def forward_reconstruct(self, x, b, c, h, w):
        #  -- f(0) : image initialization
        f = torch.zeros(b * c, 1, h, w).to(x.device)
        # -- Variance estimation for the measurements
        x, var = self.forward_variance(x, b, c, h, w)
        # -- Normalization and combination of pos/neg coefficients
        x = self.forward_preprocess(x, b, c, h, w)

        # -- Iterative image correction
        for k in range(0, self.Niter):
            # -- Variation in the measurements domain
            epsilon = x - self.H1(f.view(b * c, 1, h * w))
            # -- Linear denoising (measurements domain)
            epsilon = self.forward_denoise(epsilon, var, b, c, h, w)
            # -- Linear completion (image domain)
            f = f + self.forward_maptoimage(epsilon, b, c, h, w)
            # -- Non linear image correction
            f = self.forward_postprocess(f, b, c, h, w)

        return f

    def forward_variance(self, x, b, c, h, w):
        var = torch.mean(x[:, :, self. even_index] + x[:, :, self.uneven_index], [1, 2])
        x[:, 0, self.even_index][:, 0] = var
        var = var.view(b * c, 1, 1)
        var = var.repeat(1, 1, self.M)

        return x, var

    def forward_denoise(self, x, var, b, c, h, w):
        var = torch.div(var, self.N0 ** 2)

        if self.denoi == 0:
            x = self.fcP2(x) - torch.mul(self.fcP1(var), self.fcP0(x));

        elif self.denoi == 1:
            sigma = self.sigma.repeat(b * c, 1, 1).to(x.device);
            x = torch.mul(torch.div(sigma, sigma + var), x)

        else:
            var = torch.diag_embed(var).view([b * c, self.M, self.M])
            x = torch.matmul(torch.matmul(self.Sigma1.to(x.device),
                                          torch.linalg.inv(self.Sigma1.to(x.device) + var)),
                             torch.transpose(x, 1, 2))
            x = torch.reshape(x, (b * c, 1, self.M))

        return x

    def forward_maptoimage_expe(self, x, NVMS, b, c, h, w, C, s, K):
        #  -- f(0) : image initialization
        f = torch.zeros(b * c, 1, h, w).to(x.device)
        # -- Variance estimation for the measurements
        x = x.view(b * c, 1, 2 * self.M)
        var = K * (x[:, :, self.even_index] + x[:, :, self.uneven_index] - 2 * C) + 2 * s ** 2
        # -- Normalization and combination of pos/neg coefficients
        x, N0_est = self.forward_preprocess_expe(x, b, c, h, w)
        var = torch.div(var, N0_est ** 2)

        if self.denoi == 0:
            NVMS_est = NVMS / N0_est[0, 0, 0].numpy()
            P0, P1, P2 = self.forward_denoise_operators(self.Cov, NVMS_est, self.n, self.M)

        # -- Iterative image correction
        for k in range(0, self.Niter):
            # -- Variation in the measurements domain
            epsilon = x - self.H1(f.view(b * c, 1, h * w))
            # -- Linear denoising (measurements domain)
            if self.denoi == 0:
                epsilon = self.forward_denoise_expe_nvms(epsilon, var, P0, P1, P2, b, c, h, w)
            else:
                epsilon = self.forward_denoise_expe(epsilon, var, b, c, h, w)
            # -- Linear completion (image domain)
            f = f + self.forward_maptoimage(epsilon, b, c, h, w)

        return  f

    def forward_reconstruct_expe(self, x, NVMS, b, c, h, w, C, s, K):
        f = self.forward_maptoimage_expe(self, x, NVMS, b, c, h, w, C, s, K)
        # -- Non linear image correction
        f = self.forward_postprocess(f, b, c, h, w)

        return f

    def forward_denoise_operators(self, Cov, NVMS, n, M):
        P0 = nn.Linear(M, M, False)
        P1 = nn.Linear(M, M, False)
        P2 = nn.Linear(M, M, False)

        Op0, Op1, Op2 = stat_denoising_matrices(self.Perm, Cov, NVMS, n, M)

        P0.weight.data = torch.from_numpy(Op0);
        P0.weight.data = P0.weight.data.float();
        P0.weight.requires_grad = False;

        P1.weight.data = torch.from_numpy(Op1);
        P1.weight.data = P1.weight.data.float();
        P1.weight.requires_grad = False;

        P2.weight.data = torch.from_numpy(Op2);
        P2.weight.data = P2.weight.data.float();
        P2.weight.requires_grad = False;

        return P0, P1, P2

    def forward_denoise_expe_nvms(self, x, var, P0, P1, P2, b, c, h, w):
        x = P2(x) - torch.mul(P1(var), P0(x));
        return x

    def forward_denoise_expe(self, x, var, b, c, h, w):
        if self.denoi == 1:
            sigma = self.sigma.repeat(b * c, 1, 1).to(x.device);
            x = torch.mul(torch.div(sigma, sigma + var), x)

        else:
            var = torch.diag_embed(var).view([b * c, self.M, self.M])
            x = torch.matmul(torch.matmul(self.Sigma1.to(x.device),
                                          torch.linalg.inv(self.Sigma1.to(x.device) + var)),
                             torch.transpose(x, 1, 2))
            x = torch.reshape(x, (b * c, 1, self.M))

        return x

########################
# -- Variational classes
########################


class RegL1Sparse(DenoiCompNet_NVMS):
    def __init__(self, n, M, Mean, Cov, NVMS, Basis, reg, Prox_Niter, Niter=4, variant=0, denoi=0, N0=2500, sig=0.0, H=None, Ord=None):
        super().__init__(n, M, Mean, Cov, NVMS, Niter, variant, denoi, N0, sig, H, Ord)
        self.Basis = Basis
        self.reg = reg
        self.Prox_Niter = Prox_Niter
        self.R = np.matmul(self.Pmat, self.Basis)
        self.Rt = np.conj(self.R).T
        self.eta = 1 / LA.norm(np.matmul(self.Rt, self.R))

        # -- Gradient precalculated layers
        self.W = nn.Linear(self.n ** 2, self.M, False)
        self.Wt = nn.Linear(self.M, self.n ** 2, False)
        self.IT = nn.Linear(self.n ** 2, self.n ** 2, False)

        self.W.weight.data = torch.from_numpy(self.R)
        self.W.weight.data = self.W.weight.data.float()
        self.W.weight.requires_grad = False

        self.Wt.weight.data = torch.from_numpy(self.Rt)
        self.Wt.weight.data = self.Wt.weight.data.float()
        self.Wt.weight.requires_grad = False

        # -- Basis map to image domain layer
        self.IT.weight.data = torch.from_numpy(self.Basis)
        self.IT.weight.data = self.IT.weight.data.float()
        self.IT.weight.requires_grad = False

        print("Image reconstruction by L1 regularisation")

    def proximal_operator(self, x):
        x = x * torch.max(torch.zeros(x.shape), 1 - (self.reg * self.eta)/torch.max(torch.abs(x), 1e-10 * torch.ones(x.shape)))

        return x

    def forward_maptoimage(self, m, b, c, h, w):
        # -- Initialisation of the coefficients vector
        x = torch.zeros(b, c, h * w)

        # -- ISTA Algorithm
        for k in range(1, self.Prox_Niter):
            gradient = self.Wt(self.W(x) - m)
            x = x - self.eta * gradient
            x = self.proximal_operator(x)

        # -- Map to image domain
        x = self.IT(x).view(b, c, h, w)

        return x

    def forward_maptoimage_FISTA(self, m, b, c, h, w):
        # -- Initialisation of the coefficients vector
        x1 = torch.zeros(b, c, h * w)
        x = x1.clone().detach()
        t1 = 1

        # -- FISTA Algorithm
        for k in range(1, self.Prox_Niter):
            gradient = self.Wt(self.W(x1) - m)
            x1 = self.proximal_operator(x1 - self.eta * gradient)
            t2 = (1 + np.sqrt(1 + 4 * t1 ** 2)) / 2
            x1 = x1 + ((t1 - 1) / t2) * (x1 - x)

            t1 = t2
            x = x1.clone().detach()

        # -- Map to image domain
        x = self.IT(x).view(b, c, h, w)

        return x

    def forward_reconstruct(self, x, b, c, h, w):
        x, var = self.forward_variance(x, b, c, h, w)
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.forward_denoise(x, var, b, c, h, w)
        x = self.forward_maptoimage(x, b, c, h, w)

        return x


class RegTVL1GRAD(DenoiCompNet_NVMS):
    def __init__(self, n, M, Mean, Cov, NVMS, reg, step_size, epsilon, eta, theta, Grad_Niter, variant=0, N0=2500, sig=0.5, H=None, Ord=None):
        super().__init__(n, M, Mean, Cov, NVMS, variant, N0, sig, H, Ord)
        self.reg = reg
        self.Grad_Niter = Grad_Niter
        self.step_size = step_size
        self.epsilon = epsilon
        self.eta = eta
        self.theta = theta

        # -- Gradient precalculated layers
        self.W = nn.Linear(self.n ** 2, self.M, False)
        self.Wt = nn.Linear(self.M, self.n ** 2, False)

        self.W.weight.data = torch.from_numpy(self.Pmat)
        self.W.weight.data = self.W.weight.data.float()
        self.W.weight.requires_grad = False

        self.Wt.weight.data = torch.from_numpy(np.conj(self.Pmat).T)
        self.Wt.weight.data = self.Wt.weight.data.float()
        self.Wt.weight.requires_grad = False

        print("Image reconstruction by TV-L2 regularisation")

    def grad2D(self, x, b, c, h, w):
        In1 = x[:, :, h - 1, :]
        In1 = In1.view(b, c, 1, h)
        Ifx = torch.cat((x[:, :, 1:, :], In1), 2)
        Dx = Ifx - x

        In2 = x[:, :, :, w - 1]
        In2 = In2.view(b, c, w, 1)
        Ify = torch.cat((x[:, :, :, 1:], In2), 3)
        Dy = Ify - x

        return [Dx, Dy]

    def divergence2D(self, gx, gy, b, c, h, w):
        gx0 = gx[:, :, 0, :]
        gx0 = gx0.view(b, c, 1, h)
        gx1 = -gx[:, :, h - 2, :]
        gx1 = gx1.view(b, c, 1, h)
        divergence_x = torch.cat((gx0, gx[:, :, 1:h - 1, :] - gx[:, :, :h - 2, :], gx1), 2)

        gy0 = gy[:, :, :, 0]
        gy0 = gy0.view(b, c, w, 1)
        gy1 = -gy[:, :, :, w - 2]
        gy1 = gy1.view(b, c, w, 1)
        divergence_y = torch.cat((gy0, gy[:, :, :, 1:w - 1] - gy[:, :, :, :w - 2], gy1), 3)

        return divergence_x + divergence_y

    def forward_adjoint(self, x, m, b, c, h, w):
        [ux, uy] = self.grad2D(x, b, c, h, w)
        grad_norm = torch.sqrt((ux ** 2 + uy ** 2) + self.epsilon)

        hx = torch.div(ux, grad_norm)
        hy = torch.div(uy, grad_norm)

        div = self.divergence2D(hx, hy, b, c, h, w)
        x = self.Wt(self.W(x.view(b * c, 1, h * w)) - m) - self.reg * div.view(b * c, 1, h * w)

        return x

    def forward_gradient(self, m, b, c, h, w):
        # -- Image initialisation
        x = torch.zeros(b * c, 1, h, w)

        for i in range(1, self.Grad_Niter):
            gradient = self.forward_adjoint(x, m, b, c, h, w)
            x = x - self.step_size * gradient.view(b, c, h, w)

        return x

    def forward_gradient_conjugate(self, m, b, c, h, w):
        # -- Image initialisation
        x = self.forward_maptoimage(m, b, c, h, w)
        g0 = self.forward_adjoint(x, m, b, c, h, w)
        p = -g0.clone().detach()

        for i in range(1, self.Grad_Niter):
            x = x + self.step_size * p.view(b, c, h, w)
            g1 = self.forward_adjoint(x, m, b, c, h, w)
            beta = torch.linalg.norm(g1, dim=2) ** 2 / torch.linalg.norm(g0, dim=2) ** 2
            p = -g1 + beta.view(b * c, 1, 1) * p
            g0 = g1.clone().detach()

        x = x + self.step_size * p.view(b, c, h, w)
        return x

    def forward_reconstruct(self, x, b, c, h, w):
        x, var = self.forward_variance(x, b, c, h, w)
        x = self.forward_preprocess(x, b, c, h, w)
        x = self.forward_denoise(x, var, b, c, h, w)
        x = self.forward_gradient_conjugate(x, b, c, h, w)
        x = self.forward_postprocess(x, b, c, h, w)

        return x

    def proximal_operator(self, p, q, x):

        p[:, :, 0, :] = p[:, :, 0, :] / torch.max(torch.ones(p[:, :, 0, :].shape), torch.abs(p[:, :, 0, :]))
        p[:, :, 1, :] = p[:, :, 1, :] / torch.max(torch.ones(p[:, :, 1, :].shape), torch.abs(p[:, :, 1, :]))

        q = (q - self.eta * x) / (1 + self.eta)

        return [p, q]

    def forward_maptoimage_Primal_Dual(self, m, b, c, h, w):
        x = torch.zeros(b * c, 1, h, w)
        x_bar = x.clone().detach()
        p = torch.zeros(b * c, 1, 2, h * w)
        q = torch.zeros(b * c, 1, self.M)

        for i in range(1, self.Grad_Niter):
            [ux, uy] = self.grad2D(x_bar, b, c, h, w)
            grad = torch.cat((ux.view(b * c, 1, 1, h * w), uy.view(b * c, 1, 1, h * w)), 2)

            p = p + self.eta * self.reg * grad
            q = q + self.eta * self.W(x_bar.view(b * c, 1, h * w))
            [p, q] = self.proximal_operator(p, q, m)

            px = torch.reshape(p[:, :, 0, :], (b * c, 1, h, w))
            py = torch.reshape(p[:, :, 1, :], (b * c, 1, h, w))
            div = self.divergence2D(px, py, b, c, h, w)

            x_star = x + self.step_size * self.reg * div - self.step_size * self.Wt(q).view(b * c, 1, h, w)
            x_bar = x_star + self.theta * (x_star - x)

        return x_star

# 2. Define a custom Loss function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Creating custom loss function
# ---------------------------
# Just to make sure that all functions work the same way...   
# i.e., that they take the same number of arguments


class Weight_Decay_Loss(nn.Module):

    def __init__(self, loss):
        super(Weight_Decay_Loss, self).__init__()
        self.loss = loss;

    def forward(self, x, y, net):
        mse = self.loss(x, y);
        return mse
