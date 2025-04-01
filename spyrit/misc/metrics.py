# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt

# import skimage.metrics as skm


def batch_psnr(torch_batch, output_batch):
    list_psnr = []
    for i in range(torch_batch.shape[0]):
        img = torch_batch[i, 0, :, :]
        img_out = output_batch[i, 0, :, :]
        img = img.cpu().detach().numpy()
        img_out = img_out.cpu().detach().numpy()
        list_psnr.append(psnr(img, img_out))
    return list_psnr


def batch_psnr_(torch_batch, output_batch, r=2):
    list_psnr = []
    for i in range(torch_batch.shape[0]):
        img = torch_batch[i, 0, :, :]
        img_out = output_batch[i, 0, :, :]
        img = img.cpu().detach().numpy()
        img_out = img_out.cpu().detach().numpy()
        list_psnr.append(psnr_(img, img_out, r=r))
    return list_psnr


def batch_ssim(torch_batch, output_batch):
    list_ssim = []
    for i in range(torch_batch.shape[0]):
        img = torch_batch[i, 0, :, :]
        img_out = output_batch[i, 0, :, :]
        img = img.cpu().detach().numpy()
        img_out = img_out.cpu().detach().numpy()
        list_ssim.append(ssim(img, img_out))
    return list_ssim


def dataset_meas(dataloader, model, device):
    meas = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # with torch.no_grad():
        b, c, h, w = inputs.shape
        net_output = model.acquire(inputs, b, c, h, w)
        raw = net_output[:, 0, :]
        raw = raw.cpu().detach().numpy()
        meas.extend(raw)
    return meas


#
# def dataset_psnr_different_measures(dataloader, model, model_2, device):
#    psnr = [];
#    #psnr_fc = [];
#    for inputs, labels in dataloader:
#        inputs = inputs.to(device)
#        m = model_2.normalized measure(inputs);
#        net_output  = model.forward_reconstruct(inputs);
#        #net_output2 = model.evaluate_fcl(inputs);
#
#        psnr += batch_psnr(inputs, net_output);
#        #psnr_fc += batch_psnr(inputs, net_output2);
#    psnr = np.array(psnr);
#    #psnr_fc = np.array(psnr_fc);
#    return psnr;
#


def dataset_psnr(dataloader, model, device):
    psnr = []
    psnr_fc = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # with torch.no_grad():
        # b,c,h,w = inputs.shape;

        net_output = model.evaluate(inputs)
        net_output2 = model.evaluate_fcl(inputs)

        psnr += batch_psnr(inputs, net_output)
        psnr_fc += batch_psnr(inputs, net_output2)
    psnr = np.array(psnr)
    psnr_fc = np.array(psnr_fc)
    return psnr, psnr_fc


def dataset_ssim(dataloader, model, device):
    ssim = []
    ssim_fc = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # evaluate full model and fully connected layer
        net_output = model.evaluate(inputs)
        net_output2 = model.evaluate_fcl(inputs)
        # compute SSIM and concatenate
        ssim += batch_ssim(inputs, net_output)
        ssim_fc += batch_ssim(inputs, net_output2)
    ssim = np.array(ssim)
    ssim_fc = np.array(ssim_fc)
    return ssim, ssim_fc


def dataset_psnr_ssim(dataloader, model, device):
    # init lists
    psnr = []
    ssim = []
    # loop over batches
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # evaluate full model
        net_output = model.evaluate(inputs)
        # compute PSNRs and concatenate
        psnr += batch_psnr(inputs, net_output)
        # compute SSIMs and concatenate
        ssim += batch_ssim(inputs, net_output)
    # convert
    psnr = np.array(psnr)
    ssim = np.array(ssim)
    return psnr, ssim


def dataset_psnr_ssim_fcl(dataloader, model, device):
    # init lists
    psnr = []
    ssim = []
    # loop over batches
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # evaluate fully connected layer
        net_output = model.evaluate_fcl(inputs)
        # compute PSNRs and concatenate
        psnr += batch_psnr(inputs, net_output)
        # compute SSIMs and concatenate
        ssim += batch_ssim(inputs, net_output)
    # convert
    psnr = np.array(psnr)
    ssim = np.array(ssim)
    return psnr, ssim


def psnr(I1, I2):
    """
    Computes the psnr between two images I1 and I2
    """
    d = np.amax(I1) - np.amin(I1)
    diff = np.square(I2 - I1)
    MSE = diff.sum() / I1.size
    Psnr = 10 * np.log(d**2 / MSE) / np.log(10)
    return Psnr


def psnr_(img1, img2, r=2):
    """
    Computes the psnr between two image with values expected in a given range

    Args:
        img1, img2 (np.ndarray): images
        r (float): image range

    Returns:
        Psnr (float): Peak signal-to-noise ratio

    """
    MSE = np.mean((img1 - img2) ** 2)
    Psnr = 10 * np.log(r**2 / MSE) / np.log(10)
    return Psnr


def psnr_torch(img_gt, img_rec, dim=(-2, -1), img_dyn=None):
    r"""
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.

    .. math::

        \text{PSNR} = 20 \, \log_{10} \left( \frac{\text{d}}{\sqrt{\text{MSE}}} \right), \\
        \text{MSE} = \frac{1}{L}\sum_{\ell=1}^L \|I_\ell - \tilde{I}_\ell\|^2_2,

    where :math:`d` is the image dynamic and :math:`\{I_\ell\}` (resp. :math:`\{\tilde{I}_\ell\}`) is the set of ground truth (resp. reconstructed) images.

    Args:
        :attr:`img_gt`: Tensor containing the *ground-truth* image.

        :attr:`img_rec`: Tensor containing the reconstructed image.

        :attr:`dim`: Dimensions where the squared error is computed. Defaults to the last two dimensions.

        :attr:`img_dyn`: Image dynamic range (e.g., 1.0 for normalized images, 255 for 8-bit images). When :attr:`img_dyn` is `None`, the dynamic range is computed from the ground-truth image.

    Returns:
        PSNR value.

    .. note::
        :attr:`psnr_torch(img_gt, img_rec)` is different from  :attr:`psnr_torch(img_rec, img_gt)`. The first expression assumes :attr:`img_gt` is the ground truth while the second assumes that this is :attr:`img_rec`. This leads to different dynamic ranges.

    Example 1: 10 images of size 64x64 with values in [0,1) corrupted with 5% noise
        >>> x = torch.rand(10,1,64,64)
        >>> n = x + 0.05*torch.randn(x.shape)
        >>> out = psnr_torch(x,n)
        >>> print(out.shape)
        torch.Size([10, 1])

    Example 2: 10 images of size 64x64 with values in [0,1) corrupted with 5% noise
        >>> psnr_torch(n,x)
        tensor(...)
        >>> psnr_torch(x,n)
        tensor(...)
        >>> psnr_torch(n,x,img_dyn=1.0)
        tensor(...)

    """
    mse = (img_gt - img_rec) ** 2
    mse = torch.mean(mse, dim=dim)

    if img_dyn is None:
        img_dyn = torch.amax(img_gt, dim=dim) - torch.amin(img_gt, dim=dim)

    return 10 * torch.log10(img_dyn**2 / mse)


def ssim(I1, I2):
    """
    Computes the ssim between two images I1 and I2
    """
    L = np.amax(I1) - np.amin(I1)
    mu1 = np.mean(I1)
    mu2 = np.mean(I2)
    s1 = np.std(I1)
    s2 = np.std(I2)
    s12 = np.mean(np.multiply((I1 - mu1), (I2 - mu2)))
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    result = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2)
    )
    return result


# def ssim_sk(x_gt, x, img_dyn=None):
#     """
#     SSIM from skimage

#     Args:
#         torch tensors

#     Returns:
#         torch tensor
#     """
#     if not isinstance(x, np.ndarray):
#         x = x.cpu().detach().numpy().squeeze()
#         x_gt = x_gt.cpu().detach().numpy().squeeze()
#     ssim_val = np.zeros(x.shape[0])
#     for i in range(x.shape[0]):
#         ssim_val[i] = skm.structural_similarity(x_gt[i], x[i], data_range=img_dyn)
#     return torch.tensor(ssim_val)


def batch_psnr_vid(input_batch, output_batch):
    list_psnr = []
    batch_size, seq_length, c, h, w = input_batch.shape
    input_batch = input_batch.reshape(batch_size * seq_length * c, 1, h, w)
    output_batch = output_batch.reshape(batch_size * seq_length * c, 1, h, w)
    for i in range(input_batch.shape[0]):
        img = input_batch[i, 0, :, :]
        img_out = output_batch[i, 0, :, :]
        img = img.cpu().detach().numpy()
        img_out = img_out.cpu().detach().numpy()
        list_psnr.append(psnr(img, img_out))
    return list_psnr


def batch_ssim_vid(input_batch, output_batch):
    list_ssim = []
    batch_size, seq_length, c, h, w = input_batch.shape
    input_batch = input_batch.reshape(batch_size * seq_length * c, 1, h, w)
    output_batch = output_batch.reshape(batch_size * seq_length * c, 1, h, w)
    for i in range(input_batch.shape[0]):
        img = input_batch[i, 0, :, :]
        img_out = output_batch[i, 0, :, :]
        img = img.cpu().detach().numpy()
        img_out = img_out.cpu().detach().numpy()
        list_ssim.append(ssim(img, img_out))
    return list_ssim


def compare_video_nets_supervised(net_list, testloader, device):
    psnr = [[] for i in range(len(net_list))]
    ssim = [[] for i in range(len(net_list))]
    for batch, (inputs, labels) in enumerate(testloader):
        [batch_size, seq_length, c, h, w] = inputs.shape
        print("Batch :{}/{}".format(batch + 1, len(testloader)))
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            for i in range(len(net_list)):
                outputs = net_list[i].evaluate(inputs)
                psnr[i] += batch_psnr_vid(labels, outputs)
                ssim[i] += batch_ssim_vid(labels, outputs)
    return psnr, ssim


def compare_nets_unsupervised(net_list, testloader, device):
    psnr = [[] for i in range(len(net_list))]
    ssim = [[] for i in range(len(net_list))]
    for batch, (inputs, labels) in enumerate(testloader):
        [batch_size, seq_length, c, h, w] = inputs.shape
        print("Batch :{}/{}".format(batch + 1, len(testloader)))
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            for i in range(len(net_list)):
                outputs = net_list[i].evaluate(inputs)
                psnr[i] += batch_psnr_vid(outputs, labels)
                ssim[i] += batch_ssim_vid(outputs, labels)
    return psnr, ssim


def print_mean_std(x, tag=""):
    print("{}psnr = {} +/- {}".format(tag, np.mean(x), np.std(x)))
