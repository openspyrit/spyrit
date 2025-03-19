import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

"""
    Modified by J Abascal https://github.com/cszn/DPIR/blob/master/models/network_unet.py
    June 2023
    Plug-and-Play Image Restoration with Deep Denoiser Prior
"""


class UNetRes(nn.Module):
    def __init__(
        self,
        in_nc=1,
        out_nc=1,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    ):
        super(UNetRes, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=False, mode="C")

        # downsample
        if downsample_mode == "avgpool":
            downsample_block = downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError(
                "downsample mode [{:s}] is not found".format(downsample_mode)
            )

        self.m_down1 = sequential(
            *[
                ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[0], nc[1], bias=False, mode="2"),
        )
        self.m_down2 = sequential(
            *[
                ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[1], nc[2], bias=False, mode="2"),
        )
        self.m_down3 = sequential(
            *[
                ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[2], nc[3], bias=False, mode="2"),
        )

        self.m_body = sequential(
            *[
                ResBlock(nc[3], nc[3], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ]
        )

        # upsample
        if upsample_mode == "upconv":
            upsample_block = upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )

        self.m_up3 = sequential(
            upsample_block(nc[3], nc[2], bias=False, mode="2"),
            *[
                ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
        )
        self.m_up2 = sequential(
            upsample_block(nc[2], nc[1], bias=False, mode="2"),
            *[
                ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
        )
        self.m_up1 = sequential(
            upsample_block(nc[1], nc[0], bias=False, mode="2"),
            *[
                ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
        )

        self.m_tail = conv(nc[0], out_nc, bias=False, mode="C")

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        return x


class DRUNet(UNetRes):
    r"""Plug-and-Play Image Restoration with Deep Denoiser Prior.
    DRUNet is a pretrained plug-and-play denoising network that has been pretrained for a wide range of noise levels.
    It admits the noise level as an input, so it does not require training.
    DRUNet was proposed in the work

    [ZhLZ21] K. Zhang et al., Plug-and-Play Image Restoration with Deep Denoiser Prior. In: IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10), 6360-6376, 2021.

    Original Code: https://github.com/cszn/DPIR/blob/master/models/network_unet.py

    Args:
        :attr:`noise_level` (float): noise level value in the range [0, 255].
        This is used to create a noise level map that is concatenated to the input images.

        :attr:`n_channels` (int): number of image channels

        :attr:`nc` (list of int): number of features

        :attr:`nb` (int): number of residual blocks

        :attr:`act_mode` (str): activation function mode

        :attr:`downsample_mode` (str): downsample mode

        :attr:`upsample_mode` (str): upsample mode

        :attr:`normalize` (bool): Normalize to (-1,1). Defaults to True.

    Input / Output:
        :attr:`x`: input images with shape (:math:`B`, :attr:`n_channels`, :math:`H`, :math:`W`).

        :attr:`output`: denoised images with shape (:math:`B`, :attr:`n_channels`, :math:`H`, :math:`W`).

        :attr:`normalize` (bool): Normalize to (-1,1) if True.

    Attributes:
        :attr:`noise_level` (tensor): noise level tensor with shape :math:`(1)`.

    .. note::
        :class:`~spyrit.external.drunet.DRUNet` has been tested only with :attr:`n_channels` =1 but
        :class:`~spyrit.external.drunet.UNetRes` can be used with :attr:`n_channels` >1.
    """

    def __init__(
        self,
        noise_level=5,
        n_channels=1,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
        normalize=True,
    ):
        super(DRUNet, self).__init__(
            n_channels + 1, n_channels, nc, nb, act_mode, downsample_mode, upsample_mode
        )
        self.register_buffer("noise_level", torch.FloatTensor([noise_level / 255.0]))
        self.normalize = normalize

    def forward(self, x):
        # Image domain denoising
        x = self.concat_noise_map(x)

        # Pass input images through the network
        x = super(DRUNet, self).forward(x)
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
        if self.normalize:
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
        self.noise_level = torch.FloatTensor([noise_level / 255.0]).to(
            self.noise_level.device
        )


# ----------------------------------------------
# Functions taken from basicblock.py
# https://github.com/cszn/DPIR/tree/master/models
# ----------------------------------------------


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        mode="CRC",
        negative_slope=0.2,
    ):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, "Only support in_channels==out_channels."
        if mode[0] in ["R", "L"]:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
            mode,
            negative_slope,
        )

    def forward(self, x):
        # res = self.res(x)
        return x + self.res(x)


"""
# --------------------------------------------
# Upsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# upsample_pixelshuffle
# upsample_upconv
# upsample_convtranspose
# --------------------------------------------
"""


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    up1 = conv(
        in_channels,
        out_channels * (int(mode[0]) ** 2),
        kernel_size,
        stride,
        padding,
        bias,
        mode="C" + mode,
        negative_slope=negative_slope,
    )
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR"
    if mode[0] == "2":
        uc = "UC"
    elif mode[0] == "3":
        uc = "uC"
    elif mode[0] == "4":
        uc = "vC"
    mode = mode.replace(mode[0], uc)
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode,
        negative_slope=negative_slope,
    )
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(
    in_channels=64,
    out_channels=3,
    kernel_size=2,
    stride=2,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
        negative_slope,
    )
    return up1


"""
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
"""


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(
    in_channels=64,
    out_channels=64,
    kernel_size=2,
    stride=2,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    down1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
        negative_slope,
    )
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "MC")
    pool = conv(
        kernel_size=kernel_size_pool,
        stride=stride_pool,
        mode=mode[0],
        negative_slope=negative_slope,
    )
    pool_tail = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode[1:],
        negative_slope=negative_slope,
    )
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "AC")
    pool = conv(
        kernel_size=kernel_size_pool,
        stride=stride_pool,
        mode=mode[0],
        negative_slope=negative_slope,
    )
    pool_tail = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode[1:],
        negative_slope=negative_slope,
    )
    return sequential(pool, pool_tail)


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="CBR",
    negative_slope=0.2,
):
    L = []
    for t in mode:
        if t == "C":
            L.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "T":
            L.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "B":
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == "I":
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        elif t == "r":
            L.append(nn.ReLU(inplace=False))
        elif t == "L":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == "l":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == "2":
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == "3":
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == "4":
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == "U":
            L.append(nn.Upsample(scale_factor=2, mode="nearest"))
        elif t == "u":
            L.append(nn.Upsample(scale_factor=3, mode="nearest"))
        elif t == "v":
            L.append(nn.Upsample(scale_factor=4, mode="nearest"))
        elif t == "M":
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == "A":
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError("Undefined type: ".format(t))
    return sequential(*L)


# --------------------------------------------
# Functions taken from utils/utils_image.py
# https://github.com/cszn/DPIR/tree/master/utils
# --------------------------------------------

"""
# =======================================
# numpy(single) <--->  numpy(unit)
# numpy(single) <--->  tensor
# numpy(unit)   <--->  tensor
# =======================================
"""


# --------------------------------
# numpy(single) <--->  numpy(unit)
# --------------------------------


def uint2single(img):
    return np.float32(img / 255.0)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.0).round())


def uint162single(img):
    return np.float32(img / 65535.0)


def single2uint16(img):
    return np.uint8((img.clip(0, 1) * 65535.0).round())


# --------------------------------
# numpy(unit) <--->  tensor
# uint (HxWxn_channels (RGB) or G)
# --------------------------------


# convert uint (HxWxn_channels) to 4-dimensional torch tensor
def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return (
        torch.from_numpy(np.ascontiguousarray(img))
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
    )


# convert uint (HxWxn_channels) to 3-dimensional torch tensor
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return (
        torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0)
    )


# convert torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


# --------------------------------
# numpy(single) <--->  tensor
# single (HxWxn_channels (RGB) or G)
# --------------------------------


# convert single (HxWxn_channels) to 4-dimensional torch tensor
def single2tensor4(img):
    return (
        torch.from_numpy(np.ascontiguousarray(img))
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
    )


def single2tensor5(img):
    return (
        torch.from_numpy(np.ascontiguousarray(img))
        .permute(2, 0, 1, 3)
        .float()
        .unsqueeze(0)
    )


def single32tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).unsqueeze(0)


def single42tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float()


# convert single (HxWxn_channels) to 3-dimensional torch tensor
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


# convert single (HxWx1, HxW) to 2-dimensional torch tensor
def single2tensor2(img):
    return torch.from_numpy(np.ascontiguousarray(img)).squeeze().float()


# convert torch tensor to single
def tensor2single(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img


def tensor2single3(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img
