from __future__ import print_function, division
from typing import Any
from pathlib import Path

import time
import torch
import torchvision
import numpy as np
from scipy.stats import rankdata

from spyrit.misc.disp import imagepanel, imagesc
import matplotlib.pyplot as plt

import spyrit.misc.walsh_hadamard as wh
import spyrit.core.torch as spytorch
import spyrit.misc.metrics as sm


# %% data loaders
def data_loaders_imagenet(
    train_root,
    val_root=None,
    img_size: int = 64,
    batch_size: int = 512,
    seed: int = 7,
    shuffle=False,
    get_size: str = "rcrop",
    normalize=True,
    **rcrop_kwargs,
):
    r"""
    Args:
        Both 'train_root' and 'val_root' need to have images in a subfolder

        :attr:`data_root`: path to image database, expected to contain an
        `/stl10_binary/` subfolder with the  `test*.bin`, `train*.bin`
        and `unlabeled_X.bin` files.

        :attr:`img_size`: image size

        :attr:`batch_size`: batch size

        :attr:`seed`: seed, only relevant for random transforms

        :attr:`shuffle`: True to shuffle train set (test set is not shuffled)

        :attr:`get_size`: specifies how images of size :attr:`img_size` are
        obtained

            - 'rcrop': random crop
            - 'resize': resize
            - 'ccrop': center crop

        :attr:`normalize`: The output of torchvision datasets are images in the range [0, 1]. Setting :attr:`normalize` to True sends them to the range [-1, 1]. When :attr:`normalize` is False, the images are left in the range [0, 1].

        :attr:`rcrop_kwargs`: Additional arguments for random crop

    .. note::

        The output of torchvision datasets are RGB images that are converted into grayscale images.
    """

    # random crop default keyword arguments
    default_kwargs = {
        "size": (img_size, img_size),
        "pad_if_needed": True,
        "padding_mode": "edge",
    }
    rcrop_kwargs = {**default_kwargs, **rcrop_kwargs}

    transform_normalize = (
        torchvision.transforms.Normalize([0.5], [0.5])
        if normalize
        else torch.nn.Identity()
    )

    if get_size == "rcrop":
        torch.manual_seed(seed)  # reproductibility of random transform
        #
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.functional.to_grayscale,
                torchvision.transforms.RandomCrop(
                    **rcrop_kwargs  # pad_if_needed=True, padding_mode="edge"
                ),
                torchvision.transforms.ToTensor(),
                transform_normalize,
            ]
        )

    elif get_size == "resize":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.functional.to_grayscale,
                torchvision.transforms.Resize(img_size),
                torchvision.transforms.CenterCrop(img_size),
                torchvision.transforms.ToTensor(),
                transform_normalize,
            ]
        )

    elif get_size == "ccrop":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.functional.to_grayscale,
                torchvision.transforms.CenterCrop(img_size),
                torchvision.transforms.ToTensor(),
                transform_normalize,
            ]
        )

    # train set
    trainset = torchvision.datasets.ImageFolder(root=train_root, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle
    )

    # validation set (if any)
    if val_root is not None:
        valset = torchvision.datasets.ImageFolder(root=val_root, transform=transform)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False
        )
    else:
        valloader = None

    dataloaders = {"train": trainloader, "val": valloader}

    return dataloaders


def data_loaders_ImageNet(
    train_root,
    val_root=None,
    img_size=64,
    batch_size=512,
    seed=7,
    shuffle=False,
    normalize=True,
):
    """
    Args:
        Both 'train_root' and 'val_root' need to have images in a subfolder
        shuffle=True to shuffle train set only (test set not shuffled)

    The output of torchvision datasets are PILImage images in the range [0, 1].
    We transform them to Tensors in the range [-1, 1]. Also RGB images are
    converted into grayscale images.
    """

    torch.manual_seed(seed)  # reproductibility of random crop
    #
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.functional.to_grayscale,
            torchvision.transforms.RandomCrop(
                size=(img_size, img_size), pad_if_needed=True, padding_mode="edge"
            ),
            torchvision.transforms.ToTensor(),
            (
                torchvision.transforms.Normalize([0.5], [0.5])
                if normalize
                else torch.nn.Identity()
            ),
        ]
    )

    # train set
    trainset = torchvision.datasets.ImageFolder(root=train_root, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle
    )

    # validation set (if any)
    if val_root is not None:
        valset = torchvision.datasets.ImageFolder(root=val_root, transform=transform)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False
        )
    else:
        valloader = None

    dataloaders = {"train": trainloader, "val": valloader}

    return dataloaders


class CenterCrop:
    """
    Args:
        img_size=int, image size

    Center crop if image not square in order to ensure that all images have same size
    """

    def __init__(self, img_size):
        self.img_size = img_size
        self.centerCrop = torchvision.transforms.CenterCrop(img_size)

    def __call__(self, inputs, *args: Any, **kwds: Any):
        # Center crop if not square
        img_shape = inputs.size
        if img_shape[0] != img_shape[1]:
            return self.centerCrop(inputs)
        else:
            return inputs


def transform_gray_norm(img_size, normalize=True):
    """
    Args:
        img_size=int, image size

    Create torchvision transform for natural images (stl10, imagenet):
    convert them to grayscale, then to tensor, and normalize between [-1, 1]
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.functional.to_grayscale,
            torchvision.transforms.Resize(
                img_size,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            ),
            # torchvision.transforms.CenterCrop(img_size),
            CenterCrop(img_size),
            torchvision.transforms.ToTensor(),
            (
                torchvision.transforms.Normalize([0.5], [0.5])
                if normalize
                else torch.nn.Identity()
            ),
        ]
    )
    return transform


def data_loaders_stl10(
    data_root,
    img_size=64,
    batch_size=512,
    seed=7,
    shuffle=False,
    download=True,
    normalize=True,
):
    """
    Args:
        shuffle=True to shuffle train set only (test set not shuffled)

    The output of torchvision datasets are PILImage images in the range [0, 1].
    We transform them to Tensors in the range [-1, 1]. Also RGB images are
    converted into grayscale images.

    """
    transform = transform_gray_norm(img_size, normalize=normalize)

    trainset = torchvision.datasets.STL10(
        root=data_root, split="train+unlabeled", download=download, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle
    )

    testset = torchvision.datasets.STL10(
        root=data_root, split="test", download=download, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    dataloaders = {"train": trainloader, "val": testloader}

    return dataloaders


# %% Metrics
def stat_psnr(
    model,
    dataloader,
    device,
    n_loop=1,
    num_batchs=None,
    img_dyn=None,
):
    """
    nloop > 1 is relevant for dataloaders with random crops such as that
    provided by data_loaders_ImageNet

    """
    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape

    if num_batchs is None:
        tot_num = len(dataloader) * b
    else:
        tot_num = num_batchs * b

    # just in case...
    model.eval()

    # Init
    n = 0
    mean = torch.tensor([0.0], device=device)

    # Pass 1: Compute Mean
    for i in range(n_loop):
        for jj, (inputs, _) in enumerate(dataloader):
            if num_batchs is not None and jj >= num_batchs:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)

            psnr_batch = sm.psnr_torch(inputs, outputs, img_dyn=img_dyn)

            # remove infinite values and NaNs
            valid_mask = torch.isfinite(psnr_batch)
            psnr_batch = psnr_batch[valid_mask]

            mean += torch.sum(psnr_batch)

            # print
            n = n + inputs.shape[0]
            print(f"Mean:  {n} / (less than) {tot_num*n_loop} images", end="\n")
        # print("", end="\n")

    mean = mean / n
    mean = torch.squeeze(mean)

    # Pass 2: Variance
    n = 0
    var = torch.tensor([0.0], device=device)
    for i in range(n_loop):
        for jj, (inputs, _) in enumerate(dataloader):
            if num_batchs is not None and jj >= num_batchs:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            psnr_batch = sm.psnr_torch(inputs, outputs, img_dyn=img_dyn)
            psnr_batch = (psnr_batch - mean) ** 2

            # remove infinite values and NaNs
            valid_mask = torch.isfinite(psnr_batch)
            psnr_batch = psnr_batch[valid_mask]

            var += torch.sum(psnr_batch)

            # print
            n = n + inputs.shape[0]
            print(f"var:  {n} / (less than) {tot_num*n_loop} images", end="\n")
        # print("", end="\n")

    var = var / (n - 1)
    var = torch.squeeze(var)

    return mean, var


def stat_ssim(
    model,
    dataloader,
    device,
    n_loop=1,
    num_batchs=None,
    img_dyn=None,
):
    """
    nloop > 1 is relevant for dataloaders with random crops such as that
    provided by data_loaders_ImageNet

    Returns:
        torch.tensor on cpu

        torch.tensor on cpu

    """
    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape
    tot_num = len(dataloader) * b

    # just in case...
    model.eval()

    # Init
    n = 0
    mean = torch.tensor([0.0], device=device)

    # Pass 1: Compute Mean
    for i in range(n_loop):
        for jj, (inputs, _) in enumerate(dataloader):
            if num_batchs is not None and jj >= num_batchs:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)

            batch = sm.ssim_sk(inputs, outputs, img_dyn=img_dyn)

            # remove infinite values and NaNs
            valid_mask = torch.isfinite(batch)
            batch = batch[valid_mask]

            mean += torch.sum(batch)

            # print
            n = n + inputs.shape[0]
            print(f"Mean:  {n} / (less than) {tot_num*n_loop} images", end="\n")
        # print("", end="\n")

    mean = mean / n
    mean = torch.squeeze(mean)
    mean = mean.to(device="cpu")

    # Pass 2: Variance
    n = 0
    var = torch.tensor([0.0], device=device)
    for i in range(n_loop):
        for jj, (inputs, _) in enumerate(dataloader):
            if num_batchs is not None and jj >= num_batchs:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            batch = sm.ssim_sk(inputs, outputs, img_dyn=img_dyn)
            batch = (batch - mean) ** 2

            # remove infinite values and NaNs
            valid_mask = torch.isfinite(batch)
            batch = batch[valid_mask]

            var += torch.sum(batch)

            # print
            n = n + inputs.shape[0]
            print(f"var:  {n} / (less than) {tot_num*n_loop} images", end="\n")
        # print("", end="\n")

    var = var / (n - 1)
    var = torch.squeeze(var)

    return mean, var


# %% Covariance in Walsh Hadamard domain


def stat_walsh_ImageNet(
    stat_root=Path("./stats/"),
    data_root=Path("./data/ILSVRC2012_img_test_v10102019/"),
    img_size=128,
    batch_size=256,
    n_loop=1,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    r"""
    Args:
        :attr:`data_root` needs to have all images in a subfolder

    Example:
        >>> from pathlib import Path
        >>> from spyrit.misc.statistics import stat_walsh_ImageNet
        >>> data_root =  Path('../data/ILSVRC2012_v10102019')
        >>> stat_root =  Path('../stat/ILSVRC2012_v10102019')
        >>> stat_walsh_ImageNet(stat_root = stat_root, data_root = data_root, img_size = 32, batch_size = 1024) # doctest: +SKIP

    """

    dataloaders = data_loaders_ImageNet(
        data_root, img_size=img_size, batch_size=batch_size, seed=7
    )

    # Walsh ordered transforms
    time_start = time.perf_counter()
    stat_walsh(dataloaders["train"], device, stat_root, n_loop)
    time_elapsed = time.perf_counter() - time_start

    print(f"Computed in {time_elapsed} seconds")


def stat_walsh_stl10(
    stat_root=Path("./stats/"),
    data_root=Path("./data/"),
    img_size=64,
    batch_size=1024,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    """
    Args:
        :attr:`data_root` is expected to contain an 'stl10_binary' subfolder with the
        test*.bin, train*.bin and unlabeled_X.bin files.

    Example:
        >>> data_root =  Path('../datasets/')
        >>> stat_root =  Path('../stat/stl10')
        >>> from spyrit.misc.statistics import stat_walsh_stl10
        >>> stat_walsh_stl10(stat_root = stat_root, data_root = data_root) # doctest: +SKIP

    """
    dataloaders = data_loaders_stl10(
        data_root, img_size=img_size, batch_size=batch_size, seed=7
    )
    # Walsh ordered transforms
    time_start = time.perf_counter()
    stat_walsh(dataloaders["train"], device, stat_root)
    time_elapsed = time.perf_counter() - time_start

    print(f"Computed in {time_elapsed} seconds")


def stat_walsh(dataloader, device, root, n_loop=1):
    """
    nloop > 1 is relevant for dataloaders with random crops such as that
    provided by data_loaders_ImageNet

    """
    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (_, _, nx, ny) = inputs.shape

    # --------------------------------------------------------------------------
    # 1. Mean
    # --------------------------------------------------------------------------
    mean = mean_walsh(dataloader, device, n_loop=n_loop)

    # Save
    if n_loop == 1:
        path = root / Path("Average_{}x{}".format(nx, ny) + ".npy")
    else:
        path = root / Path("Average_{}_{}x{}".format(n_loop, nx, ny) + ".npy")

    if not root.exists():
        root.mkdir()
    np.save(path, mean.cpu().detach().numpy())
    # --------------------------------------------------------------------------
    # 2. Covariance
    # -------------------------------------------------------------------------
    cov = cov_walsh(dataloader, mean, device, n_loop=n_loop)

    # Save
    if n_loop == 1:
        path = root / Path("Cov_{}x{}".format(nx, ny) + ".npy")
    else:
        path = root / Path("Cov_{}_{}x{}".format(n_loop, nx, ny) + ".npy")

    if not root.exists():
        root.mkdir()
    np.save(path, cov.cpu().detach().numpy())

    return mean, cov


def mean_walsh(dataloader, device, n_loop=1):
    """
    nloop > 1 is relevant for dataloaders with random crops such as that
    provided by data_loaders_ImageNet

    """

    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape
    tot_num = len(dataloader) * b

    # Init
    n = 0
    # H = wh.walsh_matrix(nx).astype(np.float32, copy=False)
    mean = torch.zeros((nx, ny), dtype=torch.float32)

    # Send to device (e.g., cuda)
    mean = mean.to(device)
    # H = torch.from_numpy(H).to(device)

    # Compute Mean
    # Accumulate sum over all images in dataset
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            trans = spytorch.fwht_2d(inputs, True)
            mean = mean.add(torch.sum(trans, 0))
            # print
            n = n + inputs.shape[0]
            print(f"Mean:  {n} / (less than) {tot_num*n_loop} images", end="\n")
            # test
            # print(f' | {inputs[53,0,33,49]}', end='\n')
        print("", end="\n")

    # Normalize
    mean = mean / n
    mean = torch.squeeze(mean)

    return mean


def cov_walsh(dataloader, mean, device, n_loop=1):
    """
    nloop > 1 is relevant for dataloaders with random crops such as that
    provided by data_loaders_ImageNet

    """

    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape
    tot_num = len(dataloader) * b

    # H = wh.walsh_matrix(nx).astype(np.float32, copy=False)
    # H = torch.from_numpy(H).to(device)

    # Covariance --------------------------------------------------------------
    # Init
    n = 0
    cov = torch.zeros((nx * ny, nx * ny), dtype=torch.float32)
    cov = cov.to(device)

    # Accumulate (im - mu)*(im - mu)^T over all images in dataset
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            trans = spytorch.fwht_2d(inputs, True)
            trans = trans - mean.repeat(inputs.shape[0], 1, 1, 1)
            trans = trans.reshape(inputs.shape[0], nx * ny, 1)
            cov = torch.addbmm(cov, trans, trans.reshape(inputs.shape[0], 1, nx * ny))
            # print
            n += inputs.shape[0]
            print(f"Cov:  {n} / (less than) {tot_num*n_loop} images", end="\n")
            # test
            # print(f' | {inputs[53,0,33,49]}', end='\n')
        print("", end="\n")

    # Normalize
    cov = cov / (n - 1)

    return cov


def stat_fwalsh_S(dataloader, device, root):  # NOT validated!
    # Get dimensions and estimate total number of images in the dataset
    inputs, classes = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape
    tot_num = len(dataloader) * b

    # 1. Mean

    # Init
    n = 0
    mean = torch.zeros((nx, ny), dtype=torch.float32)

    # Send to device (e.g., cuda)
    mean = mean.to(device)
    ind = wh.sequency_perm_ind(nx * ny)

    # Accumulate sum over all images in dataset
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        trans = wh.fwalsh2_S_torch(inputs, ind)
        mean = mean.add(torch.sum(trans, 0))
        # print
        n = n + inputs.shape[0]
        print(f"Mean:  {n} / (less than) {tot_num} images", end="\n")
    print("", end="\n")

    # Normalize
    mean = mean / n
    mean = torch.squeeze(mean)
    # torch.save(mean, root+'Average_{}x{}'.format(nx,ny)+'.pth')

    path = root / Path("Average_{}x{}".format(nx, ny) + ".npy")
    if not root.exists():
        root.mkdir()
    np.save(path, mean.cpu().detach().numpy())

    # 2. Covariance

    # Init
    n = 0
    cov = torch.zeros((nx * ny, nx * ny), dtype=torch.float32)
    cov = cov.to(device)

    # Accumulate (im - mu)*(im - mu)^T over all images in dataset
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        trans = wh.fwalsh2_S_torch(inputs, ind)
        trans = trans - mean.repeat(inputs.shape[0], 1, 1, 1)
        trans = trans.reshape(inputs.shape[0], nx * ny, 1)
        cov = torch.addbmm(cov, trans, trans.reshape(inputs.shape[0], 1, nx * ny))
        # print
        n += inputs.shape[0]
        print(f"Cov:  {n} / (less than) {tot_num} images", end="\n")
    print("", end="\n")

    # Normalize
    cov = cov / (n - 1)
    # torch.save(cov, root+'Cov_{}x{}'.format(nx,ny)+'.pth') # todo?

    path = root / Path("Cov_{}x{}".format(nx, ny) + ".npy")
    if not root.exists():
        root.mkdir()
    np.save(path, cov.cpu().detach().numpy())

    return mean, cov


# todo: rewrite in a fashion similar to stat_walsh_stl10
def stat_fwalsh_S_stl10(
    stat_root=Path("./stats/"), data_root=Path("./data/"), img_size=64, batch_size=1024
):
    """Fast Walsh S-transform of X in "2D"

    Args:
        :attr:`X` (torch.tensor): input image with shape `(*, n, n)`. `n`**2
        should be a power of two.

    Returns:
        torch.tensor: S-transformed signal with shape `(*, n, n)`

    Examples:
        >>> import spyrit.misc.statistics as st
        >>> st.stat_fwalsh_S_stl10() # doctest: +SKIP

    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)  # for reproductibility

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.functional.to_grayscale,
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    )

    trainset = torchvision.datasets.STL10(
        root=data_root, split="train+unlabeled", download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    testset = torchvision.datasets.STL10(
        root=data_root, split="test", download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    dataloaders = {"train": trainloader, "val": testloader}

    # Walsh ordered transforms
    time_start = time.perf_counter()
    stat_fwalsh_S(dataloaders["train"], device, stat_root)
    time_elapsed = time.perf_counter() - time_start
    print(time_elapsed)


# %% Covariance in image-domain


def stat_imagenet(
    stat_root=Path("./stats/"),
    data_root=Path("./data/ILSVRC2012_img_test_v10102019/"),
    img_size: int = 64,
    batch_size: int = 1024,
    get_size: str = "resize",
    n_loop: int = 1,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    normalize=True,
    ext="npy",
    **rcrop_kwargs,
):
    """
    Args:
        :attr:`stat_root`: path to the folder where the mean and covariance
        matrices are saved

        :attr:`data_root`: path to image database.  :attr:`data_root` needs to
        have all images in a subfolder

        :attr:`img_size`: image size

        :attr:`batch_size`: batch size

        :attr:`get_size`: specifies how images of size :attr:`img_size` are
        obtained (see :mod:`~spyrit.misc.statistics.data_loaders_imagenet`)
            - 'original': random crop with padding
            - 'resize': resize
            - 'ccrop': center crop
            - 'rcrop': random crop

        :attr:`n_loop` (int, optional): Number of loops across image database. Defaults to 1. nloop > 1 is only relevant for dataloaders with random transforms (e.g., 'original' or 'rcrop' resizing)

        :attr:`normalize`: Torchvision datasets are images in the range [0, 1]. Setting :attr:`normalize` to True sends them to the range [-1, 1]. When :attr:`normalize` is False, the images are left in the range [0, 1].

        :attr:`ext` (string): Extension of saved files:

            - 'npy' for numpy (default),
            - 'pt' for pytorch,
            - do not save files otherwise.

        :attr:`rcrop_kwargs`: Aditional argument for random crop


    Example:
        >>> data_root =  Path('../data/ILSVRC2012_img_test_v10102019/')
        >>> stat_root =  Path('../stat/ILSVRC2012_img_test_v10102019')
        >>> from spyrit.misc.statistics import stat_imagenet
        >>> stat_imagenet(stat_root = stat_root, data_root = data_root) # doctest: +SKIP

    """
    dataloaders = data_loaders_imagenet(
        data_root,
        img_size=img_size,
        batch_size=batch_size,
        seed=7,
        get_size=get_size,
        normalize=normalize,
        **rcrop_kwargs,
    )

    dataloader = dataloaders["train"]

    # Compute mean and covariance
    time_start = time.perf_counter()

    mean, cov = stat_2(dataloader, device, stat_root, n_loop, ext)

    if not stat_root.exists():
        stat_root.mkdir(parents=True, exist_ok=True)

    time_elapsed = time.perf_counter() - time_start

    print(f"Computed in {time_elapsed} seconds")

    # plot and save a few images
    inputs, _ = next(iter(dataloader))
    imagepanel(
        inputs[0, 0, :, :], inputs[1, 0, :, :], inputs[2, 0, :, :], inputs[3, 0, :, :]
    )
    plt.savefig(stat_root / f"images_{img_size}x{img_size}.png")

    # plot and save a few covariances
    i1 = int(img_size * img_size / 10)
    i2 = int(img_size * img_size / 5)
    i3 = int(img_size * img_size // 2 + img_size // 2)

    im1 = cov[i1, :].reshape(img_size, img_size).cpu()
    im2 = cov[i2, :].reshape(img_size, img_size).cpu()
    im3 = cov[i3, :].reshape(img_size, img_size).cpu()
    im4 = torch.diag(cov).reshape(img_size, img_size).cpu()

    imagepanel(
        im1,
        im2,
        im3,
        im4,
        "",
        f"cov ({i1}-th row)",
        f"cov ({i2}-th row)",
        f"cov ({i3}-th row)",
        "var (diagonal)",
    )

    plt.savefig(stat_root / f"cov_{img_size}x{img_size}.png")

    # plot and save mean
    imagesc(mean.detach().cpu(), "mean image")
    plt.savefig(stat_root / f"mean_{img_size}x{img_size}.png")


def stat_2(dataloader, device, root, n_loop: int = 1, ext="npy"):
    """
    Computes and saves 2D mean image and covariance matrix of an image database

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader. The fetch data are Torch tensors with shape `(B,C,N,N)`.

        device (torch.device): Device.

        root (file, str, or pathlib.Path): Path where the covariance and mean are saved.

        n_loop (int, optional): Number of loops across image database. Defaults to 1. nloop > 1 is relevant for dataloaders with random transforms.

        ext (string): Extension of saved files:

            - 'npy' for numpy (default),
            - 'pt' for pytorch,
            - do not save files otherwise.

    Returns:
        mean (np.array): Mean image with shape `(N, N)`.

        cov (np.array): Covariance matrix with shape `(N*N, N*N)`.

    """
    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (_, _, nx, ny) = inputs.shape

    # --------------------------------------------------------------------------
    # 1. Mean
    # --------------------------------------------------------------------------
    mean = mean_2(dataloader, device, n_loop=n_loop)

    # Save
    if n_loop == 1:
        path = root / Path("Average_im2_{}x{}".format(nx, ny) + "." + ext)
    else:
        path = root / Path("Average_im2_{}_{}x{}".format(n_loop, nx, ny) + "." + ext)

    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    if ext == "npy":
        np.save(path, mean.cpu().detach().numpy())
    elif ext == "pt":
        torch.save(mean.cpu().detach(), path)

    # --------------------------------------------------------------------------
    # 2. Covariance
    # -------------------------------------------------------------------------
    cov = cov_2(dataloader, mean, device, n_loop=n_loop)

    # Save
    if n_loop == 1:
        path = root / Path("Cov_im2_{}x{}".format(nx, ny) + "." + ext)
    else:
        path = root / Path("Cov_im2_{}_{}x{}".format(n_loop, nx, ny) + "." + ext)

    if ext == "npy":
        np.save(path, cov.cpu().detach().numpy())
    elif ext == "pt":
        torch.save(cov.cpu().detach(), path)

    return mean, cov


def mean_2(
    dataloader: torch.utils.data.DataLoader, device: torch.device, n_loop: int = 1
):
    """
    Computes 2D mean image computed across batches and channels

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader. The fetch data are Torch tensors with shape `(B,C,N,N)`.

        device (torch.device): Device.

        n_loop (int, optional): Number of loops across image database. Defaults to 1. nloop > 1 is relevant for dataloaders with random transforms.

    Returns:
        mean (np.array): Mean image with shape `(N, N)`.

    """

    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, _, nx, ny) = inputs.shape
    tot_num = len(dataloader) * b

    # Init
    n = 0
    mean = torch.zeros((nx, ny), dtype=torch.float32, device=device)

    # Compute Mean
    # Accumulate sum over all images in dataset
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            batched_sum = torch.sum(inputs, (0, 1))
            mean += batched_sum
            # print
            n = n + inputs.shape[0]
            print(f"Mean:  {n} / (less than) {tot_num*n_loop} images", end="\n")
            # test
            # print(f' | {inputs[53,0,33,49]}', end='\n')
        print("", end="\n")

    # Normalize
    mean = mean / n
    mean = torch.squeeze(mean)

    return mean


def cov_2(
    dataloader: torch.utils.data.DataLoader,
    mean: np.array,
    device: torch.device,
    n_loop: int = 1,
):
    """
    Computes 2D covariance matrix computed across batches and channels.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader. The fetch data are Torch tensors with shape `(B,C,N,N)`.

        mean (np.array): Mean image with shape `(N, N)`.

        device (torch.device): Device.

        n_loop (int, optional): Number of loops across image database. Defaults to 1. nloop > 1 is relevant for dataloaders with random transforms.

    Returns:
        cov (np.array): Covariance matrix with shape `(N*N, N*N)`.

    """

    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, c, nx, ny) = inputs.shape
    tot_num = len(dataloader) * b

    # Covariance --------------------------------------------------------------
    # Init
    n = 0
    cov = torch.zeros((nx * ny, nx * ny), dtype=torch.float32, device=device)

    # Accumulate (im - mu)*(im - mu)^T over all images in dataset
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            inputs -= mean
            inputs = inputs.view(inputs.shape[0], nx * ny, 1)

            cov = torch.addbmm(cov, inputs, inputs.mT)
            # cov += torch.sum(inputs @ inputs.mT, 0) # slower

            # print
            n += inputs.shape[0]
            print(f"Cov:  {n} / (less than) {tot_num*n_loop} images", end="\n")
            # test
            # print(f' | {inputs[53,0,33,49]}', end='\n')
        print("", end="\n")

    # Normalize
    cov = cov / (n - 1)

    return cov


def stat_1(dataloader, device, root, n_loop=1):
    """
    1D mean and covariance matrix of an image database.

    The statistics are computed across batches, channels, and image rows.

    nloop > 1 is relevant for dataloaders with random crops such as that
    provided by data_loaders_ImageNet

    """
    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (_, _, nx, ny) = inputs.shape

    # --------------------------------------------------------------------------
    # 1. Mean
    # --------------------------------------------------------------------------
    mean = mean_1(dataloader, device, n_loop=n_loop)

    # Save
    if n_loop == 1:
        path = root / Path("Average_1_{}x{}".format(nx, ny) + ".npy")
    else:
        path = root / Path("Average_1_{}_{}x{}".format(n_loop, nx, ny) + ".npy")

    if not root.exists():
        root.mkdir()
    np.save(path, mean.cpu().detach().numpy())
    # --------------------------------------------------------------------------
    # 2. Covariance
    # -------------------------------------------------------------------------
    cov = cov_1(dataloader, mean, device, n_loop=n_loop)

    # Save
    if n_loop == 1:
        path = root / Path("Cov_1_{}x{}".format(nx, ny) + ".npy")
    else:
        path = root / Path("Cov_1_{}_{}x{}".format(n_loop, nx, ny) + ".npy")

    if not root.exists():
        root.mkdir()
    np.save(path, cov.cpu().detach().numpy())

    return mean, cov


def mean_1(dataloader, device, n_loop=1):
    """
    The mean is computed across batches, channels, and image rows.

    nloop > 1 is relevant for dataloaders with random crops such as that
    provided by data_loaders_ImageNet

    """

    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, _, nx, ny) = inputs.shape
    tot_num = len(dataloader) * b

    # Init
    n = 0
    mean = torch.zeros(ny, dtype=torch.float32)

    # Send to device (e.g., cuda)
    mean = mean.to(device)

    # Compute Mean
    # Accumulate sum over all the image columns in the database
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            inputs = inputs.view(-1, nx, ny)
            #
            mean = mean.add(inputs.sum((0, 1)))  # Accumulate over images and rows

            # print
            n = n + inputs.shape[0]
            print(f"Mean:  {n} / (less than) {tot_num*n_loop} images", end="\n")
        print("", end="\n")

    # Normalize
    mean = mean / n / nx
    mean = torch.squeeze(mean)

    return mean


def cov_1(dataloader, mean, device, n_loop=1):
    """
    The covariance is computed across batches, channels, and image rows.

    nloop > 1 is relevant for dataloaders with random crops such as that
    provided by data_loaders_ImageNet

    """

    # Get dimensions and estimate total number of images in the dataset
    inputs, _ = next(iter(dataloader))
    (b, _, nx, ny) = inputs.shape
    tot_num = len(dataloader) * b

    H = wh.walsh_matrix(ny).astype(np.float32, copy=False)
    H = torch.from_numpy(H).to(device)

    # Covariance --------------------------------------------------------------
    # Init
    n = 0
    cov = torch.zeros((ny, ny), dtype=torch.float32)
    cov = cov.to(device)

    # Accumulate (im - mu)^T*(im - mu) over all images in dataset.
    # Each row is assumed to be an observation, so we have to transpose
    for i in range(n_loop):
        torch.manual_seed(i)
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            (b, c, _, _) = inputs.shape
            inputs = inputs.view(-1, nx, ny)  # shape (b*c, nx, ny)
            #
            dev = (inputs - mean).mT
            cov = torch.addbmm(cov, dev, dev.mT)
            # print
            n += inputs.shape[0]
            print(f"Cov:  {n} / (less than) {tot_num*n_loop} images", end="\n")
        print("", end="\n")

    # Normalize
    cov = cov / (n - 1) / (nx - 1)

    return cov


# %%


def Cov2Var(Cov: np.ndarray, out_shape=None):
    r"""
    Extracts Variance Matrix from Covariance Matrix.

    The Variance matrix is extracted from the diagonal of the Covariance matrix.

    Args:
        Cov (np.array): Covariance matrix of shape :math:`(N_x, N_x)`.

        out_shape (tuple, optional): Shape of the output variance matrix. If
        `None`, :math:`N_x` must be a perfect square and the output is a square
        matrix whose shape is :math:`(\sqrt{N_x}, \sqrt{N_x})`. Default is `None`.

    Raises:
        ValueError: If the input matrix is not square.

        ValueError: If the output shape is not valid.

    Returns:
        np.array: Variance matrix of shape :math:`(\sqrt{N_x}, \sqrt{N_x})` or
        :math:`out_shape` if provided.
    """
    row, col = Cov.shape
    # check Cov is square
    if row != col:
        raise ValueError("Covariance matrix must be a square matrix")

    if out_shape is None:
        out_shape = (int(np.sqrt(row)), int(np.sqrt(col)))
    if out_shape[0] * out_shape[1] != row:
        raise ValueError(
            f"Invalid output shape, got {out_shape} with "
            + f"{out_shape[0]}*{out_shape[1]} != {row}"
        )
    # copy is necessary (see np documentation about diagonal)
    return np.diagonal(Cov).copy().reshape(out_shape)


def img2mask(Ord, M):
    """
    Returns subsampling mask from order matrix
    """
    (nx, ny) = Ord.shape
    msk = np.ones((nx, ny))
    ranked_data = np.reshape(rankdata(-Ord, method="ordinal"), (nx, ny))
    msk[np.absolute(ranked_data) > M] = 0
    return msk
