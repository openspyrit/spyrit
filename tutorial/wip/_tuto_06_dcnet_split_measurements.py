r"""
=========================================
06. Denoised Completion Network (DCNet)
=========================================
.. _tuto_dcnet_split_measurements:
This tutorial shows how to perform image reconstruction using the denoised
completion network (DCNet) with a trainable image denoiser. In the next
tutorial, we will plug a denoiser into a DCNet, which requires no training.

.. figure:: ../fig/tuto3.png
   :width: 600
   :align: center
   :alt: Reconstruction and neural network denoising architecture sketch using split measurements
"""

######################################################################
# .. note::
#   As in the previous tutorials, we consider a split Hadamard operator and
#   measurements corrupted by Poisson noise (see :ref:`Tutorial 5 <tuto_acquisition_split_measurements>`).

# %%
# Load a batch of images
# =========================================

######################################################################
# Update search path

# sphinx_gallery_thumbnail_path = 'fig/tuto6.png'
if False:

    import os

    import torch
    import torchvision
    import matplotlib.pyplot as plt

    import spyrit.core.torch as spytorch
    from spyrit.misc.disp import imagesc
    from spyrit.misc.statistics import transform_gray_norm

    spyritPath = os.getcwd()
    imgs_path = os.path.join(spyritPath, "images/")

    ######################################################################
    # Images :math:`x` for training neural networks expect values in [-1,1]. The images are normalized and resized using the :func:`transform_gray_norm` function.

    h = 64  # image is resized to h x h
    transform = transform_gray_norm(img_size=h)

    ######################################################################
    # Create a data loader from some dataset (images must be in the folder `images/test/`)

    dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=7)

    x, _ = next(iter(dataloader))
    print(f"Shape of input images: {x.shape}")

    ######################################################################
    # Select the `i`-th image in the batch
    i = 1  # Image index (modify to change the image)
    x = x[i : i + 1, :, :, :]
    x = x.detach().clone()
    print(f"Shape of selected image: {x.shape}")
    b, c, h, w = x.shape

    ######################################################################
    # Plot the selected image

    imagesc(x[0, 0, :, :], r"$x$ in [-1, 1]")

    # %%
    # Forward operators for split measurements
    # =========================================

    ######################################################################
    # We consider noisy measurements obtained from a split Hadamard operator, and a subsampling strategy that retaines the coefficients with the largest variance (for more details, refer to :ref:`Tutorial 5 <tuto_acquisition_split_measurements>`).

    ######################################################################
    # First, we download the covariance matrix from our warehouse.

    import girder_client
    from spyrit.misc.load_data import download_girder

    # Get covariance matrix
    url = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
    dataId = "672207cbf03a54733161e95d"
    data_folder = "./stat/"
    cov_name = "Cov_64x64.pt"
    # download
    file_abs_path = download_girder(url, dataId, data_folder, cov_name)

    try:
        Cov = torch.load(file_abs_path, weights_only=True)
        print(f"Cov matrix {cov_name} loaded")
    except:
        Cov = torch.eye(h * h)
        print(f"Cov matrix {cov_name} not found! Set to the identity")

    ######################################################################
    # We define the measurement, noise and preprocessing operators and then simulate
    # a measurement vector corrupted by Poisson noise. As in the previous tutorials,
    # we simulate an accelerated acquisition by subsampling the measurement matrix
    # by retaining only the first rows of a Hadamard matrix that is permuted looking
    # at the diagonal of the covariance matrix.

    from spyrit.core.meas import HadamSplit
    from spyrit.core.noise import Poisson
    from spyrit.core.prep import SplitPoisson

    # Measurement parameters
    M = h**2 // 4  # Number of measurements (here, 1/4 of the pixels)
    alpha = 100.0  # number of photons

    # Measurement and noise operators
    Ord = spytorch.Cov2Var(Cov)
    meas_op = HadamSplit(M, h, Ord)
    noise_op = Poisson(meas_op, alpha)
    prep_op = SplitPoisson(alpha, meas_op)

    print(f"Shape of image: {x.shape}")

    # Measurements
    y = noise_op(x)  # a noisy measurement vector
    m = prep_op(y)  # preprocessed measurement vector

    m_plot = spytorch.meas2img(m, Ord)
    imagesc(m_plot[0, 0, :, :], r"Measurements $m$")

    # %%
    # Pseudo inverse solution
    # =========================================

    ######################################################################
    # We compute the pseudo inverse solution using :class:`spyrit.core.recon.PinvNet` class as in the previous tutorial.

    # Instantiate a PinvNet (with no denoising by default)
    from spyrit.core.recon import PinvNet

    pinvnet = PinvNet(noise_op, prep_op)

    # Use GPU, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    pinvnet = pinvnet.to(device)
    y = y.to(device)

    # Reconstruction
    with torch.no_grad():
        z_invnet = pinvnet.reconstruct(y)

    # %%
    # Denoised completion network (DCNet)
    # =========================================

    ######################################################################
    # .. image:: ../fig/dcnet.png
    #    :width: 400
    #    :align: center
    #    :alt: Sketch of the DCNet architecture

    ######################################################################
    # The DCNet is based on four sequential steps:
    #
    # i) Denoising in the measurement domain.
    #
    # ii) Estimation of the missing measurements from the denoised ones.
    #
    # iii) Image-domain mapping.
    #
    # iv) (Learned) Denoising in the image domain.
    #
    # Typically, only the last step involves learnable parameters.

    # %%
    # Denoised completion
    # =========================================

    ######################################################################
    # The first three steps implement denoised completion, which corresponds to Tikhonov regularization. Considering linear measurements :math:`y = Hx`, where :math:`H` is the  measurement matrix and :math:`x` is the unknown image, it estimates :math:`x` from :math:`y` by minimizing
    #
    # .. math::
    #    \| y - Hx \|^2_{\Sigma^{-1}_\alpha} + \|x\|^2_{\Sigma^{-1}},
    #
    # where :math:`\Sigma` is a covariance prior and :math:`\Sigma_\alpha` is the noise covariance. Denoised completation can be performed using  the :class:`~spyrit.core.recon.TikhonovMeasurementPriorDiag` class (see documentation for more details).

    ######################################################################
    # In practice, it is more convenient to use the :class:`spyrit.core.recon.DCNet` class, which relies on a forward operator, a preprocessing operator, and a covariance prior.

    from spyrit.core.recon import DCNet

    dcnet = DCNet(noise_op, prep_op, Cov)

    # Use GPU, if available
    dcnet = dcnet.to(device)
    y = y.to(device)

    with torch.no_grad():
        z_dcnet = dcnet.reconstruct(y)

    ######################################################################
    # .. note::
    #   In this tutorial, the covariance matrix used to define subsampling is also used as prior knowledge during reconstruction.

    # %%
    # (Learned) Denoising in the image domain
    # =========================================

    ######################################################################
    # To implement denoising in the image domain, we provide a :class:`spyrit.core.nnet.Unet` denoiser to a :class:`spyrit.core.recon.DCNet`.

    from spyrit.core.nnet import Unet

    denoi = Unet()
    dcnet_unet = DCNet(noise_op, prep_op, Cov, denoi)
    dcnet_unet = dcnet_unet.to(device)  # Use GPU, if available

    ########################################################################
    # We load pretrained weights for the UNet

    from spyrit.core.train import load_net

    local_folder = "./model/"
    # Create model folder
    if os.path.exists(local_folder):
        print(f"{local_folder} found")
    else:
        os.mkdir(local_folder)
        print(f"Created {local_folder}")

    # Load pretrained model
    url = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
    dataID = "67221559f03a54733161e960"  # unique ID of the file
    data_name = "tuto6_dc-net_unet_stl10_N0_100_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_light.pth"
    model_unet_path = os.path.join(local_folder, data_name)

    if os.path.exists(model_unet_path):
        print(f"Model found : {data_name}")

    else:
        print(f"Model not found : {data_name}")
        print(f"Downloading model... ", end="")
        try:
            gc = girder_client.GirderClient(apiUrl=url)
            gc.downloadFile(dataID, model_unet_path)
            print("Done")
        except Exception as e:
            print("Failed with error: ", e)

    # Load pretrained model
    load_net(model_unet_path, dcnet_unet, device, False)

    ######################################################################
    # We reconstruct the image
    with torch.no_grad():
        z_dcnet_unet = dcnet_unet.reconstruct(y)

    # %%
    # Results
    # =========================================

    from spyrit.misc.disp import add_colorbar, noaxis

    f, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot the ground-truth image
    im1 = axs[0, 0].imshow(x[0, 0, :, :], cmap="gray")
    axs[0, 0].set_title("Ground-truth image", fontsize=16)
    noaxis(axs[0, 0])
    add_colorbar(im1, "bottom")

    # Plot the pseudo inverse solution
    im2 = axs[0, 1].imshow(z_invnet.cpu()[0, 0, :, :], cmap="gray")
    axs[0, 1].set_title("Pseudo inverse", fontsize=16)
    noaxis(axs[0, 1])
    add_colorbar(im2, "bottom")

    # Plot the solution obtained from denoised completion
    im3 = axs[1, 0].imshow(z_dcnet.cpu()[0, 0, :, :], cmap="gray")
    axs[1, 0].set_title(f"Denoised completion", fontsize=16)
    noaxis(axs[1, 0])
    add_colorbar(im3, "bottom")

    # Plot the solution obtained from denoised completion with UNet denoising
    im4 = axs[1, 1].imshow(z_dcnet_unet.cpu()[0, 0, :, :], cmap="gray")
    axs[1, 1].set_title(f"Denoised completion with UNet denoising", fontsize=16)
    noaxis(axs[1, 1])
    add_colorbar(im4, "bottom")

    plt.show()

    ######################################################################
    # .. note::
    #   While the pseudo inverse reconstrcution is pixelized, the solution obtained by denoised completion is smoother. DCNet with UNet denoising in the image domain provides the best reconstruction.

    ######################################################################
    # .. note::
    #   We refer to `spyrit-examples tutorials <http://github.com/openspyrit/spyrit-examples/tree/master/tutorial>`_ for a comparison of different solutions (pinvNet, DCNet and DRUNet) that can be run in colab.
