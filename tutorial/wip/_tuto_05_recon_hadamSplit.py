# %%

if False:

    # %%
    # Split measurement operator and no noise
    # -----------------------------------------------------------------------------
    # .. _split_measurements:

    ###############################################################################
    # .. math::
    #       y = P\tilde{x}= \begin{bmatrix} H_{+} \\ H_{-} \end{bmatrix} \tilde{x}.

    ###############################################################################
    # Hadamard split measurement operator is defined in the :class:`spyrit.core.meas.HadamSplit` class.
    # It computes linear measurements from incoming images, where :math:`P` is a
    # linear operator (matrix) with positive entries and :math:`\tilde{x}` is an image.
    # The class relies on a matrix :math:`H` with
    # shape :math:`(M,N)` where :math:`N` represents the number of pixels in the
    # image and :math:`M \le N` the number of measurements. The matrix :math:`P`
    # is obtained by splitting the matrix :math:`H` as :math:`H = H_{+}-H_{-}` where
    # :math:`H_{+} = \max(0,H)` and :math:`H_{-} = \max(0,-H)`.

    # %%
    # Measurement and noise operators
    # -----------------------------------------------------------------------------

    ###############################################################################
    # We compute the measurement and noise operators and then
    # simulate the measurement vector :math:`y`.

    ###############################################################################
    # We consider Poisson noise, i.e., a noisy measurement vector given by
    #
    # .. math::
    #       y \sim \mathcal{P}(\alpha P \tilde{x}),
    #
    # where :math:`\alpha` is a scalar value that represents the maximum image intensity
    # (in photons). The larger :math:`\alpha`, the higher the signal-to-noise ratio.

    ###############################################################################
    # We use the :class:`spyrit.core.noise.Poisson` class, set :math:`\alpha`
    # to 100 photons, and simulate a noisy measurement vector for the two sampling
    # strategies. Subsampling is handled internally by the :class:`~spyrit.core.meas.HadamSplit` class.

    from spyrit.core.noise import Poisson
    from spyrit.core.meas import HadamSplit

    alpha = 100.0  # number of photons

    # "Naive subsampling"
    # Measurement and noise operators
    meas_nai_op = HadamSplit(M, h, Ord_naive)
    noise_nai_op = Poisson(meas_nai_op, alpha)

    # Measurement operator
    y_nai = noise_nai_op(x)  # a noisy measurement vector

    # "Variance subsampling"
    meas_var_op = HadamSplit(M, h, Ord_variance)
    noise_var_op = Poisson(meas_var_op, alpha)
    y_var = noise_var_op(x)  # a noisy measurement vector

    print(f"Shape of image: {x.shape}")
    print(f"Shape of simulated measurements y: {y_var.shape}")

    # %%
    # The preprocessing operator measurements for split measurements
    # -----------------------------------------------------------------------------

    ###############################################################################
    # We compute the preprocessing operators for the three cases considered above,
    # using the :mod:`spyrit.core.prep` module. As previously introduced,
    # a preprocessing operator applies to the noisy measurements in order to
    # compensate for the scaling factors that appear in the measurement or noise operators:
    #
    # .. math::
    #       m = \texttt{Prep}(y),

    ###############################################################################
    # We consider the :class:`spyrit.core.prep.SplitPoisson` class that intends
    # to "undo" the :class:`spyrit.core.noise.Poisson` class, for split measurements, by compensating for
    #
    # * the scaling that appears when computing Poisson-corrupted measurements
    #
    # * the affine transformation to get images in [0,1] from images in [-1,1]
    #
    # For this, it computes
    #
    # .. math::
    #       m = \frac{2(y_+-y_-)}{\alpha} - P\mathbb{1},
    #
    # where :math:`y_+=H_+\tilde{x}` and :math:`y_-=H_-\tilde{x}`.
    # This is handled internally by the :class:`spyrit.core.prep.SplitPoisson` class.

    ###############################################################################
    # We compute the preprocessing operator and the measurements vectors for
    # the two sampling strategies.

    from spyrit.core.prep import SplitPoisson

    # "Naive subsampling"
    #
    # Preprocessing operator
    prep_nai_op = SplitPoisson(alpha, meas_nai_op)

    # Preprocessed measurements
    m_nai = prep_nai_op(y_nai)

    # "Variance subsampling"
    prep_var_op = SplitPoisson(alpha, meas_var_op)
    m_var = prep_var_op(y_var)

    # %%
    # Noiseless measurements
    # -----------------------------------------------------------------------------

    ###############################################################################
    # We consider now noiseless measurements for the "naive subsampling" strategy.
    # We compute the required operators and the noiseless measurement vector.
    # For this we use the :class:`spyrit.core.noise.NoNoise` class, which normalizes
    # the input image to get an image in [0,1], as explained in
    # :ref:`acquisition operators tutorial <tuto_acquisition_operators>`.
    # For the preprocessing operator, we assign the number of photons equal to one.

    from spyrit.core.noise import NoNoise

    nonoise_nai_op = NoNoise(meas_nai_op)
    y_nai_nonoise = nonoise_nai_op(x)  # a noisy measurement vector

    prep_nonoise_op = SplitPoisson(1.0, meas_nai_op)
    m_nai_nonoise = prep_nonoise_op(y_nai_nonoise)

    ###############################################################################
    # We can now plot the three measurement vectors

    # Plot the three measurement vectors
    m_plot = meas2img(m_nai_nonoise, Ord_naive)
    m_plot2 = meas2img(m_nai, Ord_naive)
    m_plot3 = spytorch.meas2img(m_var, Ord_variance)

    m_plot_max = m_plot[0, 0, :, :].max()
    m_plot_min = m_plot[0, 0, :, :].min()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    im1 = ax1.imshow(m_plot[0, 0, :, :], cmap="gray")
    ax1.set_title("Noiseless measurements $m$ \n 'Naive' subsampling", fontsize=20)
    noaxis(ax1)
    add_colorbar(im1, "bottom", size="20%")

    im2 = ax2.imshow(m_plot2[0, 0, :, :], cmap="gray", vmin=m_plot_min, vmax=m_plot_max)
    ax2.set_title("Measurements $m$ \n 'Naive' subsampling", fontsize=20)
    noaxis(ax2)
    add_colorbar(im2, "bottom", size="20%")

    im3 = ax3.imshow(m_plot3[0, 0, :, :], cmap="gray", vmin=m_plot_min, vmax=m_plot_max)
    ax3.set_title("Measurements $m$ \n 'Variance' subsampling", fontsize=20)
    noaxis(ax3)
    add_colorbar(im3, "bottom", size="20%")

    plt.show()

    # %%
    # PinvNet network
    # -----------------------------------------------------------------------------

    ###############################################################################
    # We use the :class:`spyrit.core.recon.PinvNet` class where
    # the pseudo inverse reconstruction is performed by a neural network

    from spyrit.core.recon import PinvNet

    pinvnet_nai_nonoise = PinvNet(nonoise_nai_op, prep_nonoise_op)
    pinvnet_nai = PinvNet(noise_nai_op, prep_nai_op)
    pinvnet_var = PinvNet(noise_var_op, prep_var_op)

    # Reconstruction
    z_nai_nonoise = pinvnet_nai_nonoise.reconstruct(y_nai_nonoise)
    z_nai = pinvnet_nai.reconstruct(y_nai)
    z_var = pinvnet_var.reconstruct(y_var)

    ###############################################################################
    # We can now plot the three reconstructed images
    from spyrit.misc.disp import add_colorbar, noaxis

    # Plot
    f, axs = plt.subplots(2, 2, figsize=(10, 10))
    im1 = axs[0, 0].imshow(x[0, 0, :, :], cmap="gray")
    axs[0, 0].set_title("Ground-truth image")
    noaxis(axs[0, 0])
    add_colorbar(im1, "bottom")

    im2 = axs[0, 1].imshow(z_nai_nonoise[0, 0, :, :], cmap="gray")
    axs[0, 1].set_title("Reconstruction noiseless")
    noaxis(axs[0, 1])
    add_colorbar(im2, "bottom")

    im3 = axs[1, 0].imshow(z_nai[0, 0, :, :], cmap="gray")
    axs[1, 0].set_title("Reconstruction \n 'Naive' subsampling")
    noaxis(axs[1, 0])
    add_colorbar(im3, "bottom")

    im4 = axs[1, 1].imshow(z_var[0, 0, :, :], cmap="gray")
    axs[1, 1].set_title("Reconstruction \n 'Variance' subsampling")
    noaxis(axs[1, 1])
    add_colorbar(im4, "bottom")

    plt.show()

    ###############################################################################
    # .. note::
    #
    #       Note that reconstructed images are pixelized when using the "naive subsampling",
    #       while they are smoother and more similar to the ground-truth image when using the
    #       "variance subsampling".
    #
    #       Another way to further improve results is to include a nonlinear post-processing step,
    #       which we will consider in a future tutorial.
