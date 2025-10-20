# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:53:59 2024

@author: ducros
"""
import numpy as np
import warnings
from typing import Tuple

import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import colorsys
from pathlib import Path
from matplotlib.colors import ListedColormap


# %%
def wavelength_to_rgb(
    wavelength: float, gamma: float = 0.8
) -> Tuple[float, float, float]:
    """Converts wavelength to RGB.

    Based on https://gist.github.com/friendly/67a7df339aa999e2bcfcfec88311abfc.
    Itself based on code by Dan Bruton:
    http://www.physics.sfasu.edu/astro/color/spectra.html

    Args:
        wavelength (float):
            Single wavelength to be converted to RGB.
        gamma (float, optional):
            Gamma correction. Defaults to 0.8.

    Returns:
        Tuple[float, float, float]:
            RGB value.
    """

    if np.min(wavelength) < 380 or np.max(wavelength) > 750:
        warnings.warn("Some wavelengths are not in the visible range [380-750] nm")

    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma

    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0

    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma

    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0

    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0

    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0

    else:
        R = 0.0
        G = 0.0
        B = 0.0

    return R, G, B


def wavelength_to_rgb_mat(wav_range, gamma=1):

    rgb_mat = np.zeros((len(wav_range), 3))

    for i, wav in enumerate(wav_range):
        rgb_mat[i, :] = wavelength_to_rgb(wav, gamma)

    return rgb_mat


def spectral_colorization(M_gray, wav, axis=None):
    """
    Colorize the last dimension of an array

    Args:
        M_gray (np.ndarray): Grayscale array where the last dimension is the
        spectral dimension. This is an A-by-C array, where A can indicate multiple
        dimensions (e.g., 4-by-3-by-7) and C is the number of spectral channels.

        wav (np.ndarray): Wavelenth. This is a 1D array of size C.

        axis (None or int or tuple of ints, optional): Axis or axes along which
        the grayscale input is normalized. By default, global normalization
        across all axes is considered.

    Returns:
        M_color (np.ndarray): Color array with an extra dimension. This is an A-by-C-by-3 array.

    """

    # Normalize to adjust contrast
    M_gray_min = M_gray.min(keepdims=True, axis=axis)
    M_gray_max = M_gray.max(keepdims=True, axis=axis)
    M_gray = (M_gray - M_gray_min) / (M_gray_max - M_gray_min)

    #
    rgb_mat = wavelength_to_rgb_mat(wav, gamma=1)
    M_red = M_gray @ np.diag(rgb_mat[:, 0])
    M_green = M_gray @ np.diag(rgb_mat[:, 1])
    M_blue = M_gray @ np.diag(rgb_mat[:, 2])

    M_color = np.stack((M_red, M_green, M_blue), axis=-1)

    return M_color


def colorize(im, color, clip_percentile=0.1):
    """
    Helper function to create an RGB image from a single-channel image using a
    specific color.
    """
    # Check that we just have a 2D image
    if im.ndim > 2 and im.shape[2] != 1:
        raise ValueError("This function expects a single-channel image!")

    # Rescale the image according to how we want to display it
    im_scaled = im.astype(np.float32) - np.percentile(im, clip_percentile)
    im_scaled = im_scaled / np.percentile(im_scaled, 100 - clip_percentile)
    print(
        f"Norm: min={np.percentile(im, clip_percentile)}, max={np.percentile(im_scaled, 100 - clip_percentile)}"
    )
    print(f"New:  min={im_scaled.min()}, max={im_scaled.max()}")
    im_scaled = np.clip(im_scaled, 0, 1)

    # Need to make sure we have a channels dimension for the multiplication to work
    im_scaled = np.atleast_3d(im_scaled)

    # Reshape the color (here, we assume channels last)
    color = np.asarray(color).reshape((1, 1, -1))
    return im_scaled * color


def generate_colormap(wavelength: float, img_size: int = 256,
    gamma: float = 0.8) -> np.ndarray:
    """Generates colormap for a wavelength.

    Args:
        wavelength (float):
            Single wavelength used for colormap generation.
        img_size (int):
            Reconstructed image size.
        gamma (float):
            Gamma correction.

    Returns:
        np.ndarray:
            Array with dimensions (img_size,4). Each column corresponds to the
            RGBA values. A stands for alpha or transparency and is currently
            set to 1.
    """

    saturation = np.arange(0, 1, 1/img_size)
    
    r, g, b = wavelength_to_rgb(wavelength, gamma)
    
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Creating colormap RGBA (A stands for alpha or transparency)
    colormap = np.ones((img_size, 4))
        
    for i in range(img_size):
        r, g, b = colorsys.hsv_to_rgb(h, v, saturation[i])
        colormap[i, 0] = r
        colormap[i, 1] = g
        colormap[i, 2] = b
        
    return colormap


def plot_hs(strategy, img, wav, suptitle=None, save_fig=False, 
             results_root=None, data_folder=None):
    """Plot hyperspectral data with wavelength-aware colormaps.
    
    Creates a grid of subplots showing each spectral band with a colormap that
    corresponds to the wavelength color. Each band is displayed with a custom
    colormap generated from the actual wavelength values.
    
    Args:
        strategy (str): Strategy type, either 'slice' or 'bin'. Used for labeling.
        img (np.ndarray): 3D numpy array with shape (height, width, n_wav)
            containing the hyperspectral data.
        wav (array-like): Array of wavelength values in nanometers, length n_wav.
        suptitle (str, optional): Super title for the entire figure. 
            Defaults to None.
        save_fig (bool, optional): Whether to save the figure as PDF. 
            Defaults to False.
        results_root (Path or str, optional): Root directory for saving figures.
            Required if save_fig is True. Defaults to None.
        data_folder (Path or str, optional): Data folder name for organizing 
            saved figures. Required if save_fig is True. Defaults to None.
    
    Raises:
        ValueError: If save_fig is True but results_root or data_folder is None.
    
    Returns:
        None: Displays the plot and optionally saves it.
    
    Example:
        >>> wav = np.array([450, 500, 550, 600, 650, 700])
        >>> data = np.random.rand(64, 64, 6)
        >>> plot_hs('bin', data, 6, wav, suptitle='Test Data')
    """
    # Validate save parameters
    if save_fig and (results_root is None or data_folder is None):
        raise ValueError("results_root and data_folder must be provided when save_fig=True")
    
    height, width, n_wav = img.shape
    n_rows, n_cols = n_wav // 4, 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), 
                            gridspec_kw={'wspace': 0.3, 'hspace': 0.05})
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_wav):
        ax = axes[i // n_cols, i % n_cols]
        
        # Generate spectral-aware colormap for this wavelength
        wavelength_nm = float(wav[i])  # Convert from tensor to float
        spectral_cmap_data = generate_colormap(wavelength_nm, img_size=height * width)
        spectral_cmap = ListedColormap(spectral_cmap_data)
        
        # Display the grayscale image with spectral colormap
        im = ax.imshow(img[:, :, i], cmap=spectral_cmap)
        slice_or_bin = "Slice" if strategy == 'slice' else "Bin"
        ax.set_title(f"{slice_or_bin} {i+1} ({wavelength_nm:.0f} nm)")
        ax.axis('off')
        
        # Add colorbar with spectral colormap, closer to the axis
        cax = fig.add_axes([ax.get_position().x1 + 0.005, 
                            ax.get_position().y0, 
                            0.01, 
                            ax.get_position().height])
        plt.colorbar(im, cax=cax)
    
    # Hide unused subplots
    for i in range(n_wav, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis('off')

    if save_fig:
        path_fig = Path(results_root) / data_folder
        path_fig.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_fig / f"hs_{strategy}_{suptitle}.pdf", bbox_inches='tight')

    plt.suptitle(suptitle, fontsize=16) if suptitle else None
    plt.show()
