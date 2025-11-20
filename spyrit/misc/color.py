# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:53:59 2024

@author: ducros
"""
import numpy as np
import warnings
from typing import Tuple
from matplotlib.colors import LinearSegmentedColormap

import warnings
from typing import Tuple

import numpy as np


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


def wavelength_to_colormap(wav, gamma=.6):
    """
    Creates a linear Matplotlib colormap that transitions from black to a specific
    color corresponding to a given electromagnetic wavelength.
    
    
    Args:
        wav (float): The wavelength in nanometers (nm) to determine the target 
        color. Typically, this would be in the visible spectrum range (~380
        to 780 nm).
        
        gamma (float, optional): The gamma correction factor applied when 
        calculating the RGB color from the wavelength. Defaults to 0.6.
    
    Returns:
        matplotlib.colors.LinearSegmentedColormap: A custom colormap object 
        named 'DarkToColor' that spans from black at the low end (0.0) to the
        calculated wavelength-based color at the high end (1.0).
    
    Raises:
        NameError: If `wavelength_to_rgb` or `LinearSegmentedColormap` (from
        Matplotlib) are not defined or imported.
    
    Example:
        >>> cmap = wavelength_to_colormap(550, gamma=0.8) # Green color at 550nm
        >>> print(cmap)
        <matplotlib.colors.LinearSegmentedColormap object at ...>
    """

    # 'dark_color' is the color at 0.0 (start)
    dark_color = 'black' # You can use any dark color (e.g., '#000033', 'black', 'darkred')
    # 'target_color' is the color at 1.0 (end)
    
    target_color = wavelength_to_rgb(wav, gamma)
    
    # 2. Create the list of color nodes (tuples of position and color)
    # The colormap will transition linearly between these nodes.
    color_list = [(0.0, dark_color),
                  (1.0, target_color)]
    
    custom_cmap = LinearSegmentedColormap.from_list('DarkToColor', color_list)
    
    return custom_cmap

