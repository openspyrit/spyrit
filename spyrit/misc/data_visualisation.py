# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 08:56:13 2020

@author: crombez
"""

import matplotlib.pyplot as plt


# Plot the resulting fuction of to set of 1D data with the same dimension
def simple_plot_2D(
    Lx, Ly, fig=None, title=None, xlabel=None, ylabel=None, style_color="b"
):
    plt.figure(fig)
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(Lx, Ly, style_color)
    plt.show()


# Plot a 2D matrix
def plot_im2D(Im, fig=None, title=None, xlabel=None, ylabel=None, cmap="viridis"):
    plt.figure(fig)
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.imshow(Im, cmap=cmap)
    plt.colorbar()
    plt.show()
