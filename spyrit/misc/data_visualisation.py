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

from astropy.io import fits
import matplotlib.pyplot as plt

#Show basic information of a fits image acquired
#with the andor zyla and plot the image
def show_image_and_infos(path,file): 
    hdul = fits.open(path+file)
    show_images_infos(path,file)
    plt.figure()
    plt.imshow(hdul[0].data[0])
    plt.show()
    
def show_images_infos(path,file): #Show basic information of a fits image acquired
    hdul = fits.open(path+file)
    print("***** Name file : "+file+' *****')
    print("Type de données : "+hdul[0].header['DATATYPE'])
    print("Mode d'acquisition : "+hdul[0].header['ACQMODE'])
    print("Temps d'exposition : "+str(hdul[0].header['EXPOSURE']))
    print("Temps de lecture : "+str(hdul[0].header['READTIME']))
    print("Longeur d'onde de Rayleigh : "+str(hdul[0].header['RAYWAVE']))
    print("Longeur d'onde détectée : "+str(hdul[0].header['DTNWLGTH']))
    print("***********************************"+'\n')
    


# Plot the resulting fuction of to set of 1D data with the same dimension
def simple_plot_2D(Lx,Ly,fig=None,title=None,xlabel=None,ylabel=None,style_color='b'):

    plt.figure(fig)
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(Lx,Ly,style_color)
    plt.show()

# Plot a 2D matrix
def plot_im2D(Im,fig=None,title=None,xlabel=None,ylabel=None,cmap='viridis'):

    plt.figure(fig)
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.imshow(Im,cmap=cmap)
    plt.colorbar()
    plt.show()
  
  
