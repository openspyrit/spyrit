#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Walsh-ordered Hadamard tranforms.

Longer description of this module.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

#__author__ = "One solo developer"
__authors__ = ["Sebastien Crombez", "Nicolas Ducros"]
__contact__ = "nicolas.ducros@creatis.insa-lyon.fr"
#__copyright__ = "Copyright $YEAR, $COMPANY_NAME"
#__credits__ = ["One developer", "And another one", "etc"]
__date__ = "2020/01/15"
__deprecated__ = False
#__email__ =  "nicolas.ducros@creatis.insa-lyon.fr"
__license__ = "GPLv3"
__maintainer__ = "developer"
__status__ = "Development"
#__version__ = "0.0.1"


import math
import numpy as np 
from scipy.linalg import hadamard
from sympy.combinatorics.graycode import GrayCode
import torch


"""Generation of a Gray permutation matrix"""


def conv_list_b2_b10(l): # convert a liste of number in the base 2 to the base 10
    N = len(l)
    for i in range(N):
        l[i] = int(l[i],2)
    return(l)
    
def Mat_of_ones_from_list_index(l): # generate a matrix of zero and ones from list of index
    N = len(l)
    M_out = np.zeros((N,N))
    for i in range(N):
        M_out[i,l[i]] = 1
    return(M_out)
    
def gray_code_permutation(n): # Generate the N grey code permutation matrix
    N = int(math.log(n, 2))
    graycode = GrayCode(N)
    graycode_list = list(graycode.generate_gray())
    return(Mat_of_ones_from_list_index(conv_list_b2_b10((graycode_list))))
    
"""Generation of a bit permutation matrix"""
    
def bit_reverse_traverse(a): #internet function to generate bit reverse
    n = a.shape[0]
    assert(not n&(n-1) ) # assert that n is a power of 2

    if n == 1:
        yield a[0]
    else:
        even_index = np.arange(n//2)*2
        odd_index = np.arange(n//2)*2 + 1
        for even in bit_reverse_traverse(a[even_index]):
            yield even
        for odd in bit_reverse_traverse(a[odd_index]):
            yield odd

def get_bit_reversed_list(l): #internet function to generate bit reverse
    n = len(l)

    indexs = np.arange(n)
    b = []
    for i in bit_reverse_traverse(indexs):
        b.append(l[i])

    return b

def bit_reverse_matrix(n):#internet function to generate bit reverse
    l_br = get_bit_reversed_list([k for k in range(n)])
    Mat_out = np.zeros((n,n))
    
    for i in range(n):
        Mat_out[i,l_br[i]] = 1
    return(Mat_out)
    

def walsh_matrix(n): 
    """Return 1D Walsh-ordered Hadamard transform matrix

    Args:
        n (int): Order of the matrix, a power of two.

    Returns:
        np.ndarray: A n-by-n array
    
    Examples:
        Walsh-ordered Hadamard matrix of order 8

        >>> print(walsh_matrix(8))
    """
    BR = bit_reverse_matrix(n)
    GRp = gray_code_permutation(n)
    H = hadamard(n)
    return(np.dot(np.dot(GRp,BR),H)) # Apply permutation to the hadmard matrix 
    
    
def walsh2(X,H=None):
    r"""Return 2D Walsh-ordered Hadamard transform of an image :math:`H^\top X H`

    Args:
        X (np.ndarray): image as a 2d array. The size is a power of two.
        H (np.ndarray, optional): 1D Walsh-ordered Hadamard transformation matrix

    Returns:
        np.ndarray: Hadamard transformed image as a 2D array.
    """
    if H is None:
         H = walsh_matrix(len(X))
    return(np.dot(np.dot(H,X),H))


def iwalsh2(X,H=None):
    """Return 2D inverse Walsh-ordered Hadamard transform of an image

    Args:
        X (np.ndarray): Image as a 2D array. The image is square and its size is a power of two.
        H (np.ndarray, optional): 1D inverse Walsh-ordered Hadamard transformation matrix

    Returns:
        np.ndarray: Inverse Hadamard transformed image as a 2D array.
    """
    if H is None:
         H = walsh_matrix(len(X))
    return(walsh2(X,H)/len(X)**2)

def walsh2_matrix(n):
    """Return 2D Walsh-ordered Hadamard transformation matrix

    Args:
        n (int): Order of the matrix, which should be a power of two.

    Returns:
        H (np.ndarray): A n*n-by-n*n array
    """
    H = np.zeros((n**2, n**2))
    H1d = walsh_matrix(n)
    for i in range(n**2):
        image = np.zeros((n**2,1));
        image[i] = 1;
        image = np.reshape(image, (n, n));
        hadamard = walsh2(image, H1d);
        H[:, i] = np.reshape(hadamard, (1,n**2));
    return H

def walsh2_torch(im,H=None):
    """Return 2D Walsh-ordered Hadamard transform of an image

    Args:
        im (torch.Tensor): Image, typically a B-by-C-by-W-by-H Tensor
        H (torch.Tensor, optional): 1D Walsh-ordered Hadamard transformation matrix. A 2-D tensor of size W-by-H.

    Returns:
        torch.Tensor: Hadamard transformed image. Same size as im
        
    Examples:
        >>> im = torch.randn(256, 1, 64, 64)
        >>> had = walsh2_torch(im)
    """
    if H is None:
         H = torch.from_numpy(walsh_matrix(im.shape[3]).astype('float32'))
    H = H.to(im.device)
    return  torch.matmul(torch.matmul(H,im),H)
