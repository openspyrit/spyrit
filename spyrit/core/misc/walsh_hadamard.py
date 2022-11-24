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

#------------------------------------------------------------------------------
#-- To generate sequency (aka Walsh) order --------------------------------------
#------------------------------------------------------------------------------
def b2_to_b10(l):
    """Convert a list of numbers in base 2 to base 10    
    Args:
        l (list[str]): base2 numbers.

    Returns:
        list[int]: base10 numbers

    Examples:
        >>> 
    """ 
    N = len(l)
    for i in range(N):
        l[i] = int(l[i],2)
    return l 

def perm_matrix_from_ind(l): # generate a matrix of zero and ones from list of index
    N = len(l)
    P = np.zeros((N,N))
    for i in range(N):
        P[i, l[i]] = 1
    return P
    
def gray_code_permutation(n): # Generate the N grey code 

    N = int(math.log(n, 2))
    graycode = GrayCode(N)
    graycode_list = list(graycode.generate_gray())
    return perm_matrix_from_ind(b2_to_b10((graycode_list)))

def gray_code_list(n): # Generate the N grey code permutation matrix
    N = int(math.log(n, 2))
    graycode = GrayCode(N)
    graycode_list = list(graycode.generate_gray())
    return b2_to_b10(graycode_list)
    
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

def get_bit_reversed_list(l): # from the internet
    n = len(l)
    indexs = np.arange(n)
    b = []
    for i in bit_reverse_traverse(indexs):
        b.append(l[i])

    return b

def bit_reversed_matrix(n): #internet function to generate bit reverse
    br = bit_reversed_list(n)
    return perm_matrix_from_ind(br)

def bit_reversed_list(n):
    br = get_bit_reversed_list([k for k in range(n)])
    return br

def sequency_perm(X, ind=None):
    """ Permute the rows of a matrix to get sequency order   
    Args:
        X (np.ndarray): n-by-m input matrix
        ind : index list of length n

    Returns:
        np.ndarray: n-by-m input matrix 

    Examples:
        >>> 
    """ 
    if ind is None:
        ind = sequency_perm_ind(len(X))

    Y = np.zeros(X.shape)
    #Y = X[ind,] # returns dtype = object ?!
    for i in range(X.shape[0]):
        Y[i,] = X[ind[i],]
    return Y

def sequency_perm_torch(X, ind=None):
    """ Permute the last dimension of a tensor to get sequency order    
    Args:
        X (torch.tensor): -by-n input matrix
        ind : index list of length n

    Returns:
        torch.tensor: -by-n input matrix 
        
    Example :
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1, -2])
        >>> x = x[None, None, :]
        >>> x = wh.sequency_perm_torch(x)
        >>> print(x)
    """ 
    if ind is None:
        ind = sequency_perm_ind(X.shape[-1])

    Y = X[...,ind]
    return Y

def sequency_perm_matrix(n):
    """Return permutation matrix to get sequency from the natural order    
    Args:
        n (int): Order of the matrix, a power of two.
    
    Returns:
        np.ndarray: A n-by-n permutation matrix
    
    Examples:
        Permutation matrix of order 8
    
        >>> print(sequency_perm_matrix(8))
    """ 
    BR = bit_reversed_matrix(n)
    GC = gray_code_permutation(n) 
    return GC @ BR

def sequency_perm_ind(n):
    """Return permutation indices to get sequency from the natural order
     
    Args:
        n (int): Order of the matrix, a power of two.
    
    Returns:
        list: 
    
    Examples:
        Permutation indices to get a Walsh matrix of order 8
    
        >>> print(sequency_perm_ind(8))
    """ 
    perm_br = bit_reversed_list(n)
    perm_gc = gray_code_list(n)
    perm = [perm_br[perm_gc[k]] for k in range(n)]
    
    return perm
#------------------------------------------------------------------------------
#-- 1D Walsh/Hamadard matrix and transforms -----------------------------------
#------------------------------------------------------------------------------
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
    P = sequency_perm_matrix(n)
    H = hadamard(n)
    return P @ H

def fwht(x, order=True):
    """Fast Walsh-Hadamard transform of x
    
    Args:
        x (np.ndarray): n-by-1 input signal, where n is a power of two.
        order (bool, optional): True for sequency (default), False for natural.
        order (list, optional): permutation indices.
    
    Returns:
        np.ndarray: n-by-1 transformed signal
    
    Example 1:
    
        Fast sequency-ordered (i.e., Walsh) Hadamard transform
        
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1, -2])
        >>> y = wh.fwht(x)
        >>> print(y)
        
    Example 2:
    
        Fast Hadamard transform
        
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1, -2])
        >>> y = wh.fwht(x, False)
        >>> print(y)
    
    Example 3:
    
        Permuted fast Hadamard transform
        
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1, -2])
        >>> ind = [1, 0, 3, 2, 7, 4, 5, 6]
        >>> y = wh.fwht(x, ind)
        >>> print(y)
        
    Example 4:
    
        Comparison with Walsh-Hadamard transform via matrix-vector product
        
        >>> from spyrit.misc.walsh_hadamard import fwht, walsh_matrix
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1, -2])
        >>> y1 = fwht(x)
        >>> H = walsh_matrix(8)
        >>> y2 = H @ x
        >>> print(f"Diff = {np.linalg.norm(y1-y2)}")
        
    Example 5:
    
        Comparison with the fast Walsh-Hadamard transform from sympy
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import sympy as sp
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1, -2])
        >>> y1 = wh.fwht(x)
        >>> y2 = sp.fwht(x)
        >>> y3 = wh.sequency_perm(np.array(y2))
        >>> print(f"Diff = {np.linalg.norm(y1-y3)}")
        
    Example 6: 
        
        Computation times for a signal of length 2**12
        
        >>> import timeit
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.random.rand(2**12,1)
        >>> t = timeit.timeit(lambda: wh.fwht(x), number=10)
        >>> print(f"Fast Walsh transform, no ind (10x): {t:.3f} seconds")
        >>> ind = wh.sequency_perm_ind(len(x))
        >>> t = timeit.timeit(lambda: wh.fwht(x,ind), number=10)
        >>> print(f"Fast Walsh transform, with ind (10x): {t:.3f} seconds")
        >>> t = timeit.timeit(lambda: wh.fwht(x,False), number=10)
        >>> print(f"Fast Hadamard transform (10x): {t:.3f} seconds")
        >>> import sympy as sp
        >>> t = timeit.timeit(lambda: sp.fwht(x), number=10)
        >>> print(f"Fast Hadamard transform from sympy (10x): {t:.3f} seconds")
    """ 
    n = len(x)
    y = x.copy()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = y[j]
                v = y[j + h]
                y[j] = u + v
                y[j + h] = u - v
        h *= 2
        
    # Arbitrary order
    if type(order)==list:
        y = sequency_perm(y,order)
    # Sequency (aka Walsh) order
    elif order: 
        y = sequency_perm(y)
    # Hadamard order, otherwise
    return y
#------------------------------------------------------------------------------
#-- G-matrix and G-transforms -------------------------------------------------
#------------------------------------------------------------------------------
def walsh_G_matrix(n,H=None): 
    """Return Walsh-ordered Hadamard S-matrix of order n

    Args:
        n (int): Matrix order. n+1 should be a power of two.
        H (np.ndarray, optional): 

    Returns:
        np.ndarray: n-by-n array
    
    Examples:
        Walsh-ordered Hadamard G-matrix of order 7

        >>> print(walsh_G_matrix(7))
    """
    assert math.log2(n+1)%1 == 0, f"{n}+1 must be a power of two"
    
    if H is None:
        H = walsh_matrix(n+1)
    return H[1:,1:]

def walsh_G(x, G=None): 
    """Return the Walsh S-transform of x

    Args:
        x (np.ndarray): n-by-1 signal. n+1 should be a power of two.

    Returns:
        np.ndarray: n-by-1 S-transformed signal
    
    Examples:
        Walsh-ordered S-transform of a 15-by-1 signal

        >>> x = np.random.rand(15,1)
        >>> s = walsh_S(x)
    """
    if G is None:
         G = walsh_G_matrix(len(x))
         
    return G @ x

def fwalsh_G(x,ind=True): 
    """Fast Walsh G-transform of x

    Args:
        x (np.ndarray): n-by-1 signal. n+1 should be a power of two.
        ind (bool, optional): True for sequency (default)
        ind (list, optional): permutation indices. This is faster than True 
                            when repeating the sequency-ordered transform 
                            multilple times.  

    Returns:
        np.ndarray: n-by-1 S-transformed signal
    
    Example 1:
        Walsh-ordered G-transform of a signal of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import numpy as np        
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1])
        >>> s = wh.fwalsh_G(x)
        >>> print(s)
    
    Example 2:
        Permuted fast G-transform
        
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1])
        >>> ind = [1, 0, 3, 2, 7, 4, 5, 6]
        >>> y = wh.fwalsh_G(x, ind)
        >>> print(y)
    
    Example 3:
        Repeating the Walsh-ordered G-transform using input indices is faster  
        
        >>> import timeit
        >>> x = np.random.rand(2**12-1,1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_G(x), number=10)
        >>> print(f"No indices as inputs (10x): {t:.3f} seconds")
        >>> ind = wh.sequency_perm_ind(len(x)+1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_G(x,ind), number=10)
        >>> print(f"With indices as inputs (10x): {t:.3f} seconds")
        
    Example 4: 
        Comparison with G-transform via matrix-vector product
        
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([3, 0, -1, 7, 5, 1, -2])
        >>> y1 = wh.fwalsh_G(x)
        >>> y2 = wh.walsh_G(x)
        >>> print(f"Diff = {np.linalg.norm(y1-y2)}")
    """
    x = np.insert(x,0,0)
    y = fwht(x,ind)
    y = y[1:]
    return y

#------------------------------------------------------------------------------
#-- S-matrix and S-transforms -------------------------------------------------
#------------------------------------------------------------------------------
def walsh_S_matrix(n, H=None): 
    """Return Walsh S-matrix of order n

    Args:
        n (int): Matrix order. n+1 should be a power of two.

    Returns:
        np.ndarray: n-by-n array
    
    Examples:
        Walsh-ordered Hadamard S-matrix of order 7

        >>> print(walsh_S_matrix(7))
    """
    return (1 - walsh_G_matrix(n,H))/2

def iwalsh_S_matrix(n, H=None): 
    """Return inverse Walsh S-matrix of order n

    Args:
        n (int): Matrix order. n+1 should be a power of two.

    Returns:
        np.ndarray: n-by-n array
    
    Example 1:
        Inverse of the Walsh S-matrix of order 7

        >>> print(iwalsh_S_matrix(7))
    
    Example 2:
        Check the inverse of the Walsh S-matrix of order 7
        >>> print(iwalsh_S_matrix(7) @ walsh_S_matrix(7))
    """
    return -2*walsh_G_matrix(n,H)/(n+1)

def walsh_S(x, S=None): 
    """Return the Walsh S-transform of x

    Args:
        x (np.ndarray): n-by-1 signal. n+1 should be a power of two.

    Returns:
        np.ndarray: n-by-1 S-transformed signal
    
    Examples:
        Walsh-ordered S-transform of a 15-by-1 signal

        >>> x = np.random.rand(15,1)
        >>> s = walsh_S(x)
    """
    if S is None:
         S = walsh_S_matrix(len(x))
         
    return S @ x

def iwalsh_S(s, T=None): 
    """Return the inverse Walsh S-transform of s

    Args:
        x (np.ndarray): n-by-1 signal. n+1 should be a power of two.

    Returns:
        np.ndarray: n-by-1 inverse transformed signal
    
    Examples:
        Inverse S-transform of a 4095-by-1 signal

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import numpy as np
        >>> x = np.random.rand(4095,1)
        >>> s = wh.walsh_S(x)
        >>> y = wh.iwalsh_S(s)
        >>> err = np.linalg.norm(1-y/x)
        >>> print(f'Error: {err}')
    """
    if T is None:
         T = iwalsh_S_matrix(len(s))
         
    return T @ s

def fwalsh_S(x,ind=True): 
    """Fast Walsh S-transform of x

    Args:
        x (np.ndarray): n-by-1 signal. n+1 should be a power of two.
        ind (bool, optional): True for sequency (default)
        ind (list, optional): permutation indices. This is faster than True 
                            when repeating the sequency-ordered transform 
                            multilple times.  

    Returns:
        np.ndarray: n-by-1 S-transformed signal
    
    Example 1:
        Walsh-ordered S-transform of a signal of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import numpy as np        
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1])
        >>> s = wh.fwalsh_S(x)
        >>> print(s)
         
    Example 2:
        Permuted fast S-transform
        
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1])
        >>> ind = [1, 0, 3, 2, 7, 4, 5, 6]
        >>> y = wh.fwalsh_S(x, ind)
        >>> print(y)
    
    Example 3:
        Repeating the Walsh-ordered S-transform using input indices is faster  
        
        >>> import timeit
        >>> x = np.random.rand(2**12-1,1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_S(x), number=10)
        >>> print(f"No indices as inputs (10x): {t:.3f} seconds")
        >>> ind = wh.sequency_perm_ind(len(x)+1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_S(x,ind), number=10)
        >>> print(f"With indices as inputs (10x): {t:.3f} seconds")
        
    Example 4: 
        Comparison with S-transform via matrix-vector product
        
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([3, 0, -1, 7, 5, 1, -2])
        >>> y1 = wh.fwalsh_S(x)
        >>> y2 = wh.walsh_S(x)
        >>> print(f"Diff = {np.linalg.norm(y1-y2)}")
        
    Example 5: 
        Computation times for a signal of length 2**12-1
        
        >>> import timeit
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.random.rand(2**14-1,1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_S(x), number=10)
        >>> print(f"Fast transform, no ind (10x): {t:.3f} seconds")
        >>> ind = wh.sequency_perm_ind(len(x)+1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_S(x,ind), number=10)
        >>> print(f"Fast transform, with ind (10x): {t:.3f} seconds")
        >>> S = wh.walsh_S_matrix(len(x))
        >>> t = timeit.timeit(lambda: wh.walsh_S(x,S), number=10)
        >>> print(f"Naive transform (10x): {t:.3f} seconds")
    """
    j = x.sum()
    s = fwalsh_G(x,ind)  
    s = (j-s)/2
    return s    

def ifwalsh_S(s, ind=True): 
    """Inverse fast Walsh S-transform of s

    Args:
        x (np.ndarray): n-by-1 signal. n+1 should be a power of two.
        ind (bool, optional): True for sequency (default).
        ind (list, optional): permutation indices. This is faster than True 
                            when repeating the sequency-ordered transform 
                            multilple times.  

    Returns:
        np.ndarray: n-by-1 inverse transformed signal
    
    Examples:
        Inverse S-transform of a 15-by-1 signal
        
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import numpy as np        
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1])
        >>> print(f"signal: {x}")
        >>> s = wh.fwalsh_S(x)
        >>> print(f"s-transform: {s}")
        >>> y = wh.ifwalsh_S(s)
        >>> print(f"inverse s-transform: {y}")
        
    """
    x = -2/(len(s)+1)*fwalsh_G(s,ind) 
    
    return x
#------------------------------------------------------------------------------
#-- 2D transforms -------------------------------------------------------------
#------------------------------------------------------------------------------
def walsh2_matrix(n):
    """Return Walsh-ordered Hadamard matrix in 2D

    Args:
        n (int): Order of the matrix, which must be a power of two.

    Returns:
        H (np.ndarray): A n*n-by-n*n matrix
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

def fwalsh2_S(X,ind=True): 
    """Fast Walsh S-transform of X in "2D"

    Args:
        x (np.ndarray): n-by-n signal. n**2 should be a power of two.
        ind (bool, optional): True for sequency (default)
        ind (list, optional): permutation indices.

    Returns:
        np.ndarray: n-by-1 S-transformed signal
    
    Examples:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import numpy as np
        >>> X = np.array([[1, 3, 0, 8],[7, 5, 1, 2],[2, 3, 6, 1],[4, 6, 8, 0]])
        >>> wh.fwalsh2_S(X)
        
    """    
    x = walsh2_S_unfold(X)
    s = fwalsh_S(x,ind)
    S = walsh2_S_fold(s)
    return S

def ifwalsh2_S(Y,ind=True): 
    """Inverse Fast Walsh S-transform of Y in "2D"

    Args:
        Y (np.ndarray): n-by-n signal. n**2 should be a power of two.
        ind (bool, optional): True for sequency (default)
        ind (list, optional): permutation indices.

    Returns:
        np.ndarray: n-by-1 S-transformed signal
    
    Examples:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> X = np.array([[1, 3, 0, 8],[7, 5, 1, 2],[2, 3, 6, 1],[4, 6, 8, 0]])
        >>> print(f"image:\n {X}")
        >>> Y = wh.fwalsh2_S(X)
        >>> print(f"s-transform:\n {Y}")
        >>> Z = wh.ifwalsh2_S(Y)
        >>> print(f"inverse s-transform:\n {Z}")
        
        Note that the first pixel is not meaningful and arbitrily set to 0. 
        
    """    
    x = walsh2_S_unfold(Y)
    s = ifwalsh_S(x,ind)
    S = walsh2_S_fold(s)
    return S

def walsh2_S(X,S=None): 
    """Fast Walsh S-transform of X in "2D"

    Args:
        x (np.ndarray): n-by-n signal. n**2 should be a power of two.
        ind (bool, optional): True for sequency (default)
        ind (list, optional): permutation indices.

    Returns:
        np.ndarray: n-by-1 S-transformed signal
    
    Examples:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import numpy as np
        >>> X = np.array([[1, 3, 0, 8],[7, 5, 1, 2],[2, 3, 6, 1],[4, 6, 8, 0]])
        >>> wh.walsh2_S(X)
        
    """    
    x = walsh2_S_unfold(X)
    y = walsh_S(x,S)
    Y = walsh2_S_fold(y)
    return Y

def iwalsh2_S(Y,T=None): 
    """Inverse Fast Walsh S-transform of Y in "2D"

    Args:
        Y (np.ndarray): n-by-n signal. n**2 should be a power of two.
        T (np.ndarray): Inverse S-matrix 
        
    Returns:
        np.ndarray: n-by-1 S-transformed signal
    
    Examples:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import numpy as np
        >>> X = np.array([[1, 3, 0, 8],[7, 5, 1, 2],[2, 3, 6, 1],[4, 6, 8, 0]])
        >>> print(f"image:\n {X}")
        >>> Y = wh.walsh2_S(X)
        >>> print(f"s-transform:\n {Y}")
        >>> Z = wh.iwalsh2_S(Y)
        >>> print(f"inverse s-transform:\n {Z}")
        
        Note that the first pixel is not meaningful and arbitrily set to 0. 
        
    """    
    y = walsh2_S_unfold(Y)
    x = iwalsh_S(y,T)
    X = walsh2_S_fold(x)
    return X

def walsh2_S_matrix(n):
    """Return Walsh S-matrix in "2d"

    Args:
        n (int): Order of the matrix. n must be a power of two.

    Returns:
        S (np.ndarray): (n*n-1)-by-(n*n-1) matrix
        
    Example 1:
        >>> S = walsh2_S_matrix(4)
    """
    
    H = walsh2_matrix(n)
    S = walsh_S_matrix(n**2-1, H)
    return S

def walsh2_S_fold(x):
    """Fold a signal to get a "2d" s-transformed representation 
    
    Note: the top left (first) pixel is arbitrarily set to zero

    Args:
        x (np.ndarray): N-by- vector. N is such that N+1 = n*n, where n is a 
                        power of two. N = 2**(2b) - 1, where b is an integer 

    Returns:
        X (np.ndarray): n-by-n matrix
        
    Example 1:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> S = wh.walsh2_S_matrix(4)
        >>> X = wh.walsh2_S_fold(S[2,:])
        >>> print(X)
    """
    
    n = (len(x)+1)**0.5
    assert math.log2(n)%1 == 0, f"N+1 = n*n, where n={n:.2f} must be a power of two"

    n = int(n)
    X = np.insert(x,0,0)
    X = np.reshape(X,(n,n))
    return X

def walsh2_S_unfold(X):
    """Unfold a signal from a "2d" s-transformed representation 
    
    Note: the top left (first) pixel is arbitrarily set to zero

    Args:
        X (np.ndarray): n-by-m image.

    Returns:
        X (np.ndarray): (n*n-1)-by-1 signal
        
    Example 1:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> X = np.array([[1, 3, 0, 8],[7, 5, 1, 2]])
        >>> wh.walsh2_S_unfold(X)
    
    """
    x = X.ravel()
    x = x[1:]
    return x
#------------------------------------------------------------------------------
#-- PyTorch functions ---------------------------------------------------------
#------------------------------------------------------------------------------
def fwht_torch(x, order=True):
    """Fast Walsh-Hadamard transform of x
    
    Args:
        x (np.ndarray): -by-n input signal, where n is a power of two.
        order (bool, optional): True for sequency (default), False for natural.
        order (list, optional): permutation indices.
    
    Returns:
        np.ndarray: n-by-1 transformed signal
    
    Example 1:
        Fast sequency-ordered (i.e., Walsh) Hadamard transform
        
        >>> import torch
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1, -2])
        >>> x = x[None,:]
        >>> y = wh.fwht_torch(x)
        >>> print(y)
        
    Example 2:
        Fast Hadamard transform
        
        >>> import torch
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1, -2])
        >>> x = x[None,:]
        >>> y = wh.fwht_torch(x, False)
        >>> print(y)
    
    Example 3:
        Permuted fast Hadamard transform
        
        >>> import numpy as np
        >>> import torch
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1, -2])
        >>> ind = [1, 0, 3, 2, 7, 4, 5, 6]
        >>> y = wh.fwht_torch(x, ind)
        >>> print(y)
        
    Example 4:
        Comparison with the numpy transform
        
        >>> import numpy as np
        >>> import torch
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1, -2])
        >>> y_np = wh.fwht(x)
        >>> x_torch = torch.from_numpy(x).to(torch.device('cuda:0'))
        >>> y_torch = wh.fwht_torch(x_torch)
        >>> print(y_np)
        >>> print(y_torch)

        
    Example 5: 
        Computation times for a signal of length 2**12
        
        >>> import timeit
        >>> import torch
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.random.rand(2**12,1)
        >>> t = timeit.timeit(lambda: wh.fwht(x,False), number=200)
        >>> print(f"Fast Hadamard transform numpy CPU (200x): {t:.4f} seconds")
        >>> x_torch = torch.from_numpy(x)
        >>> t = timeit.timeit(lambda: wh.fwht_torch(x_torch,False), number=200)
        >>> print(f"Fast Hadamard transform pytorch CPU (200x): {t:.4f} seconds")
        >>> x_torch = torch.from_numpy(x).to(torch.device('cuda:0'))
        >>> t = timeit.timeit(lambda: wh.fwht_torch(x_torch,False), number=200)
        >>> print(f"Fast Hadamard transform pytorch GPU (200x): {t:.4f} seconds")
        
    Example 6:         
        CPU vs GPU: Computation times for 512 signals of length 2**12
        
        >>> import timeit
        >>> import torch
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x_cpu = torch.rand(512,2**12)
        >>> t = timeit.timeit(lambda: wh.fwht_torch(x_cpu,False), number=10)
        >>> print(f"Fast Hadamard transform pytorch CPU (10x): {t:.4f} seconds")
        >>> x_gpu = x_cpu.to(torch.device('cuda:0'))
        >>> t = timeit.timeit(lambda: wh.fwht_torch(x_gpu,False), number=10)
        >>> print(f"Fast Hadamard transform pytorch GPU (10x): {t:.4f} seconds")
    
    Example 7:    
        Repeating the Walsh-ordered transform using input indices is faster  
        
        >>> import timeit
        >>> import torch
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = torch.rand(256,2**12).to(torch.device('cuda:0'))
        >>> t = timeit.timeit(lambda: wh.fwht_torch(x), number=100)
        >>> print(f"No indices as inputs (100x): {t:.3f} seconds")
        >>> ind = wh.sequency_perm_ind(x.shape[-1])
        >>> t = timeit.timeit(lambda: wh.fwht_torch(x,ind), number=100)
        >>> print(f"With indices as inputs (100x): {t:.3f} seconds")
    """ 
    n = x.shape[-1]
        
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    y = x[..., None]

    for d in range(m)[::-1]:
        y = torch.cat((y[..., ::2, :] + y[..., 1::2, :], y[..., ::2, :] - y[..., 1::2, :]), dim=-1)
    y = y.squeeze(-2)

        
    # Arbitrary order
    if type(order)==list:
        y = sequency_perm_torch(y,order)
    # Sequency (aka Walsh) order
    elif order: 
        y = sequency_perm_torch(y)
    # Hadamard order, otherwise
    return y


def fwalsh_G_torch(x,ind=True): 
    """Fast Walsh G-transform of x

    Args:
        :attr:`x` (torch.tensor):  input signal with shape `(*, n)`. `n`+1 
                                should be a power of two.
        :attr:`ind` (bool, optional): True for sequency (default)
        :attr:`ind` (list, optional): permutation indices. This is faster than
                                True when repeating the sequency-ordered 
                                transform multilple times.  

    Returns:
        torch.tensor: S-transformed signal with shape `(*, n)`.
    
    Example 1:
        Walsh-ordered G-transform of a signal of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch        
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1])
        >>> s = wh.fwalsh_G_torch(x)
        >>> print(s)
    
    Example 2:
        Permuted fast G-transform
        
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch        
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1])
        >>> ind = [0, 1, 3, 2, 7, 4, 5, 6]
        >>> y = wh.fwalsh_G_torch(x, ind)
        >>> print(y)
        
    Example 3:
        Comparison with the numpy transform
        
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1])
        >>> y_np = wh.fwalsh_G(x)
        >>> x_torch = torch.from_numpy(x).to(torch.device('cuda:0'))
        >>> y_torch = wh.fwalsh_G_torch(x_torch)
        >>> print(y_np)
        >>> print(y_torch)
    
    Example 3:
        Repeating the Walsh-ordered G-transform using input indices is faster  
        
        >>> import timeit
        >>> x = torch.rand(512,2**12-1, device=torch.device('cuda:0'))
        >>> t = timeit.timeit(lambda: wh.fwalsh_G_torch(x), number=10)
        >>> print(f"No indices as inputs (10x): {t:.3f} seconds")
        >>> ind = wh.sequency_perm_ind(x.shape[-1]+1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_G_torch(x,ind), number=10)
        >>> print(f"With indices as inputs (10x): {t:.3f} seconds")
        
    """
    # Concatenate with zeros
    z = torch.zeros(x.shape[:-1], device = x.device) 
    z = z[..., None]
    x = torch.cat((z, x), dim=-1)
    # Fast Hadamard transform
    y = fwht_torch(x,ind)
    # Remove 0th entries
    y = y[...,1:]
    return y

def fwalsh_S_torch(x,ind=True): 
    """Fast Walsh S-transform of x

    Args:
        :attr:`x` (torch.tensor):  input signal with shape `(*, n)`. `n`+1
                            should be a power of two.
        ind (bool, optional): True for sequency (default)
        ind (list, optional): permutation indices. This is faster than True 
                            when repeating the sequency-ordered transform 
                            multilple times.  

    Returns:
        torch.tensor: -by-n S-transformed signal
    
    Example 1:
        Walsh-ordered S-transform of a signal of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch      
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1])
        >>> s = wh.fwalsh_S_torch(x)
        >>> print(s)
         
    
    Example 2:
        Repeating the Walsh-ordered S-transform using input indices is faster  
        
        >>> import timeit
        >>> import torch
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = torch.rand(512, 2**14-1, device=torch.device('cuda:0'))
        >>> t = timeit.timeit(lambda: wh.fwalsh_S_torch(x), number=10)
        >>> print(f"No indices as inputs (10x): {t:.3f} seconds")
        >>> ind = wh.sequency_perm_ind(x.shape[-1]+1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_S_torch(x,ind), number=10)
        >>> print(f"With indices as inputs (10x): {t:.3f} seconds")
    """
    j = torch.sum(x, -1, keepdim=True)
    s = fwalsh_G_torch(x, ind)
    s = (j-s)/2
    return s

def fwalsh2_S_torch(X,ind=True): #not validated!
    """Fast Walsh S-transform of X in "2D"

    Args:
        :attr:`X` (torch.tensor):  input image with shape `(*, n, n)`. `n`**2 
                                    should be a power of two.
        :attr:`ind` (bool, optional): True for sequency (default)
        :attr:`ind` (list, optional): permutation indices.

    Returns:
        torch.tensor: S-transformed signal with shape `(*, n, n)`
    
    Examples:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> X = torch.tensor([[1, 3, 0, 8],[7, 5, 1, 2],[2, 3, 6, 1],[4, 6, 8, 0]])
        >>> wh.fwalsh2_S_torch(X)
        
    Example 2:
        Repeating the Walsh-ordered S-transform using input indices is faster  
        
        >>> import timeit
        >>> import torch
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> X = torch.rand(128, 2**6, 2**6, device=torch.device('cuda:0'))
        >>> t = timeit.timeit(lambda: wh.fwalsh2_S_torch(X), number=10)
        >>> print(f"No indices as inputs (10x): {t:.3f} seconds")
        >>> ind = wh.sequency_perm_ind(X.shape[-1]*X.shape[-2])
        >>> t = timeit.timeit(lambda: wh.fwalsh2_S_torch(X,ind), number=10)
        >>> print(f"With indices as inputs (10x): {t:.3f} seconds")
        
    """    
    x = walsh2_S_unfold_torch(X)
    s = fwalsh_S_torch(x,ind)
    S = walsh2_S_fold_torch(s)
    return S

def walsh2_S_fold_torch(x):
    """Fold a signal to get a "2d" s-transformed representation 
    
    Note: the top left (first) pixel is arbitrarily set to zero

    Args:
        :attr:`x` (torch.tensor): input signal with shape `(*, n)`. n is 
        such that n+1 = N*N, where N is a power of two. n = 2**(2b) - 1, 
        where b is an integer. 

    Returns:
        torch.tensor: output matrix with shape `(*, N, N)`
        
    Example 1:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch
        >>> S = wh.walsh2_S_matrix(4)
        >>> X = torch.from_numpy(S[2:4,:])
        >>> Y = wh.walsh2_S_fold_torch(X)
        >>> print(Y)
    """
    
    # N and n should be consistent with doc
    N = x.shape[-1]
    n = (N+1)**0.5
    assert math.log2(n)%1 == 0, f"N+1 = n*n, where n={n:.2f} must be a power of two"
    n = int(n)

    # Concatenate with zeros
    z = torch.zeros(x.shape[:-1], device = x.device) 
    z = z[..., None]
    X = torch.cat((z, x), dim=-1) # Extra memory allocated here

    # Reshape to get images
    new_shape = torch.Size([*x.shape[:-1], n, n])
    X = X.view(new_shape)
    return X

def walsh2_S_unfold_torch(X):
    """Unfold a signal from a "2d" s-transformed representation 
    
    Note: Return a view of X

    Args:
        :attr:`X` (torch.tensor): input image with shape `(*, n,n)`.

    Returns:
        output signal with shape `(*, n*n-1)`
        
    Example 1:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch
        >>> X = torch.tensor([[1, 3, 0, 8],[7, 5, 1, 2]])
        >>> x = wh.walsh2_S_unfold_torch(X)
        >>> print(X)
        >>> print(x)
        
    Example 2:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch
        >>> X = torch.randint(10,(3,4,4))
        >>> x = wh.walsh2_S_unfold_torch(X)
        >>> print(X)
        >>> print(x)
    
    """
    x = X.view(*(X.size()[:-2]),-1)
    x = x[...,1:]
    return x


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