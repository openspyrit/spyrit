#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Walsh-ordered Hadamard tranforms.

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

# __author__ = "One solo developer"
__authors__ = ["Sebastien Crombez", "Nicolas Ducros"]
__contact__ = "nicolas.ducros@creatis.insa-lyon.fr"
# __copyright__ = "Copyright $YEAR, $COMPANY_NAME"
# __credits__ = ["One developer", "And another one", "etc"]
__date__ = "2020/01/15"
__deprecated__ = False
# __email__ =  "nicolas.ducros@creatis.insa-lyon.fr"
__license__ = "GPLv3"
__maintainer__ = "developer"
__status__ = "Development"
# __version__ = "0.0.1"

import warnings

import math
import torch
import numpy as np

import spyrit.core.torch as spytorch
from sympy.combinatorics.graycode import GrayCode


# ------------------------------------------------------------------------------
# -- To generate sequency (aka Walsh) order --------------------------------------
# ------------------------------------------------------------------------------
def b2_to_b10(l):
    r"""Convert a list of numbers in base 2 to base 10

    Args:
        :math:`l` (list[str]): base2 numbers.

    Returns:
        list[int]: base10 numbers

    """
    N = len(l)
    for i in range(N):
        l[i] = int(l[i], 2)
    return l


def perm_matrix_from_ind(l):  # generate a matrix of zero and ones from list of index
    N = len(l)
    P = np.zeros((N, N))
    for i in range(N):
        P[i, l[i]] = 1
    return P


def gray_code_permutation(n):  # Generate the N grey code
    N = int(math.log(n, 2))
    graycode = GrayCode(N)
    graycode_list = list(graycode.generate_gray())
    return perm_matrix_from_ind(b2_to_b10((graycode_list)))


def gray_code_list(n):  # Generate the N grey code permutation matrix
    N = int(math.log(n, 2))
    graycode = GrayCode(N)
    graycode_list = list(graycode.generate_gray())
    return b2_to_b10(graycode_list)


def bit_reverse_traverse(a):  # internet function to generate bit reverse
    n = a.shape[0]
    assert not n & (n - 1), "n must be a power of 2"  # assert n is power of 2

    if n == 1:
        yield a[0]
    else:
        even_index = np.arange(n // 2) * 2
        odd_index = np.arange(n // 2) * 2 + 1
        for even in bit_reverse_traverse(a[even_index]):
            yield even
        for odd in bit_reverse_traverse(a[odd_index]):
            yield odd


def get_bit_reversed_list(l):  # from the internet
    n = len(l)
    indexs = np.arange(n)
    b = []
    for i in bit_reverse_traverse(indexs):
        b.append(l[i])

    return b


def bit_reversed_matrix(n):  # internet function to generate bit reverse
    br = bit_reversed_list(n)
    return perm_matrix_from_ind(br)


def bit_reversed_list(n):
    br = get_bit_reversed_list([k for k in range(n)])
    return br


def sequency_perm(X, ind=None):
    r"""Permute the last dimension of a tensor. By defaults this allows the sequency order to be obtained from the natural order.

    Args:
        :attr:`X` (np.ndarray): input of shape (*,n).

        :attr:`ind` : list of index length n. Defaults to indices to get sequency order.

    Returns:
        np.ndarray: output of shape (*,n)

    Example :
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1, -2])
        >>> x = x[None, None, :]
        >>> x = wh.sequency_perm(x)
        >>> print(x)
        tensor([[[ 1,  7,  1,  0, -1, -2,  5,  3]]])
        >>> print(x.shape)
        torch.Size([1, 1, 8])
    """
    if ind is None:
        ind = sequency_perm_ind(X.shape[-1])

    Y = X[..., ind]
    return Y


def sequency_perm_matrix(n):
    r"""Return permutation matrix to get sequency from the natural order

    Args:
        :attr:`n` (int): Order of the matrix, a power of two.

    Returns:
        np.ndarray: A n-by-n permutation matrix

    Examples:
        Permutation matrix of order 8

        >>> print(sequency_perm_matrix(8))
        [[1. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 1. 0.]
         [0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 1.]
         [0. 0. 0. 0. 0. 1. 0. 0.]
         [0. 1. 0. 0. 0. 0. 0. 0.]]
    """
    BR = bit_reversed_matrix(n)
    GC = gray_code_permutation(n)
    return GC @ BR


def sequency_perm_ind(n):
    r"""Return permutation indices to get sequency from the natural order

    Args:
        :attr:`n` (int): Order of the matrix, a power of two.

    Returns:
        list:

    Examples:
        Permutation indices to get a Walsh matrix of order 8

        >>> print(sequency_perm_ind(8))
        [0, 4, 6, 2, 3, 7, 5, 1]
    """
    perm_br = bit_reversed_list(n)
    perm_gc = gray_code_list(n)
    perm = [perm_br[perm_gc[k]] for k in range(n)]

    return perm


# ------------------------------------------------------------------------------
# -- 1D Walsh/Hamadard matrix and transforms -----------------------------------
# ------------------------------------------------------------------------------
def walsh_matrix(n):
    """Return 1D Walsh-ordered Hadamard transform matrix

    Args:
        n (int): Order of the matrix, a power of two.

    Returns:
        np.ndarray: A n-by-n array

    Examples:
        Walsh-ordered Hadamard matrix of order 8

        >>> print(walsh_matrix(8))
        [[ 1  1  1  1  1  1  1  1]
         [ 1  1  1  1 -1 -1 -1 -1]
         [ 1  1 -1 -1 -1 -1  1  1]
         [ 1  1 -1 -1  1  1 -1 -1]
         [ 1 -1 -1  1  1 -1 -1  1]
         [ 1 -1 -1  1 -1  1  1 -1]
         [ 1 -1  1 -1 -1  1 -1  1]
         [ 1 -1  1 -1  1 -1  1 -1]]
    """
    # P = sequency_perm_matrix(n)
    # H = hadamard(n)                   # old way with matrix multiplication
    # return P @ H

    # check that the input is a power of 2
    spytorch.assert_power_of_2(n, raise_error=True)

    # define recursive function
    def recursive_walsh(k):
        if k >= 3:
            j = k // 2
            a = recursive_walsh(j)
            out = np.empty((k, k), dtype=int)

            # generate upper half of the matrix
            out[:j, ::2] = a
            out[:j, 1::2] = a
            # by symmetry, fill in lower left corner
            out[j:, :j] = out[:j, j:].T
            # fill in lower right corner
            alternate = np.tile([1, -1], j // 2)
            out[j:, j:] = alternate * out[:j, j:][::-1, :]
            return out

        elif k == 2:
            return np.array([[1, 1], [1, -1]])
        else:
            return np.array[[1]]

    return recursive_walsh(n)


def fwht(x, order=True):
    """Fast Walsh-Hadamard transform of x.

    Due to the inherent numerical instability of the Hadamard transform
    (lots of additions and subtractions), it is recommended to use float64
    whenever possible.

    This is computed using Amit Portnoy's algorithm available in the package
    `hadamard-transform` at https://github.com/amitport/hadamard-transform.

    Args:
        x (np.ndarray): batched input signal of shape :math:`(* , n)`, where
        :math:`n` is a power of two. The transform applies to the last dimension.

        order (bool | list, optional): True for sequency (default), False for
        natural. It is also possible to provide a list of permutation indices.

    Returns:
        np.ndarray: transformed signal of shape :math:`(* , n)`

    Example:
        Example 1: Fast sequency-ordered (i.e., Walsh) Hadamard transform

        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1, -2])
        >>> y = wh.fwht(x)
        >>> print(y)
        [14 -8 -8 18 -4 -2 -6  4]

        Example 2: Fast Hadamard transform

        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1, -2])
        >>> y = wh.fwht(x, False)
        >>> print(y)
        [14  4 18 -4 -8 -6 -8 -2]

        Example 3: Permuted fast Hadamard transform

        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1, -2])
        >>> ind = [1, 0, 3, 2, 7, 4, 5, 6]
        >>> y = wh.fwht(x, ind)
        >>> print(y)
        [ 4 14 -4 18 -2 -8 -6 -8]

        Example 4: Comparison with Walsh-Hadamard transform via matrix-vector product

        >>> from spyrit.misc.walsh_hadamard import fwht, walsh_matrix
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1, -2])
        >>> y1 = fwht(x)
        >>> H = walsh_matrix(8)
        >>> y2 = H @ x
        >>> print(f"Diff = {np.linalg.norm(y1-y2)}")
        Diff = 0.0

        Example 5: Comparison with the fast Walsh-Hadamard transform from sympy

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import sympy as sp
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1, -2])
        >>> y1 = wh.fwht(x)
        >>> y2 = sp.fwht(x)
        >>> y3 = wh.sequency_perm(np.array(y2,dtype=x.dtype))
        >>> print(y1)
        [14 -8 -8 18 -4 -2 -6  4]
        >>> print(f"Diff = {np.linalg.norm(y1-y3)}")
        Diff = 0.0

        Example 6: Computation times for 100 signals of length 2**12

        >>> import timeit
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.random.rand(100, 2**12)
        >>> t = timeit.timeit(lambda: wh.fwht(x), number=100)
        >>> print(f"Fast Walsh transform, no ind (100x): {t:.3f} seconds")
        Fast Walsh transform, no ind (100x): ... seconds
        >>> t = timeit.timeit(lambda: wh.fwht(x,False), number=100)
        >>> print(f"Fast Hadamard transform (100x): {t:.3f} seconds")
        Fast Hadamard transform (100x): ... seconds
        >>> ind = wh.sequency_perm_ind(x.shape[-1])
        >>> t = timeit.timeit(lambda: wh.fwht(x,ind), number=100)
        >>> print(f"Fast Walsh transform, with ind (100x): {t:.3f} seconds")
        Fast Walsh transform, with ind (100x): ... seconds

        >>> import sympy as sp
        >>> t = timeit.timeit(lambda: sp.fwht(x[0,:]), number=10)
        >>> print(f"Fast Hadamard transform from sympy (only 1 signal, 10x): {t:.3f} seconds")
        Fast Hadamard transform from sympy (only 1 signal, 10x): ... seconds
    """

    ###########################################################################
    # MIT License

    # Copyright (c) 2022 Amit Portnoy

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    ###########################################################################

    # BELOW IS ADAPTED CODE FROM AMIT PORTNOY
    # ---------------------------------------
    original_shape = x.shape

    # create batch if x is 1D
    if len(original_shape) == 1:
        x = x.reshape(1, -1)  # shape (1, n)

    *batch, d = x.shape  # batch is tuple and d is int
    spytorch.assert_power_of_2(d, raise_error=True)

    h = 2

    while h <= d:

        x = x.reshape(*batch, d // h, h)
        half1, half2 = np.split(x, 2, axis=-1)

        # do we want sequency-ordered transform ?
        # two lines below not from Amit Portnoy
        if order == True:
            half2[..., 1::2] *= -1  # not from Amit Portnoy
            x = np.stack((half1 + half2, half1 - half2), axis=-1)  # not from AP
        else:
            x = np.concatenate((half1 + half2, half1 - half2), axis=-1)

        h *= 2

    x = x.reshape(original_shape)
    # ---------------------------------------
    # END OF ADAPTED CODE FROM AMIT PORTNOY

    # Arbitrary order
    if type(order) == list:
        x = sequency_perm(x, order)

    return x


# ------------------------------------------------------------------------------
# -- G-matrix and G-transforms -------------------------------------------------
# ------------------------------------------------------------------------------
def walsh_G_matrix(n, H=None):
    """Return Walsh-ordered Hadamard G-matrix of order n

    Args:
        n (int): Matrix order. n+1 should be a power of two.
        H (np.ndarray, optional): Hadamard matrix of order n+1.

    Returns:
        np.ndarray: G-matrix of shape `(n,n)`

    Examples:
        Walsh-ordered Hadamard G-matrix of order 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> print(wh.walsh_G_matrix(7))
        [[ 1  1  1 -1 -1 -1 -1]
         [ 1 -1 -1 -1 -1  1  1]
         [ 1 -1 -1  1  1 -1 -1]
         [-1 -1  1  1 -1 -1  1]
         [-1 -1  1 -1  1  1 -1]
         [-1  1 -1 -1  1 -1  1]
         [-1  1 -1  1 -1  1 -1]]
    """
    assert math.log2(n + 1) % 1 == 0, f"{n}+1 must be a power of two"

    if H is None:
        H = walsh_matrix(n + 1)
    return H[1:, 1:]


def walsh_G(x, G=None):
    """Return the Walsh G-transform of the signal x

    Args:
        :attr:`x` (np.ndarray): signals of shape `(*,n)`, where `n+1` must be a power of two.

    Returns:
        np.ndarray: G-transformed signal of shape `(*,n)`.

    Examples:
        Example 1: Walsh-ordered G-transform of a signal of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1])
        >>> s = wh.walsh_G(x)
        >>> print(s)
        [ -8  -2  -2 -16   8   6  -2]

        Example 2: Walsh-ordered G-transform of two signals of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([[1, 3, 0, -1, 7, 5, 1],[2, 1, -1, 0, 4, 5, 3]])
        >>> s = wh.walsh_G(x)
        >>> print(s)
        [[ -8  -2  -2 -16   8   6  -2]
         [-10   6  -2 -10   2   2  -2]]
    """
    if G is None:
        G = walsh_G_matrix(x.shape[-1])

    return x @ G  # Matris G is symmetric


def fwalsh_G(x, ind=True):
    r"""Fast Walsh G-transform of the signal x

    Args:
        :attr:`x` (np.ndarray): signals of shape `(*,n)`, where `n+1` must be a power of two.

        :attr:`ind` (bool, optional): `True` for sequency (default).

        :attr:`ind` (list, optional): permutation indices. This is faster than
        `True` when repeating the sequency-ordered transform multilple times.

    Returns:
        np.ndarray: G-transformed signal of shape `(*,n)`.

    Example:
        Example 1: Walsh-ordered G-transform of a signal of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import numpy as np
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1])
        >>> s = wh.fwalsh_G(x)
        >>> print(s)
        [ -8  -2  -2 -16   8   6  -2]

        Example 2: Permuted fast G-transform

        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1])
        >>> ind = [1, 0, 3, 2, 7, 4, 5, 6]
        >>> y = wh.fwalsh_G(x, ind)
        >>> print(y)
        [ 16 -16  -2   8  -8   6  -2]

        Example 3: Repeating the Walsh-ordered G-transform using input indices is faster

        >>> import timeit
        >>> x = np.random.rand(100,2**12-1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_G(x), number=100)
        >>> print(f"No indices as inputs (10x): {t:.3f} seconds")
        No indices as inputs (10x): ... seconds
        >>> ind = wh.sequency_perm_ind(x.shape[-1]+1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_G(x,ind), number=100)
        >>> print(f"With indices as inputs (10x): {t:.3f} seconds")
        With indices as inputs (10x): ... seconds

        Example 4: Comparison with G-transform via matrix-vector product

        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([3, 0, -1, 7, 5, 1, -2])
        >>> y1 = wh.fwalsh_G(x)
        >>> y2 = wh.walsh_G(x)
        >>> print(f"Diff = {np.linalg.norm(y1-y2)}")
        Diff = 0.0
    """
    zeros_arr = np.zeros((*x.shape[:-1], 1), dtype=x.dtype)
    x = np.concatenate((zeros_arr, x), axis=-1)
    y = fwht(x, ind)
    y = y[..., 1:]
    return y


# ------------------------------------------------------------------------------
# -- S-matrix and S-transforms -------------------------------------------------
# ------------------------------------------------------------------------------
def walsh_S_matrix(n, H=None):
    """Return Walsh-ordered Hadamard S-matrix of order n

    Args:
        n (int): Matrix order. n+1 should be a power of two.
        H (np.ndarray, optional): Hadamard matrix of order n+1.

    Returns:
        np.ndarray: S-matrix of shape `(n,n)`

    Examples:
        Walsh-ordered Hadamard S-matrix of order 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> print(wh.walsh_S_matrix(7))
        [[0. 0. 0. 1. 1. 1. 1.]
         [0. 1. 1. 1. 1. 0. 0.]
         [0. 1. 1. 0. 0. 1. 1.]
         [1. 1. 0. 0. 1. 1. 0.]
         [1. 1. 0. 1. 0. 0. 1.]
         [1. 0. 1. 1. 0. 1. 0.]
         [1. 0. 1. 0. 1. 0. 1.]]
    """
    return (1 - walsh_G_matrix(n, H)) / 2


def iwalsh_S_matrix(n, H=None):
    """Return the inverse Walsh S-matrix of order n.

    Args:
        :attr:`x` (np.ndarray): signals of shape `(*,n)`, where `n+1` must be a power of two.

        :attr:`H` (np.ndarray, optional): Hadamard matrix of order n+1.

    Returns:
        np.ndarray: Inverse S-matrix of shape `(n,n)`.

    Examples:
        Example 1: Inverse Walsh S-matrix of order 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> print(wh.iwalsh_S_matrix(7))
        [[-0.25 -0.25 -0.25  0.25  0.25  0.25  0.25]
         [-0.25  0.25  0.25  0.25  0.25 -0.25 -0.25]
         [-0.25  0.25  0.25 -0.25 -0.25  0.25  0.25]
         [ 0.25  0.25 -0.25 -0.25  0.25  0.25 -0.25]
         [ 0.25  0.25 -0.25  0.25 -0.25 -0.25  0.25]
         [ 0.25 -0.25  0.25  0.25 -0.25  0.25 -0.25]
         [ 0.25 -0.25  0.25 -0.25  0.25 -0.25  0.25]]

        Example 2: Check the inverse of the Walsh S-matrix of order 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> print(wh.iwalsh_S_matrix(7) @ wh.walsh_S_matrix(7))
        [[1. 0. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 0. 1.]]
    """
    return -2 * walsh_G_matrix(n, H) / (n + 1)


def walsh_S(x, S=None):
    """Return the Walsh S-transform of the signal x

    Args:
        :attr:`x` (np.ndarray): signals of shape `(*,n)`, where `n+1` must be a power of two.

        :attr:`H` (np.ndarray, optional): S-matrix of order n.

    Returns:
        np.ndarray: S-transformed signal of shape `(*,n)`.

    Examples:
        Example 1: Walsh-ordered S-transform of a signal of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1])
        >>> s = wh.walsh_S(x)
        >>> print(s)
        [12.  9.  9. 16.  4.  5.  9.]

        Example 2: Walsh-ordered S-transform of two signals of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([[1, 3, 0, -1, 7, 5, 1],[2, 1, -1, 0, 4, 5, 3]])
        >>> s = wh.walsh_S(x)
        >>> print(s)
        [[12.  9.  9. 16.  4.  5.  9.]
         [12.  4.  8. 12.  6.  6.  8.]]
    """
    if S is None:
        S = walsh_S_matrix(x.shape[-1])

    return x @ S  # Matris S is symmetric


def iwalsh_S(s, T=None):
    """Return the inverse Walsh S-transform of the signal x

    Args:
        :attr:`x` (np.ndarray): signals of shape `(*,n)`, where `n+1` must be a power of two.

        :attr:`H` (np.ndarray, optional): Inverse S-matrix of order n.

    Returns:
        np.ndarray: S-transformed signal of shape `(*,n)`.

    Examples:
        Example 1: Inverse Walsh-ordered S-transform of a signal of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> s = np.array([12, 9, 9, 16, 4, 5, 9])
        >>> x = wh.iwalsh_S(s)
        >>> print(x)
        [ 1.  3.  0. -1.  7.  5.  1.]

        Example 2: Walsh-ordered S-transform of two signals of length 7

        >>> s = np.array([[12, 9, 9, 16, 4, 5, 9],[12, 4, 8, 12, 6, 6, 8]])
        >>> x = wh.iwalsh_S(s)
        >>> print(x)
        [[ 1.  3.  0. -1.  7.  5.  1.]
         [ 2.  1. -1.  0.  4.  5.  3.]]

        Example 3: Foroward and inverse S-transform of two signals of length 4095

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import numpy as np
        >>> x = np.random.rand(2,4095)
        >>> s = wh.walsh_S(x)
        >>> y = wh.iwalsh_S(s)
        >>> err = np.linalg.norm(1-y/x)
        >>> print(f'Error: {err}')
        Error: ...e-...
        >>> print(err < 1e-5)
        True
    """
    if T is None:
        T = iwalsh_S_matrix(s.shape[-1])

    return s @ T  # T is symmetric


def fwalsh_S(x, ind=True):
    r"""Fast Walsh S-transform of x

    Args:
        :attr:`x` (np.ndarray): n-by-1 signal. n+1 should be a power of two.

        :attr:`ind` (bool, optional): True for sequency (default)

        :attr:`ind` (list, optional): permutation indices. This is faster than True
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
        [12.  9.  9. 16.  4.  5.  9.]

    Example 2:
        Permuted fast S-transform

        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1])
        >>> ind = [1, 0, 3, 2, 7, 4, 5, 6]
        >>> y = wh.fwalsh_S(x, ind)
        >>> print(y)
        [ 0. 16.  9.  4. 12.  5.  9.]

    Example 3:
        Repeating the Walsh-ordered S-transform using input indices is faster

        >>> import timeit
        >>> x = np.random.rand(2**12-1,1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_S(x), number=100)
        >>> print(f"No indices as inputs (10x): {t:.3f} seconds")
        No indices as inputs (10x): ... seconds
        >>> ind = wh.sequency_perm_ind(x.shape[-1]+1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_S(x,ind), number=100)
        >>> print(f"With indices as inputs (10x): {t:.3f} seconds")
        With indices as inputs (10x): ... seconds

    Example 4:
        Comparison with S-transform via matrix-vector product

        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([3, 0, -1, 7, 5, 1, -2])
        >>> y1 = wh.fwalsh_S(x)
        >>> y2 = wh.walsh_S(x)
        >>> print(f"Diff = {np.linalg.norm(y1-y2)}")
        Diff = 0.0

    Example 5:
        Computation times for a signal of length 2**12-1

        >>> import timeit
        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.random.rand(2**14-1,1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_S(x), number=100)
        >>> print(f"Fast transform, no ind (10x): {t:.3f} seconds")
        Fast transform, no ind (10x): ... seconds
        >>> ind = wh.sequency_perm_ind(x.shape[-1]+1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_S(x,ind), number=100)
        >>> print(f"Fast transform, with ind (10x): {t:.3f} seconds")
        Fast transform, with ind (10x): ... seconds
        >>> S = wh.walsh_S_matrix(x.shape[-1])
        >>> t = timeit.timeit(lambda: wh.walsh_S(x,S), number=100)
        >>> print(f"Naive transform (10x): {t:.3f} seconds")
        Naive transform (10x): ... seconds
    """
    j = x.sum()
    s = fwalsh_G(x, ind)
    s = (j - s) / 2
    return s


def ifwalsh_S(s, ind=True):
    r"""Inverse fast Walsh S-transform of s

    Args:
        :attr:`x` (np.ndarray): n-by-1 signal. n+1 should be a power of two.

        :attr:`ind` (bool, optional): True for sequency (default).

        :attr:`ind` (list, optional): permutation indices. This is faster than True
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
        signal: [ 1  3  0 -1  7  5  1]
        >>> s = wh.fwalsh_S(x)
        >>> print(f"s-transform: {s}")
        s-transform: [12.  9.  9. 16.  4.  5.  9.]
        >>> y = wh.ifwalsh_S(s)
        >>> print(f"inverse s-transform: {y}")
        inverse s-transform: [ 1.  3. -0. -1.  7.  5.  1.]

    """
    x = -2 / (len(s) + 1) * fwalsh_G(s, ind)

    return x


# ------------------------------------------------------------------------------
# -- 2D transforms -------------------------------------------------------------
# ------------------------------------------------------------------------------
def walsh_matrix_2d(n):
    """Return Walsh-ordered Hadamard matrix in 2D

    Args:
        n (int): Order of the matrix, which must be a power of two.

    Returns:
        H (np.ndarray): A n*n-by-n*n matrix
    """
    H1d = walsh_matrix(n)
    return np.kron(H1d, H1d)


def walsh2(X, H=None):
    r"""Return 2D Walsh-ordered Hadamard transform of an image :math:`H^\top X H`

    Args:
        X (np.ndarray): image as a 2d array. The size is a power of two.
        H (np.ndarray, optional): 1D Walsh-ordered Hadamard transformation matrix

    Returns:
        np.ndarray: Hadamard transformed image as a 2D array.
    """
    if H is None:
        H = walsh_matrix(len(X))
    return np.dot(np.dot(H, X), H)


def iwalsh2(X, H=None):
    """Return 2D inverse Walsh-ordered Hadamard transform of an image

    Args:
        X (np.ndarray): Image as a 2D array. The image is square and its size is a power of two.
        H (np.ndarray, optional): 1D inverse Walsh-ordered Hadamard transformation matrix

    Returns:
        np.ndarray: Inverse Hadamard transformed image as a 2D array.
    """
    if H is None:
        H = walsh_matrix(len(X))
    return walsh2(X, H) / len(X) ** 2


def fwalsh2_S(X, ind=True):
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
        array([[ 0., 30., 27., 33.],
               [37., 33., 26., 26.],
               [32., 16., 29., 25.],
               [35., 33., 38., 28.]])

    """
    x = walsh2_S_unfold(X)
    s = fwalsh_S(x, ind)
    S = walsh2_S_fold(s)
    return S


def ifwalsh2_S(Y, ind=True):
    r"""Inverse Fast Walsh S-transform of Y in "2D"

    Args:
        :attr:`Y` (np.ndarray): n-by-n signal. n**2 should be a power of two.

        :attr:`ind` (bool, optional): True for sequency (default)

        :attr:`ind` (list, optional): permutation indices.

    Returns:
        np.ndarray: n-by-1 S-transformed signal

    Examples:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> X = np.array([[1, 3, 0, 8],[7, 5, 1, 2],[2, 3, 6, 1],[4, 6, 8, 0]])
        >>> print(f"image:\n {X}")
        image:
         [[1 3 0 8]
         [7 5 1 2]
         [2 3 6 1]
         [4 6 8 0]]
        >>> Y = wh.fwalsh2_S(X)
        >>> print(f"s-transform:\n {Y}")
        s-transform:
         [[ 0. 30. 27. 33.]
         [37. 33. 26. 26.]
         [32. 16. 29. 25.]
         [35. 33. 38. 28.]]
        >>> Z = wh.ifwalsh2_S(Y)
        >>> print(f"inverse s-transform:\n {Z}")
        inverse s-transform:
         [[ 0.  3. -0.  8.]
         [ 7.  5.  1.  2.]
         [ 2.  3.  6.  1.]
         [ 4.  6.  8. -0.]]

        Note that the first pixel is not meaningful and arbitrily set to 0.

    """
    x = walsh2_S_unfold(Y)
    s = ifwalsh_S(x, ind)
    S = walsh2_S_fold(s)
    return S


def walsh2_S(X, S=None):
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
        array([[ 0., 30., 27., 33.],
               [37., 33., 26., 26.],
               [32., 16., 29., 25.],
               [35., 33., 38., 28.]])

    """
    x = walsh2_S_unfold(X)
    y = walsh_S(x, S)
    Y = walsh2_S_fold(y)
    return Y


def iwalsh2_S(Y, T=None):
    r"""Inverse Fast Walsh S-transform of Y in "2D"

    Args:
        :attr:`Y` (np.ndarray): n-by-n signal. n**2 should be a power of two.

        :attr:`T` (np.ndarray): Inverse S-matrix

    Returns:
        np.ndarray: n-by-1 S-transformed signal

    Examples:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import numpy as np
        >>> X = np.array([[1, 3, 0, 8],[7, 5, 1, 2],[2, 3, 6, 1],[4, 6, 8, 0]])
        >>> print(f"image:\n {X}")
        image:
         [[1 3 0 8]
         [7 5 1 2]
         [2 3 6 1]
         [4 6 8 0]]
        >>> Y = wh.walsh2_S(X)
        >>> print(f"s-transform:\n {Y}")
        s-transform:
         [[ 0. 30. 27. 33.]
         [37. 33. 26. 26.]
         [32. 16. 29. 25.]
         [35. 33. 38. 28.]]
        >>> Z = wh.iwalsh2_S(Y)
        >>> print(f"inverse s-transform:\n {Z}")
        inverse s-transform:
         [[0. 3. 0. 8.]
         [7. 5. 1. 2.]
         [2. 3. 6. 1.]
         [4. 6. 8. 0.]]

        Note that the first pixel is not meaningful and arbitrily set to 0.

    """
    y = walsh2_S_unfold(Y)
    x = iwalsh_S(y, T)
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

    H = walsh_matrix_2d(n)
    S = walsh_S_matrix(n**2 - 1, H)
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
        [[0. 1. 0. 1.]
         [0. 1. 0. 1.]
         [0. 1. 0. 1.]
         [0. 1. 0. 1.]]
    """

    n = (len(x) + 1) ** 0.5
    assert math.log2(n) % 1 == 0, f"N+1 = n*n, where n={n:.2f} must be a power of two"

    n = int(n)
    X = np.insert(x, 0, 0)
    X = np.reshape(X, (n, n))
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
        array([3, 0, 8, 7, 5, 1, 2])

    """
    x = X.ravel()
    x = x[1:]
    return x


# ------------------------------------------------------------------------------
# -- PyTorch functions ---------------------------------------------------------
# ------------------------------------------------------------------------------
def fwht_torch(x, order=True):
    """Deprecated function. Please use spyrit.core.torch.fwht instead."""
    raise NotImplementedError(
        "This function is deprecated. Please call spyrit.core.torch.fwht instead."
    )


def fwalsh_G_torch(x, ind=True):
    r"""Fast Walsh G-transform of x

    Args:
        :attr:`x` (torch.tensor):  input signal with shape :math:`(*, n)`. :math:`n`+1 should be a power of two.

        :attr:`ind` (bool, optional): True for sequency (default)

        :attr:`ind` (list, optional): permutation indices. This is faster than True when repeating the sequency-ordered transform multilple times.

    Returns:
        torch.tensor: S-transformed signal with shape :math:`(*, n)`.

    Example 1:
        Walsh-ordered G-transform of a signal of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1])
        >>> s = wh.fwalsh_G_torch(x)
        >>> print(s)
        tensor([ -8.,  -2.,  -2., -16.,   8.,   6.,  -2.])

    Example 2:
        Permuted fast G-transform

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch
        >>> x = torch.tensor([1, 3, 0, -1, 7, 5, 1])
        >>> ind = [0, 1, 3, 2, 7, 4, 5, 6]
        >>> y = wh.fwalsh_G_torch(x, ind)
        >>> print(y)
        tensor([ -2., -16.,  -2.,   8.,  -8.,   6.,  -2.])

    Example 3:
        Comparison with the numpy transform

        >>> import numpy as np
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = np.array([1, 3, 0, -1, 7, 5, 1])
        >>> y_np = wh.fwalsh_G(x)
        >>> x_torch = torch.from_numpy(x).to(torch.device('cpu'))
        >>> y_torch = wh.fwalsh_G_torch(x_torch)
        >>> print(y_np)
        [ -8  -2  -2 -16   8   6  -2]
        >>> print(y_torch)
        tensor([ -8.,  -2.,  -2., -16.,   8.,   6.,  -2.])

    Example 3:
        Repeating the Walsh-ordered G-transform using input indices is faster

        >>> import timeit
        >>> x = torch.rand(512,2**12-1, device=torch.device('cpu'))
        >>> t = timeit.timeit(lambda: wh.fwalsh_G_torch(x), number=10)
        >>> print(f"No indices as inputs (10x): {t:.3f} seconds")
        No indices as inputs (10x): ... seconds
        >>> ind = wh.sequency_perm_ind(x.shape[-1]+1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_G_torch(x,ind), number=10)
        >>> print(f"With indices as inputs (10x): {t:.3f} seconds")
        With indices as inputs (10x): ... seconds

    """
    # Concatenate with zeros
    z = torch.zeros(x.shape[:-1], device=x.device)
    z = z[..., None]
    x = torch.cat((z, x), dim=-1)
    # Fast Hadamard transform
    y = spytorch.fwht(x, ind, -1)
    # Remove 0th entries
    y = y[..., 1:]
    return y


def fwalsh_S_torch(x, ind=True):
    r"""Fast Walsh S-transform of x

    Args:
        :attr:`x` (torch.tensor):  input signal with shape `(*, n)`. `n`+1
                            should be a power of two.

        :attr:`ind` (bool, optional): True for sequency (default)

        :attr:`ind` (list, optional): permutation indices. This is faster than True
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
        tensor([12.,  9.,  9., 16.,  4.,  5.,  9.])


    Example 2:
        Repeating the Walsh-ordered S-transform using input indices is faster

        >>> import timeit
        >>> import torch
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> x = torch.rand(512, 2**14-1, device=torch.device('cpu'))
        >>> t = timeit.timeit(lambda: wh.fwalsh_S_torch(x), number=10)
        >>> print(f"No indices as inputs (10x): {t:.3f} seconds")
        No indices as inputs (10x): ... seconds
        >>> ind = wh.sequency_perm_ind(x.shape[-1]+1)
        >>> t = timeit.timeit(lambda: wh.fwalsh_S_torch(x,ind), number=10)
        >>> print(f"With indices as inputs (10x): {t:.3f} seconds")
        With indices as inputs (10x): ... seconds
    """
    j = torch.sum(x, -1, keepdim=True)
    s = fwalsh_G_torch(x, ind)
    s = (j - s) / 2
    return s


def fwalsh2_S_torch(X, ind=True):  # not validated!
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
        tensor([[ 0., 30., 27., 33.],
                [37., 33., 26., 26.],
                [32., 16., 29., 25.],
                [35., 33., 38., 28.]])

    Example 2:
        Repeating the Walsh-ordered S-transform using input indices is faster

        >>> import timeit
        >>> import torch
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> X = torch.rand(128, 2**6, 2**6, device=torch.device('cpu'))
        >>> t = timeit.timeit(lambda: wh.fwalsh2_S_torch(X), number=10)
        >>> print(f"No indices as inputs (10x): {t:.3f} seconds")
        No indices as inputs (10x): ... seconds
        >>> ind = wh.sequency_perm_ind(X.shape[-1]*X.shape[-2])
        >>> t = timeit.timeit(lambda: wh.fwalsh2_S_torch(X,ind), number=10)
        >>> print(f"With indices as inputs (10x): {t:.3f} seconds")
        With indices as inputs (10x): ... seconds

    """
    x = walsh2_S_unfold_torch(X)
    s = fwalsh_S_torch(x, ind)
    S = walsh2_S_fold_torch(s)
    return S


def ifwalsh_S_torch(s, ind=True):
    r"""Inverse Fast Walsh S-transform of x

    Args:
        :attr:`x` (torch.tensor):  input signal with shape :attr:`(*, n)`,
        where n+1 should be a power of two.

        :attr:`ind` (bool, optional): True for sequency (default)

        :attr:`ind` (list, optional): permutation indices. This is faster than True
        when repeating the sequency-ordered transform
        multilple times.

    Returns:
        torch.tensor: -by-n S-transformed signal

    Examples:
        Example 1: Inverse Walsh-ordered S-transform of a signal of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch
        >>> x = torch.tensor([12., 9, 9, 16, 4, 5, 9])
        >>> s = wh.ifwalsh_S_torch(x)
        >>> print(s)
        tensor([ 1.,  3., -0., -1.,  7.,  5.,  1.])


        Example 2: Check the inverse of the direct transform of a signal of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch
        >>> x = torch.tensor([12., 9, 9, 16, 4, 5, 9])
        >>> s = wh.fwalsh_S_torch(wh.ifwalsh_S_torch(x))
        >>> print(s-x)
        tensor([0., 0., 0., 0., 0., 0., 0.])

        Example 3: Inverse Walsh-ordered S-transform of 2 signals of length 7

        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch
        >>> x = torch.tensor([[1, 3, 0, -1, 7, 5, 1],[12., 9, 9, 16, 4, 5, 9]])
        >>> s = wh.ifwalsh_S_torch(x)
        >>> print(s)
        tensor([[ 2.0000,  0.5000,  0.5000,  4.0000, -2.0000, -1.5000,  0.5000],
                [ 1.0000,  3.0000, -0.0000, -1.0000,  7.0000,  5.0000,  1.0000]])

    """
    n = s.shape[-1]
    x = -2 / (n + 1) * fwalsh_G_torch(s, ind)

    return x


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
        tensor(..., dtype=torch.float64)
    """

    # N and n should be consistent with doc
    N = x.shape[-1]
    n = (N + 1) ** 0.5
    assert math.log2(n) % 1 == 0, f"N+1 = n*n, where n={n:.2f} must be a power of two"
    n = int(n)

    # Concatenate with zeros
    z = torch.zeros(x.shape[:-1], device=x.device)
    z = z[..., None]
    X = torch.cat((z, x), dim=-1)  # Extra memory allocated here

    # Reshape to get images
    new_shape = torch.Size([*x.shape[:-1], n, n])
    X = X.reshape(new_shape)
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
        tensor([[1, 3, 0, 8],
                [7, 5, 1, 2]])
        >>> print(x)
        tensor([3, 0, 8, 7, 5, 1, 2])

    Example 2:
        >>> import spyrit.misc.walsh_hadamard as wh
        >>> import torch
        >>> X = torch.randint(10,(3,4,4))
        >>> x = wh.walsh2_S_unfold_torch(X)
        >>> print(X)
        tensor(...)
        >>> print(x)
        tensor(...)

    """
    x = X.reshape(*(X.size()[:-2]), -1)
    x = x[..., 1:]
    return x


def walsh_torch(x, H=None):
    """Deprecated function. Please use spyrit.core.torch.fwht instead."""
    raise NotImplementedError(
        "This function is deprecated. Please call spyrit.core.torch.fwht instead."
    )


def walsh2_torch(im, H=None):
    """Deprecated function. Please use spyrit.core.torch.fwht_2d instead."""
    raise NotImplementedError(
        "This function is deprecated. Please call spyrit.core.torch.fwht_2d instead."
    )
