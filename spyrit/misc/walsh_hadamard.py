#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/01/2020

@author: seb
"""

import math
import numpy as np 
from scipy.linalg import hadamard
from sympy.combinatorics.graycode import GrayCode


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
    
"""Generation of a Waslh ordered matrix"""

def walsh_ordered(n): # Generate the n*n Walsh ordered matrix
    BR = bit_reverse_matrix(n)
    GRp = gray_code_permutation(n)
    H = hadamard(n)
    return(np.dot(np.dot(GRp,BR),H)) # Apply permutation to the hadmard matrix 