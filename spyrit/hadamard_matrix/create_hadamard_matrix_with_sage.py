from sage.all import *
from sage.combinat.matrices.hadamard_matrix import (
    hadamard_matrix,
    skew_hadamard_matrix,
    is_hadamard_matrix,
    is_skew_hadamard_matrix,
)
import numpy as np
import glob

# Get all Hadamard matrices of order 4*n for Sage
# https://github.com/sagemath/sage/
# run in conda env with:
# sage create_hadamard_matrix_with_sage.py

k = Integer(2000)
for n in range(Integer(1), k + Integer(1)):
    try:
        H = hadamard_matrix(Integer(4) * n, check=False)

        if is_hadamard_matrix(H):
            print(n * 4)
            a = np.array(H)
            a[a == -1] = 0
            a = a.astype(bool)

            # find the files with that order
            files = glob.glob("had." + str(n * 4) + "*.npz")
            already_saved = False
            for file in files:
                b = np.load(file)
                if a == b:
                    already_saved = True
                if already_saved:
                    break

            if not already_saved:
                name = "had." + str(n * 4) + ".sage.npz"
                np.savez_compressed(name, a)
    except ValueError as e:
        pass
