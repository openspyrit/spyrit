# -----------------------------------------------------------------------------
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE.md for further details
# -----------------------------------------------------------------------------

import numpy as np;


def translation_matrix(img_size,nb_pixels):
    init_ind = np.reshape(np.arange(img_size**2), (img_size, img_size));
    final_ind = np.zeros((img_size, img_size));
    final_ind[:,:(img_size-nb_pixels)] = init_ind[:,nb_pixels:];
    final_ind[:,(img_size-nb_pixels):] = init_ind[:,:nb_pixels];

    final_ind = np.reshape(final_ind, (img_size**2,1));
    init_ind = np.reshape(init_ind, (img_size**2,1));
    F = permutation_matrix(final_ind, init_ind);
    return F;

def permutation_matrix(A,B):
    N = A.shape[0];
    I = np.eye(N);
    P = np.zeros((N,N));

    for i in range(N):
        pat = np.matlib.repmat(A[i,:],N,1);
        ind = np.where(np.sum((pat == B),axis=1));
        P[ind,:] = I[i,:];

    return P;


def circle(img_size,R, x_max):
    x = np.linspace(-x_max, x_max, img_size);
    X,Y = np.meshgrid(x,x);
    return 1.0*(X**2+Y**2<R);
