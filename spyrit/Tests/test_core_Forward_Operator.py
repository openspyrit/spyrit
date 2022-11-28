import numpy as np
from spyrit.core.Forward_Operator import *
import torch
from spyrit.misc.walsh_hadamard import walsh_matrix

# test Forward_operator

img_size = 32*32
nb_measurements = 400
batch_size = 100
Hcomplete = walsh_matrix(img_size)
Hsub = Hcomplete[0:M,:]
Forward_OP = Forward_operator(Hsub)
x = torch.tensor(np.random.random([batch_size,img_size]), dtype=torch.float)
y = Forward_OP(x)
print('Hsub shape:', Hsub.shape)
print('input shape:', x.shape)
print('output shape:', y.shape)