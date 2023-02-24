# -*- coding: utf-8 -*-
#%%
import numpy as np
import torch
from spyrit.core.meas import HadamSplit

#%%
from spyrit.core.recon import PseudoInverse

# constructor
recon_op = PseudoInverse()

# forward
H = np.random.random([400,32*32])
Perm = np.random.random([32*32,32*32])
meas_op =  HadamSplit(H, Perm, 32, 32)
y = torch.rand([85,400], dtype=torch.float)  

x = recon_op(y, meas_op)
print(x.shape)

#%%
from spyrit.core.recon import TikhonovMeasurementPriorDiag

# constructor
sigma = np.random.random([32*32, 32*32])
recon_op = TikhonovMeasurementPriorDiag(sigma, 400)

# forward
H = np.random.random([400,32*32])
Perm = np.random.random([32*32,32*32])
meas_op =  HadamSplit(H, Perm, 32, 32)
y = torch.rand([85,400], dtype=torch.float)  

x_0 = torch.zeros((85, 32*32), dtype=torch.float)
var = torch.zeros((85, 400), dtype=torch.float)

x = recon_op(y, x_0, var, meas_op)
print(x.shape)