# -*- coding: utf-8 -*-
#%%
import numpy as np
import torch
from spyrit.core.forwop import HadamSplit

#%%
from spyrit.core.Data_Consistency import PseudoInverse

H = np.random.random([400,32*32])
Perm = np.random.random([32*32,32*32])
meas_op =  HadamSplit(H, Perm, 32, 32)
y = torch.rand([85,400], dtype=torch.float)  
pinv_op = PseudoInverse()
x = pinv_op(y, meas_op)
print(x.shape)
torch.Size([85, 1024])