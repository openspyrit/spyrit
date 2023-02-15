# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:19:24 2023

@author: ducros
"""
#%% Test Linear
from spyrit.core.Preprocess import SplitPoisson

# constructor
H = np.array(np.random.random([400,32*32]))
linop = Linear(H)

# forward
x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
y = linop(x)
print('Output shape of forward:', y.shape)
