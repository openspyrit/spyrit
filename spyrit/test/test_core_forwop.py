# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:04:11 2023

@author: ducros
"""


#%%

import torch
import numpy as np
#from spyrit.misc.walsh_hadamard import walsh_matrix





#%% Test Linear
from spyrit.core.Forward_Operator import Linear

# constructor
H = np.array(np.random.random([400,32*32]))
linop = Linear(H)

# forward
x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
y = linop(x)
print('Output shape of forward:', y.shape)

# adjoint
x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
y = linop.adjoint(x)
print('Output shape of adjoint:', y.shape)

# get_mat
H = linop.get_mat()
print('Shape of the measurement matrix:', H.shape)

#%% Test LinearSplit
from spyrit.core.Forward_Operator import LinearSplit

# constructor
H = np.array(np.random.random([400,32*32]))
linop = LinearSplit(H)

# forward
x = torch.tensor(np.random.random([10,32*32]), dtype=torch.float)
y = linop(x)
print('Output shape of forward:', y.shape)

# adjoint
x = torch.tensor(np.random.random([10,400]), dtype=torch.float)
y = linop.adjoint(x)
print('Output shape of adjoint:', y.shape)

# get_mat
H = linop.get_mat()
print('Shape of the measurement matrix:', H.shape)

#%% Instantiate Hsub

img_size = 32*32
nb_measurements = 400
batch_size = 10
Hcomplete = walsh_matrix(img_size)
Hsub = Hcomplete[0:nb_measurements,:]


#%% Forward_operator
# 
Forward_OP = Forward_operator(Hsub)
x = torch.tensor(np.random.random([batch_size,img_size]), dtype=torch.float)
y = Forward_OP(x)
print('Hsub shape:', Hsub.shape)
print('input shape:', x.shape)
print('output shape:', y.shape)


# ### adjoint

# In[4]:


x_back = Forward_OP.adjoint(y)
print(x_back.shape)


# ## Forward_operator_Split

# In[5]:


Forward_Op_Split = Forward_operator_Split(Hsub)
y = Forward_Op_Split(x)
print(y.shape)


# ## Forward_operator_Split_ft_had

# ### inverse

# In[6]:


Perm = np.array(np.random.random([32*32,32*32]))
FO_Had = Forward_operator_Split_ft_had(Hsub, Perm, 32, 32) 
x_inverse = FO_Had.inverse(x)
print(x_inverse.shape)


# ### pinv

# In[7]:


x = torch.Tensor(np.random.random([10,400]))
x_pinv = FO_Had.pinv(x)
print(x_pinv.shape)


# ## Forward_operator_shift

# In[8]:


FO_Shift = Forward_operator_shift(Hsub, Perm)
x = torch.tensor(np.random.random([batch_size,img_size]), dtype=torch.float)
y = FO_Shift(x)
print(y.shape)


# ## Forward_operator_pos

# In[9]:


Forward_OP_pos = Forward_operator_pos(Hsub, Perm)
y = Forward_OP_pos(x)
print(y.shape)


# ## Forward_operator_shift_had

# In[10]:


FO_Shift_Had = Forward_operator_shift_had(Hsub, Perm)
x_reconstruct = FO_Shift_Had.inverse(x)
print(x_reconstruct.shape)

