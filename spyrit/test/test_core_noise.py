# -*- coding: utf-8 -*-
import numpy as np
import torch
from spyrit.core.forwop import Linear, LinearSplit, LinearRowSplit, HadamSplit

#%% NoNoise
from spyrit.core.noise import NoNoise

# constructor
H = np.random.random([400,32*32])
linear_op = Linear(H)
linear_acq = NoNoise(linear_op)

# forward
x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
y = linear_acq(x)
print(y.shape)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

# forward with HadamSplit
Perm = np.random.random([32*32,32*32])
split_op = HadamSplit(H, Perm, 32, 32)
split_acq = NoNoise(split_op)

y = split_acq(x)
print(y.shape)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

#%% Poisson
from spyrit.core.noise import Poisson

# EXAMPLE 1
# constructor (example 1)
H = np.random.random([400,32*32])
meas_op = Linear(H)
noise_op = Poisson(meas_op, 10.0)

# forward (example 1)
x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
y = noise_op(x)
print(y.shape)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

y = noise_op(x)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

# EXAMPLE 2
# constructor with HadamSplit
Perm = np.random.random([32*32,32*32])
meas_op = HadamSplit(H, Perm, 32, 32)
noise_op = Poisson(meas_op, 200.0)

# forward with HadamSplit
x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
y = noise_op(x)
print(y.shape)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

y = noise_op(x)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

# EXAMPLE 3
H_pos = np.random.rand(24,64)
H_neg = np.random.rand(24,64)
meas_op = LinearRowSplit(H_pos,H_neg)
noise_op = Poisson(meas_op, 50.0)

x = torch.FloatTensor(10, 64, 92).uniform_(-1, 1)
y = noise_op(x)
print(y.shape)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

y = noise_op(x)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
            
#%% PoissonApproxGauss
from spyrit.core.noise import PoissonApproxGauss

# EXAMPLE 1
# constructor
H = np.random.random([400,32*32])
meas_op = Linear(H)
noise_op = PoissonApproxGauss(meas_op, 10.0)

# forward (example 1)
x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
y = noise_op(x)
print(y.shape)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

y = noise_op(x)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

# EXAMPLE 2
# constructor with HadamSplit
Perm = np.random.random([32*32,32*32])
meas_op = HadamSplit(H, Perm, 32, 32)
noise_op = PoissonApproxGauss(meas_op, 200.0)

# forward with HadamSplit
x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
y = noise_op(x)
print(y.shape)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

y = noise_op(x)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

# EXAMPLE 3
H_pos = np.random.rand(24,64)
H_neg = np.random.rand(24,64)
meas_op = LinearRowSplit(H_pos,H_neg)
noise_op = PoissonApproxGauss(meas_op, 50.0)

x = torch.FloatTensor(10, 64, 92).uniform_(-1, 1)
y = noise_op(x)
print(y.shape)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

y = noise_op(x)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

#%% PoissonApproxGaussSameNoise
from spyrit.core.noise import PoissonApproxGaussSameNoise

# EXAMPLE 1
# constructor
H = np.random.random([400,32*32])
meas_op = Linear(H)
noise_op = PoissonApproxGaussSameNoise(meas_op, 10.0)

# forward (example 1)
x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
y = noise_op(x)
print(y.shape)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

y = noise_op(x)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

# EXAMPLE 2
# constructor with HadamSplit
Perm = np.random.random([32*32,32*32])
meas_op = HadamSplit(H, Perm, 32, 32)
noise_op = PoissonApproxGaussSameNoise(meas_op, 200.0)

# forward with HadamSplit
x = torch.FloatTensor(10, 32*32).uniform_(-1, 1)
y = noise_op(x)
print(y.shape)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

y = noise_op(x)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

# EXAMPLE 3
H_pos = np.random.rand(24,64)
H_neg = np.random.rand(24,64)
meas_op = LinearRowSplit(H_pos,H_neg)
noise_op = PoissonApproxGaussSameNoise(meas_op, 50.0)

x = torch.FloatTensor(10, 64, 92).uniform_(-1, 1)
y = noise_op(x)
print(y.shape)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

y = noise_op(x)
print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")