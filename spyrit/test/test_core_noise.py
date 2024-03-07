# -*- coding: utf-8 -*-
import numpy as np
import torch
from spyrit.core.meas import Linear, LinearSplit, HadamSplit  # , LinearRowSplit
from test_helpers import assert_test


def test_core_noise():

    print("\n*** Testing noise.py ***")
    # %% NoNoise
    from spyrit.core.noise import NoNoise

    # constructor
    H = np.random.random([400, 32 * 32])
    linear_op = Linear(H)
    linear_acq = NoNoise(linear_op)

    # forward
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = linear_acq(x)
    print(y.shape)
    assert_test(y.shape, torch.Size([10, 400]), "Wrong matrix size")
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # forward with HadamSplit
    Perm = np.random.random([32, 32])
    split_op = HadamSplit(400, 32, Perm)
    split_acq = NoNoise(split_op)

    y = split_acq(x)
    print(y.shape)
    assert_test(y.shape, torch.Size([10, 800]), "Wrong matrix size")
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # %% Poisson
    from spyrit.core.noise import Poisson

    # EXAMPLE 1
    # constructor (example 1)
    H = np.random.random([400, 32 * 32])
    meas_op = Linear(H)
    noise_op = Poisson(meas_op, 10.0)

    # forward (example 1)
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = noise_op(x)
    print(y.shape)
    assert_test(y.shape, torch.Size([10, 400]), "Wrong matrix size")
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    y = noise_op(x)
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # EXAMPLE 2
    # constructor with HadamSplit
    Perm = np.random.random([32, 32])
    meas_op = HadamSplit(400, 32, Perm)
    noise_op = Poisson(meas_op, 200.0)

    # forward with HadamSplit
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = noise_op(x)
    print(y.shape)
    assert_test(y.shape, torch.Size([10, 800]), "Wrong matrix size")
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    y = noise_op(x)
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # EXAMPLE 3
    H = np.random.rand(24, 64)
    meas_op = LinearSplit(H)
    noise_op = Poisson(meas_op, 50.0)

    x = torch.FloatTensor(10, 64).uniform_(-1, 1)
    y = noise_op(x)
    print(y.shape)
    assert_test(y.shape, torch.Size([10, 48]), "Wrong matrix size")
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    y = noise_op(x)
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # %% PoissonApproxGauss
    from spyrit.core.noise import PoissonApproxGauss

    # EXAMPLE 1
    # constructor
    H = np.random.random([400, 32 * 32])
    meas_op = Linear(H)
    noise_op = PoissonApproxGauss(meas_op, 10.0)

    # forward (example 1)
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = noise_op(x)
    print(y.shape)
    assert_test(y.shape, torch.Size([10, 400]), "Wrong matrix size")
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    y = noise_op(x)
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # EXAMPLE 2
    # constructor with HadamSplit
    Perm = np.random.random([32, 32])
    meas_op = HadamSplit(400, 32, Perm)
    noise_op = PoissonApproxGauss(meas_op, 200.0)

    # forward with HadamSplit
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = noise_op(x)
    print(y.shape)
    assert_test(y.shape, torch.Size([10, 800]), "Wrong matrix size")
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    y = noise_op(x)
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # EXAMPLE 3
    H = np.random.rand(24, 64)
    meas_op = LinearSplit(H)
    noise_op = PoissonApproxGauss(meas_op, 50.0)

    x = torch.FloatTensor(10, 64).uniform_(-1, 1)
    y = noise_op(x)
    print(y.shape)
    assert_test(y.shape, torch.Size([10, 48]), "Wrong matrix size")
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    y = noise_op(x)
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # %% PoissonApproxGaussSameNoise
    from spyrit.core.noise import PoissonApproxGaussSameNoise

    # EXAMPLE 1
    # constructor
    H = np.random.random([400, 32 * 32])
    meas_op = Linear(H)
    noise_op = PoissonApproxGaussSameNoise(meas_op, 10.0)

    # forward (example 1)
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = noise_op(x)
    print(y.shape)
    assert_test(y.shape, torch.Size([10, 400]), "Wrong matrix size")
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    y = noise_op(x)
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # EXAMPLE 2
    # constructor with HadamSplit
    Perm = np.random.random([32, 32])
    meas_op = HadamSplit(400, 32, Perm)
    noise_op = PoissonApproxGaussSameNoise(meas_op, 200.0)

    # forward with HadamSplit
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = noise_op(x)
    print(y.shape)
    assert_test(y.shape, torch.Size([10, 800]), "Wrong matrix size")
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    y = noise_op(x)
    print(f"Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    print("All tests passed for noise.py")
    print("===============================")
    return True


if __name__ == "__main__":
    test_core_noise()
