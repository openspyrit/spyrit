"""
Test for module noise.py
"""

import torch

from test_helpers import assert_shape

from spyrit.core.meas import Linear, LinearSplit, HadamSplit  # , LinearRowSplit


def test_core_noise():

    print("\n*** Testing noise.py ***")

    # =========================================================================
    ## NoNoise
    print("NoNoise")
    from spyrit.core.noise import NoNoise

    # constructor
    print("\tconstructor... ", end="")
    H = torch.rand(400, 32 * 32)
    linear_op = Linear(H)
    linear_acq = NoNoise(linear_op)
    print("ok")

    # forward
    print("\tforward... ", end="")
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = linear_acq(x)
    assert_shape(y.shape, torch.Size([10, 400]), "Wrong matrix size")
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # forward with HadamSplit
    print("\tforward with HadamSplit... ", end="")
    Perm = torch.rand([32, 32])
    split_op = HadamSplit(400, 32, Perm)
    split_acq = NoNoise(split_op)
    y = split_acq(x)
    assert_shape(y.shape, torch.Size([10, 800]), "Wrong matrix size")
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # =========================================================================
    ## Poisson
    print("Poisson")
    from spyrit.core.noise import Poisson

    # EXAMPLE 1 : WITH LINEAR
    # constructor (example 1)
    print("\tconstructor with Linear... ", end="")
    H = torch.rand([400, 32 * 32])
    meas_op = Linear(H)
    noise_op = Poisson(meas_op, 10.0)
    print("ok")

    # forward (example 1)
    print("\tforward example 1... ", end="")
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = noise_op(x)
    assert_shape(y.shape, torch.Size([10, 400]), "Wrong matrix size")
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
    print("\tforward example 2... ", end="")
    y = noise_op(x)
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # EXAMPLE 2 : WITH HADAMSPLIT
    # constructor with HadamSplit
    print("\tconstructor with HadamSplit... ", end="")
    Perm = torch.rand([32, 32])
    meas_op = HadamSplit(400, 32, Perm)
    noise_op = Poisson(meas_op, 200.0)
    print("ok")

    # forward with HadamSplit
    print("\tforward with HadamSplit example 1... ", end="")
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = noise_op(x)
    assert_shape(y.shape, torch.Size([10, 800]), "Wrong matrix size")
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
    print("\tforward with HadamSplit example 2... ", end="")
    y = noise_op(x)
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # EXAMPLE 3 : WITH LINEARSPLIT
    # constructor with LinearSplit
    print("\tconstructor with LinearSplit... ", end="")
    H = torch.rand(24, 64)
    meas_op = LinearSplit(H)
    noise_op = Poisson(meas_op, 50.0)
    print("ok")

    # forward with LinearSplit
    print("\tforward with LinearSplit example 1... ", end="")
    x = torch.FloatTensor(10, 64).uniform_(-1, 1)
    y = noise_op(x)
    assert_shape(y.shape, torch.Size([10, 48]), "Wrong matrix size")
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
    print("\tforward with LinearSplit example 2... ", end="")
    y = noise_op(x)
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # =========================================================================
    ## PoissonApproxGauss
    print("PoissonApproxGauss")
    from spyrit.core.noise import PoissonApproxGauss

    # EXAMPLE 1
    # constructor with Linear
    print("\tconstructor with Linear... ", end="")
    H = torch.rand([400, 32 * 32])
    meas_op = Linear(H)
    noise_op = PoissonApproxGauss(meas_op, 10.0)
    print("ok")

    # forward (example 1)
    print("\tforward example 1... ", end="")
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = noise_op(x)
    assert_shape(y.shape, torch.Size([10, 400]), "Wrong matrix size")
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
    print("\tforward example 2... ", end="")
    y = noise_op(x)
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # EXAMPLE 2
    # constructor with HadamSplit
    print("\tconstructor with HadamSplit... ", end="")
    Perm = torch.rand([32, 32])
    meas_op = HadamSplit(400, 32, Perm)
    noise_op = PoissonApproxGauss(meas_op, 200.0)
    print("ok")

    # forward with HadamSplit
    print("\tforward with HadamSplit example 1... ", end="")
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = noise_op(x)
    assert_shape(y.shape, torch.Size([10, 800]), "Wrong matrix size")
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
    print("\tforward with HadamSplit example 2... ", end="")
    y = noise_op(x)
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # EXAMPLE 3
    # constructor with LinearSplit
    print("\tconstructor with LinearSplit... ", end="")
    H = torch.rand(24, 64)
    meas_op = LinearSplit(H)
    noise_op = PoissonApproxGauss(meas_op, 50.0)
    print("ok")

    # forward with LinearSplit
    print("\tforward with LinearSplit example 1... ", end="")
    x = torch.FloatTensor(10, 64).uniform_(-1, 1)
    y = noise_op(x)
    assert_shape(y.shape, torch.Size([10, 48]), "Wrong matrix size")
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
    print("\tforward with LinearSplit example 2... ", end="")
    y = noise_op(x)
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # =========================================================================
    ## PoissonApproxGaussSameNoise
    print("PoissonApproxGaussSameNoise")
    from spyrit.core.noise import PoissonApproxGaussSameNoise

    # EXAMPLE 1
    # constructor with Linear
    print("\tconstructor with Linear... ", end="")
    H = torch.rand([400, 32 * 32])
    meas_op = Linear(H)
    noise_op = PoissonApproxGaussSameNoise(meas_op, 10.0)
    print("ok")

    # forward (example 1)
    print("\tforward example 1... ", end="")
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = noise_op(x)
    assert_shape(y.shape, torch.Size([10, 400]), "Wrong matrix size")
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
    print("\tforward example 2... ", end="")
    y = noise_op(x)
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # EXAMPLE 2
    # constructor with HadamSplit
    print("\tconstructor with HadamSplit... ", end="")
    Perm = torch.rand([32, 32])
    meas_op = HadamSplit(400, 32, Perm)
    noise_op = PoissonApproxGaussSameNoise(meas_op, 200.0)
    print("ok")

    # forward with HadamSplit
    print("\tforward with HadamSplit example 1... ", end="")
    x = torch.FloatTensor(10, 32 * 32).uniform_(-1, 1)
    y = noise_op(x)
    assert_shape(y.shape, torch.Size([10, 800]), "Wrong matrix size")
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")
    print("\tforward with HadamSplit example 2... ", end="")
    y = noise_op(x)
    print(f"ok - Measurements in ({torch.min(y):.2f} , {torch.max(y):.2f})")

    # =========================================================================
    print("All tests passed for noise.py")
    print("===============================")
    return True


if __name__ == "__main__":
    test_core_noise()
