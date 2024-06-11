"""
Created on Wed Feb 15 15:19:24 2023

@author: ducros & phan
"""

import torch

from spyrit.core.meas import Linear, LinearSplit, HadamSplit

from test_helpers import assert_shape


def test_core_prep():

    print("\n*** Testing prep.py ***")

    # =========================================================================
    ## DirectPoisson
    print("DirectPoisson")
    from spyrit.core.prep import DirectPoisson

    # constructor
    print("\tconstructor... ", end="")
    H = torch.rand([400, 32 * 32])
    meas_op = Linear(H)
    prep_op = DirectPoisson(1.0, meas_op)
    print("ok")

    # forward
    print("\tforward... ", end="")
    x = torch.rand([10, 400], dtype=torch.float)
    m = prep_op(x)
    assert_shape(m.shape, torch.Size([10, 400]), "Wrong matrix size")
    print("ok")

    # variance
    print("\tvariance... ", end="")
    x = torch.rand([10, 400], dtype=torch.float)
    v = prep_op.sigma(x)
    assert_shape(v.shape, torch.Size([10, 400]), "Wrong matrix size")
    print("ok")

    # denormalize_expe
    print("\tdenormalize_expe... ", end="")
    x = torch.rand([10, 1, 32, 32], dtype=torch.float)
    beta = 9 * torch.rand([10])
    y = prep_op.denormalize_expe(x, beta, 32, 32)
    assert_shape(y.shape, torch.Size([10, 1, 32, 32]), "Wrong matrix size")
    print("ok")

    # =========================================================================
    ## Test SplitPoisson
    print("SplitPoisson")
    from spyrit.core.prep import SplitPoisson

    # constructor with LinearSplit
    print("\tconstructor with LinearSplit... ", end="")
    H = torch.rand([400, 32 * 32])
    meas_op = LinearSplit(H)
    split_op = SplitPoisson(10, meas_op)
    print("ok")

    # forward with LinearSplit
    print("\tforward with LinearSplit... ", end="")
    x = torch.rand([10, 2 * 400], dtype=torch.float)
    m = split_op(x)
    assert_shape(m.shape, torch.Size([10, 400]), "Wrong matrix size")
    print("ok")

    # constructor with HadamSplit
    print("\tconstructor with HadamSplit... ", end="")
    Perm = torch.rand([32, 32])
    meas_op = HadamSplit(400, 32, Perm)
    split_op = SplitPoisson(10, meas_op)
    print("ok")

    # forward with HadamSplit
    print("\tforward with HadamSplit... ", end="")
    x = torch.rand([10, 2 * 400], dtype=torch.float)
    m = split_op(x)
    assert_shape(m.shape, torch.Size([10, 400]), "Wrong matrix size")
    print("ok")

    # forward_expe
    print("\tforward_expe... ", end="")
    m, alpha = split_op.forward_expe(x, meas_op)
    assert_shape(m.shape, torch.Size([10, 400]), "Wrong matrix size for m")
    assert_shape(alpha.shape, torch.Size([10]), "Wrong matrix size for alpha")
    print("ok")

    # sigma
    print("\tsigma... ", end="")
    x = torch.rand([10, 2 * 400], dtype=torch.float)
    v = split_op.sigma(x)
    assert_shape(v.shape, torch.Size([10, 400]), "Wrong matrix size")
    print("ok")

    # set_expe
    print("\tset_expe... ", end="")
    split_op.set_expe(gain=1.6)
    assert_shape(split_op.gain, 1.6, "Wrong gain")
    print("ok")

    # sigma_expe
    print("\tsigma_expe... ", end="")
    v = split_op.sigma_expe(x)
    assert_shape(v.shape, torch.Size([10, 400]), "Wrong matrix size")
    print("ok")

    # sigma_from_image
    print("\tsigma_from_image... ", end="")
    x = torch.rand([10, 32 * 32], dtype=torch.float)
    v = split_op.sigma_from_image(x, meas_op)
    assert_shape(v.shape, torch.Size([10, 400]), "Wrong matrix size")
    print("ok")

    # denormalize_expe
    print("\tdenormalize_expe... ", end="")
    x = torch.rand([10, 1, 32, 32], dtype=torch.float)
    beta = 9 * torch.rand([10, 1])
    y = split_op.denormalize_expe(x, beta, 32, 32)
    assert_shape(y.shape, torch.Size([10, 1, 32, 32]), "Wrong matrix size")
    print("ok")

    # % Test SplitRowPoisson
    # from spyrit.core.meas import LinearSplit
    # from spyrit.core.prep import SplitRowPoisson

    # # constructor
    # split_op = SplitRowPoisson(2.0, 24, 64)

    # # forward with LinearSplit
    # x = torch.rand([10, 48, 64], dtype=torch.float)
    # H_pos = np.random.random([24, 64])
    # meas_op = LinearSplit(H_pos)

    # # forward
    # m = split_op(x, meas_op)
    # print(m.shape)
    # assert_shape(m.shape, torch.Size([10, 24, 64]), "Wrong matrix size")

    # =========================================================================
    print("All tests passed for prep.py")
    print("==============================")
    return True


if __name__ == "__main__":
    test_core_prep()
