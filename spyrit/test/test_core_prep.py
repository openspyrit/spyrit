# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:19:24 2023

@author: ducros
"""
# %% Test DirectPoisson
from spyrit.core.meas import Linear
from spyrit.core.prep import DirectPoisson
import numpy as np
import torch
from test_helpers import assert_test


def test_core_prep():
    
    print('\n*** Testing prep.py ***')
    # constructor and forward
    x = torch.rand([10, 400], dtype=torch.float)
    H = np.random.random([400, 32 * 32])
    meas_op = Linear(H)
    prep_op = DirectPoisson(1.0, meas_op)
    m = prep_op(x)
    print(m.shape)
    assert_test(m.shape, torch.Size([10, 400]), "Wrong matrix size")

    # variance
    x = torch.rand([10, 400], dtype=torch.float)
    v = prep_op.sigma(x)
    print(v.shape)
    assert_test(v.shape, torch.Size([10, 400]), "Wrong matrix size")

    # denormalize_expe
    x = torch.rand([10, 1, 32, 32], dtype=torch.float)
    beta = 9 * torch.rand([10])
    y = prep_op.denormalize_expe(x, beta, 32, 32)
    print(y.shape)
    assert_test(y.shape, torch.Size([10, 1, 32, 32]), "Wrong matrix size")

    # %% Test SplitPoisson
    from spyrit.core.meas import LinearSplit, HadamSplit
    from spyrit.core.prep import SplitPoisson

    # forward with LinearSplit
    x = torch.rand([10, 2 * 400], dtype=torch.float)
    H = np.random.random([400, 32 * 32])

    meas_op = LinearSplit(H)
    split_op = SplitPoisson(10, meas_op)
    m = split_op(x)
    print(m.shape)
    assert_test(m.shape, torch.Size([10, 400]), "Wrong matrix size")

    # forward with HadamSplit
    Perm = np.random.random([32, 32])
    meas_op = HadamSplit(400, 32, Perm)
    split_op = SplitPoisson(10, meas_op)
    m = split_op(x)
    print(m.shape)
    assert_test(m.shape, torch.Size([10, 400]), "Wrong matrix size")

    # forward_expe
    m, alpha = split_op.forward_expe(x, meas_op)
    print(m.shape)
    assert_test(m.shape, torch.Size([10, 400]), "Wrong matrix size")
    print(alpha.shape)
    assert_test(alpha.shape, torch.Size([10]), "Wrong matrix size")

    # sigma
    x = torch.rand([10, 2 * 400], dtype=torch.float)
    v = split_op.sigma(x)
    print(v.shape)
    assert_test(v.shape, torch.Size([10, 400]), "Wrong matrix size")

    # set_expe
    split_op.set_expe(gain=1.6)
    print(split_op.gain)
    assert_test(split_op.gain, 1.6, "Wrong gain")

    # sigma_expe
    v = split_op.sigma_expe(x)
    print(v.shape)
    assert_test(v.shape, torch.Size([10, 400]), "Wrong matrix size")

    # sigma_from_image
    x = torch.rand([10, 32 * 32], dtype=torch.float)
    v = split_op.sigma_from_image(x, meas_op)
    print(v.shape)
    assert_test(v.shape, torch.Size([10, 400]), "Wrong matrix size")

    # denormalize_expe
    x = torch.rand([10, 1, 32, 32], dtype=torch.float)
    beta = 9 * torch.rand([10, 1])
    y = split_op.denormalize_expe(x, beta, 32, 32)
    print(y.shape)
    assert_test(y.shape, torch.Size([10, 1, 32, 32]), "Wrong matrix size")

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
    # assert_test(m.shape, torch.Size([10, 24, 64]), "Wrong matrix size")

    print("✓ All tests passed for prep.py")
    print("==============================")
    return True

if __name__ == "__main__":
    test_core_prep()
