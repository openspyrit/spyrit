"""
Created on Mon Feb 13 18:04:11 2023

@author: ducros & phan
"""

import torch

from test_helpers import assert_test


def test_core_meas():

    print("\n*** Testing meas.py ***")

    # =========================================================================
    ## Linear
    print("Linear")
    from spyrit.core.meas import Linear

    # constructor
    print("\tconstructor... ", end="")
    H = torch.rand(400, 2500)
    meas_op = Linear(H)
    print("ok")

    # forward
    print("\tforward... ", end="")
    x = torch.rand([10, 2500], dtype=torch.float)
    y = meas_op(x)
    assert_test(y.shape, torch.Size([10, 400]), "Wrong forward size")
    print("ok")

    # adjoint
    print("\tadjoint... ", end="")
    x = torch.rand([10, 400], dtype=torch.float)
    y = meas_op.adjoint(x)
    assert_test(y.shape, torch.Size([10, 2500]), "Wrong adjoint size")
    print("ok")

    # get_mat
    print("\tget_mat... ", end="")
    H = meas_op.get_H()
    assert_test(H.shape, torch.Size([400, 2500]), "Wrong get_mat size")
    print("ok")

    # =========================================================================
    ## LinearSplit
    print("LinearSplit")
    from spyrit.core.meas import LinearSplit

    # constructor
    print("\tconstructor... ", end="")
    H = torch.rand(400, 2500)
    meas_op = LinearSplit(H)
    print("ok")

    # forward
    print("\tforward... ", end="")
    x = torch.rand([10, 2500], dtype=torch.float)
    y = meas_op(x)
    assert_test(y.shape, torch.Size([10, 800]), "Wrong forward size")
    print("ok")

    # forward_H
    print("\tforward_H... ", end="")
    x = torch.rand([10, 2500], dtype=torch.float)
    y = meas_op.forward_H(x)
    assert_test(y.shape, torch.Size([10, 400]), "Wrong forward_H size")
    print("ok")

    # adjoint
    print("\tadjoint... ", end="")
    x = torch.rand([10, 400], dtype=torch.float)
    y = meas_op.adjoint(x)
    assert_test(y.shape, torch.Size([10, 2500]), "Wrong adjoint size")
    print("ok")

    # get_mat
    print("\tget_mat... ", end="")
    H = meas_op.get_H()
    assert_test(H.shape, torch.Size([400, 2500]), "Wrong measurement matrix size")
    print("ok")

    # =========================================================================
    ## HadamSplit
    print("HadamSplit")
    from spyrit.core.meas import HadamSplit

    # constructor
    print("\tconstructor... ", end="")
    Ord = torch.rand(32, 32)
    meas_op = HadamSplit(400, 32, Ord)
    print("ok")

    # forward
    print("\tforward... ", end="")
    x = torch.rand([10, 32 * 32], dtype=torch.float)
    y = meas_op(x)
    assert_test(y.shape, torch.Size([10, 800]), "Wrong forward size")
    print("ok")

    # forward_H
    print("\tforward_H... ", end="")
    x = torch.rand([10, 32 * 32], dtype=torch.float)
    y = meas_op.forward_H(x)
    assert_test(y.shape, torch.Size([10, 400]), "Wrong forward_H size")
    print("ok")

    # adjoint
    print("\tadjoint... ", end="")
    x = torch.rand([10, 400], dtype=torch.float)
    y = meas_op.adjoint(x)
    assert_test(y.shape, torch.Size([10, 1024]), "Wrong adjoint size")
    print("ok")

    # get_mat
    print("\tget_mat... ", end="")
    H = meas_op.get_H()
    assert_test(H.shape, torch.Size([400, 1024]), "Wrong measurement matrix size")
    print("ok")

    # pinv
    print("\tpinv... ", end="")
    y = torch.rand([85, 400], dtype=torch.float)
    x = meas_op.pinv(y)
    assert_test(x.shape, torch.Size([85, 1024]), "Wrong pinv size")
    print("ok")

    # inverse
    print("\tinverse... ", end="")
    y = torch.rand([85, 32 * 32], dtype=torch.float)
    x = meas_op.inverse(y)
    assert_test(x.shape, torch.Size([85, 1024]), "Wrong inverse size")
    print("ok")

    # %% Test LinearRowSplit
    # from spyrit.core.meas import LinearRowSplit

    # # constructor
    # H_pos = np.random.rand(24, 64)
    # H_neg = np.random.rand(24, 64)
    # meas_op = LinearRowSplit(H_pos, H_neg)

    # # forward
    # x = torch.rand([10, 64, 92], dtype=torch.float)
    # y = meas_op(x)
    # print(y.shape)
    # assert_test(y.shape, torch.Size([10, 48, 92]), "Wrong forward size")

    # # forward_H
    # x = torch.rand([10, 64, 92], dtype=torch.float)
    # y = meas_op(x)
    # print(y.shape)
    # assert_test(y.shape, torch.Size([10, 48, 92]), "Wrong forward size")

    # # get_H
    # H = meas_op.get_H()
    # print(H.shape)
    # assert_test(H.shape, torch.Size([24, 64]), "Wrong measurement matrix size")

    # =========================================================================
    print("All tests passed for meas.py")
    print("==============================")
    return True


if __name__ == "__main__":
    test_core_meas()
