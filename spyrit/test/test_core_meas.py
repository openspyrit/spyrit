"""
Created on Mon Feb 13 18:04:11 2023

@author: ducros & phan
"""

import warnings

# for self defined warnings
warnings.filterwarnings("ignore", message="The dynamic")

import torch
from test_helpers import *


def test_core_meas():

    print("\n*** Testing meas.py ***")

    # =========================================================================
    ## _Base: do not test constructor
    print("_Base")
    from spyrit.core.meas import _Base
    from spyrit.core.meas import Linear

    # attributes, no Ord no meas_shape
    print("\tattributes, no Ord no meas_shape... ", end="")
    H_static = torch.rand(400, 2500)
    meas_op = Linear(H_static)
    assert_equal_all(meas_op._param_H_static.data, H_static, "Wrong H_static")
    assert_equal(meas_op.M, 400, "Wrong M")
    assert_equal(meas_op.N, 2500, "Wrong N")
    assert_equal(meas_op.h, 50, "Wrong h")
    assert_equal(meas_op.w, 50, "Wrong w")
    assert_equal(meas_op.meas_shape, torch.Size([50, 50]), "Wrong meas_shape")
    assert_equal_all(meas_op.indices, torch.arange(400), "Wrong indices")
    # test reindex method
    test_reindex = torch.randn(400)
    test_reindexed = meas_op.reindex(test_reindex, "rows", False)
    assert_equal_all(test_reindexed, test_reindex, "Wrong reindex method")
    test_reindexed = meas_op.reindex(test_reindex, "cols", False)
    assert_equal_all(test_reindexed, test_reindex, "Wrong reindex method")
    # test repr method
    meas_op.__repr__()
    print("ok")

    # attributes, with Ord, no meas_shape
    print("\tattributes, with Ord, no meas_shape... ", end="")
    H_static = torch.rand(400, 2500)
    Ord = torch.arange(0, 400, 1)
    meas_op = Linear(H_static, Ord=Ord)
    assert_equal_all(meas_op.Ord, Ord, "Wrong Ord")
    assert_equal_all(meas_op.indices, torch.arange(399, -1, -1), "Wrong indices")
    assert_equal_all(
        meas_op._param_H_static.data, H_static.flip(0), "Wrong sorting with Ord"
    )
    # Ord setter
    Ord_new = torch.arange(399, -1, -1)  # put back in original order
    meas_op.Ord = Ord_new
    assert_equal_all(meas_op.Ord, Ord_new, "Wrong Ord")
    assert_equal_all(meas_op.indices, torch.arange(400), "Wrong indices")
    assert_equal_all(meas_op._param_H_static.data, H_static, "Wrong sorting with Ord")
    print("ok")

    # attributes, no Ord, with meas_shape
    print("\tattributes, no Ord, with meas_shape... ", end="")
    H_static = torch.rand(400, 2500)
    meas_shape = (25, 100)  # height, width
    meas_op = Linear(H_static, meas_shape=meas_shape)
    assert_equal(meas_op.h, 25, "Wrong h")
    assert_equal(meas_op.w, 100, "Wrong w")
    assert_equal(meas_op.meas_shape, (25, 100), "Wrong meas_shape")
    print("ok")

    # =========================================================================
    ## Linear
    print("Linear")

    # constructor
    print("\tconstructor... ", end="")
    H = torch.rand(400, 2500)
    meas_op = Linear(H)
    print("ok")

    # forward
    print("\tforward... ", end="")
    x = torch.rand([10, 2500], dtype=torch.float)
    y = meas_op(x)
    assert_shape(y.shape, torch.Size([10, 400]), "Wrong forward size")
    print("ok")

    # adjoint
    print("\tadjoint... ", end="")
    x = torch.rand([10, 400], dtype=torch.float)
    y = meas_op.adjoint(x)
    assert_shape(y.shape, torch.Size([10, 2500]), "Wrong adjoint size")
    print("ok")

    # get_mat
    print("\tget_mat... ", end="")
    H = meas_op.H
    assert_shape(H.shape, torch.Size([400, 2500]), "Wrong get_mat size")
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
    assert_shape(y.shape, torch.Size([10, 800]), "Wrong forward size")
    print("ok")

    # forward_H
    print("\tforward_H... ", end="")
    x = torch.rand([10, 2500], dtype=torch.float)
    y = meas_op.forward_H(x)
    assert_shape(y.shape, torch.Size([10, 400]), "Wrong forward_H size")
    print("ok")

    # adjoint
    print("\tadjoint... ", end="")
    x = torch.rand([10, 400], dtype=torch.float)
    y = meas_op.adjoint(x)
    assert_shape(y.shape, torch.Size([10, 2500]), "Wrong adjoint size")
    print("ok")

    # get_mat
    print("\tget_mat... ", end="")
    H = meas_op.H
    assert_shape(H.shape, torch.Size([400, 2500]), "Wrong measurement matrix size")
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
    assert_shape(y.shape, torch.Size([10, 800]), "Wrong forward size")
    print("ok")

    # forward_H
    print("\tforward_H... ", end="")
    x = torch.rand([10, 32 * 32], dtype=torch.float)
    y = meas_op.forward_H(x)
    assert_shape(y.shape, torch.Size([10, 400]), "Wrong forward_H size")
    print("ok")

    # adjoint
    print("\tadjoint... ", end="")
    x = torch.rand([10, 400], dtype=torch.float)
    y = meas_op.adjoint(x)
    assert_shape(y.shape, torch.Size([10, 1024]), "Wrong adjoint size")
    print("ok")

    # get_mat
    print("\tget_mat... ", end="")
    H = meas_op.H
    assert_shape(H.shape, torch.Size([400, 1024]), "Wrong measurement matrix size")
    print("ok")

    # pinv
    print("\tpinv... ", end="")
    y = torch.rand([85, 400], dtype=torch.float)
    x = meas_op.pinv(y)
    assert_shape(x.shape, torch.Size([85, 1024]), "Wrong pinv size")
    print("ok")

    # inverse
    print("\tinverse... ", end="")
    Ord = torch.rand(32, 32)
    # must build full matrix to use self.inverse()
    meas_op = HadamSplit(1024, 32, Ord)
    x = torch.rand([85, 32 * 32], dtype=torch.float32)
    y = meas_op(x)
    # subtract positive and negative parts
    y_sum = y[:, 0::2] - y[:, 1::2]
    x_inv = meas_op.inverse(y_sum)
    assert_shape(x_inv.shape, torch.Size([85, 1024]), "Wrong inverse size")
    assert_close_all(x_inv, x, "Wrong inverse", atol=1e-5)
    print("ok")

    # =========================================================================
    ## DynamicLinear
    print("DynamicLinear")
    from spyrit.core.meas import DynamicLinear

    # constructor
    print("\tconstructor... ", end="")
    # 400 (20x20) measurements for a 50x50 image
    H = torch.rand(400, 2500)
    meas_op = DynamicLinear(H)
    assert_equal(meas_op.img_h, 50, "Wrong img_h")
    assert_equal(meas_op.img_w, 50, "Wrong img_w")
    assert_equal(meas_op.img_shape, torch.Size([50, 50]), "Wrong img_shape")
    # try with a rectangular image
    meas_op = DynamicLinear(H, img_shape=(60, 50))
    assert_equal(meas_op.img_h, 60, "Wrong img_h")
    assert_equal(meas_op.img_w, 50, "Wrong img_w")
    assert_equal(meas_op.img_shape, torch.Size([60, 50]), "Wrong img_shape")
    print("ok")

    # constructor with image size too small
    print("\tconstructor with image size too small... ", end="")
    try:
        meas_op = DynamicLinear(H, img_shape=(5, 5))
    except ValueError as e:
        s = "The image shape must be at least as large as the measurement"
        if s in str(e):
            print("ok")
        else:
            raise e

    # forward
    print("\tforward... ", end="")
    # a batch of 10 motion pictures of 400 images each, of size 50x50
    H = torch.rand(400, 2500)
    meas_op = DynamicLinear(H)
    x = torch.rand([10, 400, 2500], dtype=torch.float)
    y = meas_op(x)
    assert_shape(y.shape, torch.Size([10, 400]), "Wrong forward size")
    print("ok")

    # Build dynamic measurement matrix
    from spyrit.core.warp import AffineDeformationField

    print("\tBuild dynamic measurement matrix... ", end="")
    H = torch.rand(400, 2500, dtype=torch.float64)
    meas_op = DynamicLinear(H)

    # deformation field that flips the image in x/y axis
    def f(t):
        return -torch.eye(3, dtype=torch.float64)

    time_vector = torch.linspace(0, 1, 400)
    field = AffineDeformationField(f, time_vector, img_shape=(50, 50))
    meas_op.build_H_dyn(field, "bilinear")
    assert_close_all(
        meas_op.H, H.flip(1), "Wrong dynamic measurement matrix", atol=1e-6
    )
    meas_op.build_H_dyn(field, "bicubic")
    print("ok")

    # build pseudo inverse H_dyn_pinv
    print("\tBuild pseudo inverse H_dyn_pinv... ", end="")
    meas_op.build_H_dyn_pinv()
    assert_shape(
        meas_op.H_dyn_pinv.shape, torch.Size([2500, 400]), "Wrong H_dyn_pinv size"
    )
    print("ok")

    # pinv method
    print("\tpinv... ", end="")
    y = torch.rand([10, 400], dtype=torch.float)
    x = meas_op.pinv(y)
    assert_shape(x.shape, torch.Size([10, 2500]), "Wrong pinv size")
    print("ok")

    # reset Ord
    print("\treset Ord... ", end="")
    Ord = torch.rand(400)
    meas_op.Ord = Ord
    # check that H_dyn and H_dyn_pinv are deleted
    try:
        meas_op.H_dyn
        error("H_dyn should have been deleted")
    except AttributeError:
        pass
    try:
        meas_op.H_dyn_pinv
        error("H_dyn_pinv should have been deleted")
    except AttributeError:
        pass
    # check that Ord is reset
    assert_equal_all(meas_op.Ord, Ord, "Wrong Ord")
    print("ok")

    # =============================================================================
    ## DynamicLinearSplit
    print("DynamicLinearSplit")
    from spyrit.core.meas import DynamicLinearSplit

    # constructor and attributes
    print("\tconstructor... ", end="")
    H = torch.randn(400, 2500)
    Ord = torch.arange(400)
    meas_op = DynamicLinearSplit(H, Ord, (50, 50), (70, 70))
    assert_shape(meas_op.P.shape, torch.Size([800, 2500]), "Wrong P size")
    print("ok")

    # construction of P
    print("\tconstruction of P... ", end="")
    H = torch.rand(400, 2500)  # only positive values
    meas_op = DynamicLinearSplit(H)
    assert_equal_all(meas_op.P[::2, :], H, "Wrong P")
    H = -torch.rand(400, 2500)  # only negative values
    meas_op = DynamicLinearSplit(H)
    assert_equal_all(meas_op.P[1::2, :], -H, "Wrong P")
    print("ok")

    # forward
    print("\tforward... ", end="")
    Ord = torch.arange(399, -1, -1)  # keep natural order, most important before
    H = torch.rand(400, 2500)  # only positive values
    meas_op = DynamicLinearSplit(H, Ord)
    x = torch.rand([10, 800, 2500], dtype=torch.float)
    y = meas_op(x)
    assert_shape(y.shape, torch.Size([10, 800]), "Wrong forward size")
    print("ok")

    # forward_H
    print("\tforward_H... ", end="")
    x = torch.rand([10, 400, 2500], dtype=torch.float)
    y = meas_op.forward_H(x)
    assert_shape(y.shape, torch.Size([10, 400]), "Wrong forward_H size")
    print("ok")

    # re-set Ord
    print("\tre-set Ord... ", end="")
    Ord = torch.arange(400)  # now in reverse order
    # H has only positive values
    meas_op.Ord = Ord
    assert_equal_all(meas_op.P[::2, :], H.flip(0), "Wrong P")
    print("ok")

    # =========================================================================
    ## DynamicHadamard
    print("DynamicHadamard")
    from spyrit.core.meas import DynamicHadamSplit

    # constructor
    print("\tconstructor... ", end="")
    Ord = torch.rand(32, 32)
    meas_op = DynamicHadamSplit(400, 32, Ord, (50, 50))
    print("ok")

    # attributes
    print("\tattributes... ", end="")
    assert_equal(meas_op.M, 400, "Wrong M")
    assert_equal(meas_op.N, 1024, "Wrong N")
    assert_equal(meas_op.h, 32, "Wrong h")
    assert_equal(meas_op.w, 32, "Wrong w")
    assert_equal(meas_op.meas_shape, torch.Size([32, 32]), "Wrong meas_shape")
    assert_equal(meas_op.img_h, 50, "Wrong img_h")
    assert_equal(meas_op.img_w, 50, "Wrong img_w")
    assert_equal(meas_op.img_shape, torch.Size([50, 50]), "Wrong img_shape")
    print("ok")

    # =========================================================================
    print("All tests passed for meas.py\n============================")
    return True


if __name__ == "__main__":
    test_core_meas()
