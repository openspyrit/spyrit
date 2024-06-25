"""
Test for module recon.py
"""

import warnings

warnings.filterwarnings("ignore", message="The dynamic measurement")

import torch
import math

from test_helpers import *
from spyrit.core.meas import HadamSplit, DynamicLinear, DynamicHadamSplit
from spyrit.core.noise import NoNoise
from spyrit.core.prep import SplitPoisson
from spyrit.core.warp import AffineDeformationField


def test_core_recon():

    print("\n*** Testing recon.py ***")

    # =========================================================================
    ## PseudoInverse
    print("PseudoInverse")
    from spyrit.core.recon import PseudoInverse

    # constructor
    print("\tconstructor... ", end="")
    recon_op = PseudoInverse()
    print("ok")

    # forward from random measurement
    print("\tforward from random measurement... ", end="")
    Ord = torch.rand([32, 32])
    meas_op = HadamSplit(400, 32, Ord)
    y = torch.rand([85, 400], dtype=torch.float)
    x = recon_op(y, meas_op)
    assert_shape(x.shape, torch.Size([85, 1024]), "Wrong forward size")
    print("ok")

    # forward from measured random image
    print("\tforward from measured random image... ", end="")
    B = 1
    H = 64
    img = torch.FloatTensor(B, H**2).uniform_(-1, 1)
    Ord = torch.rand([H, H])
    M = 64
    meas_op = HadamSplit(M, H, Ord)
    noise_op = NoNoise(meas_op)
    y = noise_op(img)
    split_op = SplitPoisson(1.0, meas_op)
    m = split_op(y)
    recon_op = PseudoInverse()
    z = recon_op(m, meas_op)
    assert_shape(z.shape, torch.Size([1, 4096]), "Wrong recon size")
    tensor_test = torch.linalg.norm(img - z) / torch.linalg.norm(img)
    print(f"ok - {tensor_test=}")

    # inverse from moving objcet, DynamicLinear, comparing images
    print("\tInverse from moving object, DynamicLinear, comparing images... ", end="")

    def rotate(t):
        ans = torch.tensor(
            [
                [math.cos(t * 2 * math.pi), -math.sin(t * 2 * math.pi), 0],
                [math.sin(t * 2 * math.pi), math.cos(t * 2 * math.pi), 0],
                [0, 0, 1],
            ],
            dtype=torch.float64,
        )
        return ans

    channels = 1
    H = 16
    img = torch.DoubleTensor(channels, H**2).uniform_(-1, 1)
    M = H**2
    H_matrix = torch.rand([M, M])
    meas_op = DynamicLinear(H_matrix)
    # deformation field
    time_vector = torch.linspace(0.25, M // 4, M)
    field = AffineDeformationField(rotate, time_vector, (H, H))
    img_motion = field(img, mode="bilinear")
    # measurement
    y = meas_op(img_motion)
    # build H_dyn and H_dyn_pinv
    meas_op.build_H_dyn(field, "bilinear")
    meas_op.build_H_dyn_pinv()
    # reconstruction
    recon_op = PseudoInverse()
    z = recon_op(y, meas_op)
    assert_shape(z.shape, torch.Size([channels, H**2]), "Wrong recon size")
    print("ok")

    # Inverse from moving object, DynamicHadamSplit, comparing images
    print(
        "\tInverse from moving object, DynamicHadamSplit, comparing images... ", end=""
    )
    # no particular order, keep native
    meas_op = DynamicHadamSplit(M, H)
    # deformation field
    time_vector = torch.linspace(0.25, M // 2, 2 * M)
    field = AffineDeformationField(rotate, time_vector, (H, H))
    img_motion = field(img)
    # measurement
    y = meas_op(img_motion)
    # build H_dyn and H_dyn_pinv
    meas_op.build_H_dyn(field)
    meas_op.build_H_dyn_pinv()
    # reconstruction
    z = recon_op(y, meas_op)
    assert_shape(z.shape, torch.Size([channels, H**2]), "Wrong recon size")
    assert_close_all(img, z, "Wrong recon value", atol=1e-6)
    print("ok")

    # inverse from moving object, DynamicLinear, comparing measurements
    print(
        "\tInverse from moving object, DynamicLinear, comparing measurements... ",
        end="",
    )
    # more random field, keep same image
    time_vector = torch.linspace(0, math.e**2, M)
    field = AffineDeformationField(rotate, time_vector, (H, H))
    img_motion = field(img)
    # measurement
    meas_op = DynamicLinear(H_matrix)
    y = meas_op(img_motion)
    # reconstruction
    meas_op.build_H_dyn(field)
    y_hat = img @ meas_op.H_dyn.T.to(img.dtype)
    assert_close_all(y, y_hat, "Wrong recon value", atol=1e-6)
    print("ok")

    # Inverse from moving object, DynamicHadamSplit, comparing measurements
    print(
        "\tInverse from moving object, DynamicHadamSplit, comparing measurements... ",
        end="",
    )
    # field
    time_vector = torch.linspace(0, math.e**2, 2 * M)
    field = AffineDeformationField(rotate, time_vector, (H, H))
    img_motion = field(img)
    # measurement
    meas_op = DynamicHadamSplit(M, H)
    y = meas_op(img_motion)
    # reconstruction
    meas_op.build_H_dyn(field)
    meas_op.build_H_dyn_pinv()
    z = recon_op(y, meas_op)
    y_hat = z @ meas_op.H_dyn.T.to(z.dtype)
    assert_close_all(y, y_hat, "Wrong recon value", atol=1e-5)
    print("ok")

    # =========================================================================
    ## PinvNet
    print("PinvNet")
    from spyrit.core.recon import PinvNet

    # constructor
    print("\tconstructor... ", end="")
    B, C, H, M = 10, 1, 64, 64**2
    Ord = torch.randn((H, H))
    meas = HadamSplit(M, H, Ord)
    noise = NoNoise(meas)
    prep = SplitPoisson(1.0, meas)
    recnet = PinvNet(noise, prep)
    print("ok")

    # forward
    print("\tforward... ", end="")
    x = torch.FloatTensor(B, C, H, H).uniform_(-1, 1)
    z = recnet(x)
    assert_shape(z.shape, torch.Size([10, 1, 64, 64]), "Wrong recon size")
    tensor_test = torch.linalg.norm(x - z) / torch.linalg.norm(x)
    print(f"ok - {tensor_test=}")

    # meas2img
    print("\tmeas2img... ", end="")
    x = torch.rand(B * C, 2 * M)
    z = recnet.meas2img(x)
    assert_shape(z.shape, torch.Size([10, 1, 64, 64]), "Wrong recon size")
    print("ok")

    # acquire
    print("\tacquire... ", end="")
    x = torch.FloatTensor(B, C, H, H).uniform_(-1, 1)
    z = recnet.acquire(x)
    assert_shape(z.shape, torch.Size([10, 8192]), "Wrong recon size")
    print("ok")

    # reconstruct
    print("\treconstruct... ", end="")
    x = torch.rand((B * C, 2 * M), dtype=torch.float)
    z = recnet.reconstruct(x)
    assert_shape(z.shape, torch.Size([10, 1, 64, 64]), "Wrong recon size")
    print("ok")

    # =========================================================================
    ## TikhonovMeasurementPriorDiag
    print("TikhonovMeasurementPriorDiag")
    from spyrit.core.recon import TikhonovMeasurementPriorDiag

    # constructor
    print("\tconstructor... ", end="")
    B, H, M = 85, 32, 512
    sigma = torch.rand([H**2, H**2])
    recon = TikhonovMeasurementPriorDiag(sigma, M)
    print("ok")

    # forward
    print("\tforward... ", end="")
    Ord = torch.ones((H, H))
    meas = HadamSplit(M, H, Ord)
    y = torch.rand([B, M], dtype=torch.float)
    x_0 = torch.zeros((B, H**2), dtype=torch.float)
    var = torch.zeros((B, M), dtype=torch.float)
    x = recon(y, x_0, var, meas)
    assert_shape(x.shape, torch.Size([85, 1024]), "Wrong recon size")
    print("ok")

    # =========================================================================
    ## DCNet
    print("DCNet")
    from spyrit.core.recon import DCNet

    # constructor
    print("\tconstructor... ", end="")
    B, C, H, M = 10, 1, 64, 64**2 // 2
    Ord = torch.ones((H, H))
    meas = HadamSplit(M, H, Ord)
    noise = NoNoise(meas)
    prep = SplitPoisson(1.0, meas)
    sigma = torch.rand([H**2, H**2])
    recnet = DCNet(noise, prep, sigma)
    print("ok")

    # forward
    print("\tforward... ", end="")
    x = torch.FloatTensor(B, C, H, H).uniform_(-1, 1)
    z = recnet(x)
    assert_shape(z.shape, torch.Size([10, 1, 64, 64]), "Wrong recon size")
    print("ok")

    # reconstruct
    print("\treconstruct... ", end="")
    x = torch.rand((B * C, 2 * M), dtype=torch.float)
    z = recnet.reconstruct(x)
    assert_shape(z.shape, torch.Size([10, 1, 64, 64]), "Wrong recon size")
    print("ok")

    # =========================================================================
    print("All tests passed for recon.py")
    print("===============================")
    return True


if __name__ == "__main__":
    test_core_recon()
