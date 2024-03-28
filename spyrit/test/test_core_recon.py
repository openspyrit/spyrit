"""
Test for module recon.py
"""

import torch

from test_helpers import assert_test

from spyrit.core.meas import HadamSplit
from spyrit.core.noise import NoNoise
from spyrit.core.prep import SplitPoisson


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

    # EXAMPLE 1
    # forward from random measurement
    print("\tforward from random measurement... ", end="")
    Ord = torch.rand([32, 32])
    meas_op = HadamSplit(400, 32, Ord)
    y = torch.rand([85, 400], dtype=torch.float)
    x = recon_op(y, meas_op)
    assert_test(x.shape, torch.Size([85, 1024]), "Wrong forward size")
    print("ok")

    # EXAMPLE 2
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
    assert_test(z.shape, torch.Size([1, 4096]), "Wrong recon size")
    tensor_test = torch.linalg.norm(img - z) / torch.linalg.norm(img)
    print(f"ok - {tensor_test=}")

    # =========================================================================
    ## PinvNet
    print("PinvNet")
    from spyrit.core.recon import PinvNet

    # constructor
    print("\tconstructor... ", end="")
    B, C, H, M = 10, 1, 64, 64**2
    Ord = torch.ones((H, H))
    meas = HadamSplit(M, H, Ord)
    noise = NoNoise(meas)
    prep = SplitPoisson(1.0, meas)
    recnet = PinvNet(noise, prep)
    print("ok")

    # forward
    print("\tforward... ", end="")
    x = torch.FloatTensor(B, C, H, H).uniform_(-1, 1)
    z = recnet(x)
    assert_test(z.shape, torch.Size([10, 1, 64, 64]), "Wrong recon size")
    tensor_test = torch.linalg.norm(x - z) / torch.linalg.norm(x)
    print(f"ok - {tensor_test=}")

    # meas2img
    print("\tmeas2img... ", end="")
    x = torch.rand(B * C, 2 * M)
    z = recnet.meas2img(x)
    assert_test(z.shape, torch.Size([10, 1, 64, 64]), "Wrong recon size")
    print("ok")

    # acquire
    print("\tacquire... ", end="")
    x = torch.FloatTensor(B, C, H, H).uniform_(-1, 1)
    z = recnet.acquire(x)
    assert_test(z.shape, torch.Size([10, 8192]), "Wrong recon size")
    print("ok")

    # reconstruct
    print("\treconstruct... ", end="")
    x = torch.rand((B * C, 2 * M), dtype=torch.float)
    z = recnet.reconstruct(x)
    assert_test(z.shape, torch.Size([10, 1, 64, 64]), "Wrong recon size")
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
    assert_test(x.shape, torch.Size([85, 1024]), "Wrong recon size")
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
    assert_test(z.shape, torch.Size([10, 1, 64, 64]), "Wrong recon size")
    print("ok")

    # reconstruct
    print("\treconstruct... ", end="")
    x = torch.rand((B * C, 2 * M), dtype=torch.float)
    z = recnet.reconstruct(x)
    assert_test(z.shape, torch.Size([10, 1, 64, 64]), "Wrong recon size")
    print("ok")

    # =========================================================================
    print("All tests passed for recon.py")
    print("===============================")
    return True


if __name__ == "__main__":
    test_core_recon()
