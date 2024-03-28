"""
Test for module time.py
Author: Romain Phan
"""

import torch
import math

from test_helpers import assert_test, assert_elementwise_equal


def test_core_time():

    print("\n*** Testing time.py ***")

    # =========================================================================
    ## DeformationField
    print("DeformationField")
    from spyrit.core.time import DeformationField

    # constructor
    print("\tconstructor... ", end="")
    n_frames = 10
    nx, ny = 64, 64
    matrix = torch.randn(n_frames, nx, ny, 2, dtype=torch.float)
    def_field = DeformationField(matrix)
    print("ok")

    # forward greyscale (1D)
    print("\tforward greyscale... ", end="")
    img = torch.randn(1, nx, ny, dtype=torch.float)
    warped_img = def_field(img, 0, n_frames)
    assert_test(
        warped_img.shape, torch.Size([10, 1, nx, ny]), "Wrong forward greyscale size"
    )
    print("ok")

    # forward color (3D)
    print("\tforward color... ", end="")
    img = torch.randn(3, nx, ny, dtype=torch.float)
    warped_img = def_field(img, 0, n_frames)
    assert_test(
        warped_img.shape, torch.Size([10, 3, nx, ny]), "Wrong forward color size"
    )
    print("ok")

    # forward color with batch of images
    print("\tforward color with batch of images... ", end="")
    batch_imgs = torch.randn(5, 3, nx, ny, dtype=torch.float)
    warped_batch_imgs = def_field(batch_imgs, 0, n_frames)
    assert_test(
        warped_batch_imgs.shape,
        torch.Size([5, 10, 3, nx, ny]),
        "Wrong forward color with batch of images size",
    )
    print("ok")

    # forward rotating clockwise
    print("\tforward rotating clockwise... ", end="")
    img = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float)
    v = torch.tensor([[[[-1.0, 1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, -1.0]]]])
    field = DeformationField(v, align_corners=True)
    warped_img = field(img, 0, 1)
    assert_elementwise_equal(
        warped_img,
        torch.tensor([[[3, 1], [4, 2]]], dtype=torch.float),
        "Wrong forward rotating clockwise",
    )
    print("ok")


    # =========================================================================
    ## AffineDeformationField
    print("AffineDeformationField")
    from spyrit.core.time import AffineDeformationField

    # constructor
    print("\tconstructor... ", end="")
    mat = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float)
    def f(t):
        return mat
    field = AffineDeformationField(f, 0, 0, 1, (64, 64))
    print("ok")

    # forward, test with a counter clockwise rotation
    print("\tforward... ", end="")
    def s(t):
        return math.sin(2 * math.pi * t)
    def c(t):
        return math.cos(2 * math.pi * t)
    def f(t):
        return torch.tensor([[c(t), -s(t), 0], [s(t), c(t), 0], [0, 0, 1]])
    img = torch.FloatTensor([[[1, 2], [3, 4]]])
    img_size = img.shape[-2:]
    # 4 frames, sampled at [0, 0.25, 0.5, 0.75]
    t0, t1, n_frames = 0, 0.75, 4
    field = AffineDeformationField(f, t0, t1, n_frames, img_size, align_corners=False)
    warped_img = field(img, 0, n_frames, "bilinear")
    expected_img = torch.FloatTensor(
        [
            [[[1, 2], [3, 4]]],
            [[[2, 4], [1, 3]]],
            [[[4, 3], [2, 1]]],
            [[[3, 1], [4, 2]]],
        ]
    )
    assert_elementwise_equal(
        torch.round(warped_img - expected_img, decimals=5),
        torch.zeros(4, 1, 2, 2),
        "Wrong forward 4 images",
    )
    print("ok")

    # test 4 frames with 10 images in a batch
    print("\tforward 4 frames with 10 images in a batch... ", end="")
    nx, ny = 2, 2
    batch_imgs = torch.randn(10, 3, nx, ny, dtype=torch.float)
    warped_batch_imgs = field(batch_imgs, 0, n_frames, "bilinear")
    assert_test(
        warped_batch_imgs.shape,
        torch.Size([10, 4, 3, nx, ny]),
        "Wrong forward 4 frames with 10 images in a batch size",
    )
    print("ok")

    # =========================================================================
    print("All tests passed for time.py")
    print("==============================")
    return True


if __name__ == "__main__":
    test_core_time()
