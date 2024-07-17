"""
Test for module warp.py
Author: Romain Phan
"""

import warnings

warnings.filterwarnings(
    "ignore", ".*The deformation field goes beyond the range [-1;1].*"
)

import torch
import math

from test_helpers import *


def test_core_warp():

    print("\n*** Testing warp.py ***")

    # =========================================================================
    ## DeformationField
    print("DeformationField")
    from spyrit.core.warp import DeformationField

    # constructor
    print("\tconstructor... ", end="")
    n_frames = 10
    nx, ny = 64, 64
    matrix = torch.rand(n_frames, ny, nx, 2, dtype=torch.float64)
    def_field = DeformationField(matrix)
    print("ok")

    # constructor with rectangular size
    print("\tconstructor with rectangular size... ", end="")
    n_frames = 10
    height, width = 32, 64
    matrix = torch.rand(n_frames, height, width, 2, dtype=torch.float64)
    def_field = DeformationField(matrix)
    assert_shape(
        def_field.field.shape,
        torch.Size([10, 32, 64, 2]),
        "Wrong constructor with rectangular size",
    )
    print("ok")

    # forward greyscale (1D)
    print("\tforward greyscale... ", end="")
    matrix = torch.rand(n_frames, ny, nx, 2, dtype=torch.float64)
    def_field = DeformationField(matrix)
    img = torch.rand(1, nx * ny, dtype=torch.float64)
    warped_img = def_field(img, 0, n_frames)
    assert_shape(
        warped_img.shape, torch.Size([1, 10, nx * ny]), "Wrong forward greyscale size"
    )
    print("ok")

    # forward color (3D)
    print("\tforward color... ", end="")
    img = torch.rand(3, nx * ny, dtype=torch.float64)
    warped_img = def_field(img, 0, n_frames)
    assert_shape(
        warped_img.shape, torch.Size([3, 10, nx * ny]), "Wrong forward color size"
    )
    print("ok")

    # # forward color with batch of images
    # print("\tforward color with batch of images... ", end="")
    # batch_imgs = torch.rand(5, 3, nx, ny, dtype=torch.float)
    # warped_batch_imgs = def_field(batch_imgs, 0, n_frames)
    # assert_shape(
    #     warped_batch_imgs.shape,
    #     torch.Size([5, 10, 3, nx, ny]),
    #     "Wrong forward color with batch of images size",
    # )
    # print("ok")

    # forward rotating clockwise
    print("\tforward rotating clockwise... ", end="")
    img = torch.tensor([[1, 2, 3, 4]], dtype=torch.float64)
    v = torch.tensor(
        [[[[-1.0, 1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, -1.0]]]], dtype=torch.float64
    )
    field = DeformationField(v)
    warped_img = field(img, 0, 1)
    assert_equal_all(
        warped_img,
        torch.tensor([[3, 1, 4, 2]], dtype=torch.float64),
        "Wrong forward rotating clockwise",
    )
    print("ok")

    # =========================================================================
    ## AffineDeformationField
    print("AffineDeformationField")
    from spyrit.core.warp import AffineDeformationField

    # constructor
    print("\tconstructor... ", end="")
    mat = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)

    def f(t):
        return mat

    field = AffineDeformationField(f, torch.arange(10), (64, 64))
    print("ok")

    # forward, test with a counter clockwise rotation
    print("\tforward... ", end="")

    def s(t):
        return math.sin(2 * math.pi * t)

    def c(t):
        return math.cos(2 * math.pi * t)

    def f(t):
        return torch.tensor(
            [[c(t), -s(t), 0], [s(t), c(t), 0], [0, 0, 1]], dtype=torch.float64
        )

    img = torch.tensor([[1, 2, 3, 4]], dtype=torch.float64)
    img_size = (2, 2)
    # 4 frames, sampled at [0, 0.25, 0.5, 0.75]
    n_frames = 4
    time_vector = torch.tensor([0, 0.25, 0.5, 0.75])
    field = AffineDeformationField(f, time_vector, img_size)
    warped_img = field(img, 0, n_frames, "bilinear")
    expected_img = torch.tensor(
        [
            [1.0, 2, 3, 4],
            [2.0, 4, 1, 3],
            [4.0, 3, 2, 1],
            [3.0, 1, 4, 2],
        ],
        dtype=torch.float64,
    )
    assert_close_all(warped_img, expected_img, "Wrong forward 4 images")
    print("ok")

    # # test 4 frames with 10 images in a batch
    # print("\tforward 4 frames with 10 images in a batch... ", end="")
    # nx, ny = 2, 2
    # batch_imgs = torch.rand(10, 3, nx, ny, dtype=torch.float)
    # warped_batch_imgs = field(batch_imgs, 0, n_frames, "bilinear")
    # assert_shape(
    #     warped_batch_imgs.shape,
    #     torch.Size([10, 4, 3, nx, ny]),
    #     "Wrong forward 4 frames with 10 images in a batch size",
    # )
    # print("ok")

    # =========================================================================
    print("All tests passed for warp.py")
    print("==============================")
    return True


if __name__ == "__main__":
    test_core_warp()
