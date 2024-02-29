"""
Test for module time.py
Author: Romain Phan
"""

import torch
import numpy as np

from spyrit.core.time import DeformationField, AffineDeformationField

from test_helpers import assert_test, assert_elementwise_equal


def test_core_time():

    ## Test DeformationField
    n_frames = 10
    nx, ny = 64, 64
    # constructor
    matrix = torch.randn(n_frames, nx, ny, 2, dtype=torch.float)
    def_field = DeformationField(matrix)

    # forward greyscale (1D)
    img = torch.randn(1, nx, ny, dtype=torch.float)
    warped_img = def_field(img, 0, n_frames)
    print("forward greyscale:", warped_img.shape)
    assert_test(
        warped_img.shape, 
        torch.Size([10, 1, nx, ny]), 
        "Wrong forward greyscale size"
    )

    # forward color (3D)
    img = torch.randn(3, nx, ny, dtype=torch.float)
    warped_img = def_field(img, 0, n_frames)
    print("forward color:", warped_img.shape)
    assert_test(
        warped_img.shape, 
        torch.Size([10, 3, nx, ny]), 
        "Wrong forward color size"
    )
    
    # forward color with batch of images
    batch_imgs = torch.randn(5, 3, nx, ny, dtype=torch.float)
    warped_batch_imgs = def_field(batch_imgs, 0, n_frames)
    print("forward color with batch of images:", warped_batch_imgs.shape)
    assert_test(
        warped_batch_imgs.shape, 
        torch.Size([5, 10, 3, nx, ny]), 
        "Wrong forward color with batch of images size"
    )

    # forward rotating clockwise
    img = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float)
    v = torch.tensor([[[[-1.0, 1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, -1.0]]]])
    field = DeformationField(v, align_corners=True)
    warped_img = field(img, 0, 1)
    print("forward rotating clockwise:", warped_img.shape)
    assert_elementwise_equal(
        warped_img,
        torch.tensor([[[3, 1], [4, 2]]], dtype=torch.float),
        "Wrong forward rotating clockwise",
    )

    # =========================================================================
    ## Test AffineDeformationField
    # constructor
    mat = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float)
    field = AffineDeformationField(mat)

    # forward, test with a counter clockwise rotation
    def s(t):
        return np.sin(2 * np.pi * t)

    def c(t):
        return np.cos(2 * np.pi * t)

    def f(t):
        return torch.tensor([[c(t), -s(t), 0], [s(t), c(t), 0], [0, 0, 1]])

    field = AffineDeformationField(f, align_corners=False)
    img = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float)
    # single image
    warped_img = field(img, 0.25)
    print("forward single image:", warped_img.shape)
    assert_elementwise_equal(
        warped_img,
        torch.tensor([[[[2, 4], [1, 3]]]], dtype=torch.float),
        "Wrong forward single image",
    )
    # 4 images, sampled at [0, 0.25, 0.5, 0.75]
    t0, t1, n_frames = 0, 0.75, 4
    warped_img = field(img, t0, t1, n_frames)
    print("forward 4 images:", warped_img.shape)
    assert_elementwise_equal(
        warped_img,
        torch.tensor(
            [
                [[[1, 2], [3, 4]]],
                [[[2, 4], [1, 3]]],
                [[[4, 3], [2, 1]]],
                [[[3, 1], [4, 2]]],
            ],
            dtype=torch.float,
        ),
        "Wrong forward 4 images",
    )

    # test 4 frames with 10 images in a batch
    batch_imgs = torch.randn(10, 3, nx, ny, dtype=torch.float)
    warped_batch_imgs = field(batch_imgs, t0, t1, n_frames)
    print("forward 4 frames with 5 images in a batch:", warped_batch_imgs.shape)
    assert_test(
        warped_batch_imgs.shape, 
        torch.Size([10, 4, 3, nx, ny]), 
        "Wrong forward 4 frames with 5 images in a batch size"
    )

    return True


if __name__ == "__main__":
    test_core_time()
    print("✓ All tests passed for time.py")
    print("================================================================\n")
