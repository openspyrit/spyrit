import sys
import torch


def error(disp):
    print("_" * 80)
    print("ERROR:\n" + disp)
    sys.exit(-1)


def assert_shape(shape1, shape2, message):
    if not (shape1 == shape2):
        print(message)
        error(str(shape1) + "\nis not equal to\n" + str(shape2))


def assert_equal(elem1, elem2, message):
    if elem1 != elem2:
        print(message)
        error(str(elem1) + "\nis not equal to\n" + str(elem2))


def assert_equal_all(tensor1, tensor2, message):
    if (tensor1 != tensor2).any():
        print(message)
        error(str(tensor1) + "\nis not equal to\n" + str(tensor2))


def assert_close_all(tensor1, tensor2, message, **kgwargs):
    if not torch.allclose(tensor1, tensor2, **kgwargs):
        print(message)
        error(str(tensor1) + "\nis not close to\n" + str(tensor2))
