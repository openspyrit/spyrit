import sys


def assert_test(condition1, condition2, message):
    if not (condition1 == condition2):
        print("ERROR: " + str(condition1) + " is not equal to " + str(condition2))
        print(message)
        sys.exit(-1)


def assert_elementwise_equal(elem1, elem2, message):
    if not (elem1 == elem2).all():
        print("ERROR: " + str(elem1) + " is not equal to " + str(elem2))
        print(message)
        sys.exit(-1)
