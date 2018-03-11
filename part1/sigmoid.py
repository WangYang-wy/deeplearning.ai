import math


def basic_sigmoid(x):
    """
    Compute sigmoid of x
    :param x: A scalar
    :return: sigmoid(x)
    """
    return 1 / (1 + math.exp(-x))
