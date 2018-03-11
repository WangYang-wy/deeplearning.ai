import numpy as np

x = np.array([1, 2, 3])
print(np.exp(x))

x_1 = np.array([1, 2, 3])
print(x + 3)


def sigmoid(x):
    """
    Compute sigmoid of x, using numpy.
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """

    :param x:
    :return:
    """
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds


def image2vector(image):
    """

    :param image:
    :return:
    """
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2]), 1)
    return v


def normalize_rows(x):
    """

    :param x:
    :return:
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_return = x / x_norm
    return x_return


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


if __name__ == '__main__':
    x = np.array([
        [0, 3, 4],
        [1, 6, 4]
    ])

    print(normalize_rows(x))
