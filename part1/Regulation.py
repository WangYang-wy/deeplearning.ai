import numpy as np


def l1(y_hat, y):
    loss = sum(abs(y - y_hat))
    return loss


def l2(y_hat, y):
    loss = np.dot(y - y_hat, y - y_hat)
    return loss


if __name__ == '__main__':
    y_hat = np.array([0.9, 0.2, 0.1, 0.4, 0.9])
    y = np.array([1, 0, 0, 1, 1])
    print("L2 = ", l2(y_hat, y))
    print("L1 = ", l1(y_hat, y))
