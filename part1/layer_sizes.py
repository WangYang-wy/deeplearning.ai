import numpy as np


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def layer_size(X, Y):
    """

    :param X:
    :param Y:
    :return:
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y


def initial_parameters(n_x, n_h, n_y):
    """

    :param n_x:
    :param n_h:
    :param n_y:
    :return:
    """
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.001
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.001
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


def forward_propagation(X, parameters):
    """

    :param X:
    :param parameters:
    :return:
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache


def compute_cost(A2, Y, parameters):
    """

    :param A2:
    :param Y:
    :param parameters:
    :return:
    """
    m = Y.shape[1]
    log_probs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = - np.sum(log_probs) / m

    cost = np.squeeze(cost)

    return cost


def backward_propagation(parameters, cache, X, Y):
    """

    :param parameters:
    :param cache:
    :param X:
    :param Y:
    :return:
    """
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    update parameters by learning_rate.
    :param parameters: parameters.
    :param grads: grads.
    :param learning_rate: learning rate.
    :return: updated parameters.
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - dW1 * learning_rate
    b1 = b1 - db1 * learning_rate
    W2 = W2 - dW2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def nn_model(X, Y, n_h, num_iteration=10000, print_cost=False):
    """

    :param X:
    :param Y:
    :param n_h:
    :param num_iteration:
    :param print_cost:
    :return:
    """
    np.random.seed(3)
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]
    parameters = initial_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iteration):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

        if print_cost and 0 == i % 100:
            print("cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    """

    :param parameters:
    :param X:
    :return:
    """
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    return predictions


def linear_forward(A, W, b):
    """

    :param A:
    :param W:
    :param b:
    :return:
    """
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """

    :param A_prev:
    :param W:
    :param b:
    :param activation:
    :return:
    """
    if "sigmoid" == activation:
        Z, linear_cache = linear
