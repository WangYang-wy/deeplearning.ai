import numpy as np


def initial_with_zeros(dim):
    """
    initial with zeros.
    :param dim:
    :return:
    """
    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def propagate(w, b, X, Y):
    """

    :param w:
    :param b:
    :param X:
    :param Y:
    :return:
    """
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(np.dot(Y, np.log(A.T)) + np.dot(np.log(1 - A), (1 - Y).T)) / m

    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {
        "dw": dw,
        "db": db
    }
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """

    :param w:
    :param b:
    :param X:
    :param Y:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and 0 == i % 100:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {
        "w": w,
        "b": b
    }

    grads = {
        "dw": dw,
        "db": db
    }

    return params, grads, costs


def predict(w, b, X):
    """

    :param w:
    :param b:
    :param X:
    :return:
    """
    m = X.shape[1]
    y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0][i] <= 0.5:
            A[0][i] = 0
        else:
            A[0][i] = 1

    y_prediction = A
    assert (y_prediction.shape == (1, m))
    return y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """

    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """
    w, b = initial_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    y_prediction_test = predict(w, b, X_test)
    y_prediction_train = predict(w, b, X_train)

    print("trains accuracy: {}".format(100 - np.mean(np.abs(y_prediction_train - Y_train)) * 100))
    print("test accuracy: {}".format(100 - np.mean(np.abs(y_prediction_test - Y_test)) * 100))

    d = {
        "costs": costs,
        "y_prediction_test": y_prediction_test,
        "y_prediction_train": y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    return d


if __name__ == '__main__':
    print(1)
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2, -1], [3, 4, -3.2]]), np.array([[1, 0, 1]])
    params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

    print("w = ", str(params["w"]))
    print("b = ", str(params["b"]))
    print("dw = ", str(grads["dw"]))
    print("db = ", str(grads["db"]))
