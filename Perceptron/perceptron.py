import numpy.typing as npt
import numpy as np


def train(X: npt.NDArray,
          y: npt.NDArray,
          r: float,
          epochs: int):
    """
    Train a standard perceptron

    :param X: training examples
    :param y: training labels
    :param r: learning rate
    :param epochs: number of epochs to perform

    :return: weight vector
    """

    w = np.zeros(X.shape[1])

    for _ in range(epochs):
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for i in indices:
            if y[i] * (w @ X[i]) <= 0:
                w = w + r * y[i] * X[i]

    return w


def predict(x: npt.ArrayLike,
            w: npt.ArrayLike):
    """
    Predict the value of x using linear regression

    :param x: x value
    :param w: weight vector

    :return: predicted value (sign of w @ x.T)
    """
    return np.sign(w @ x.T)
