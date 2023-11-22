import numpy.typing as npt
import numpy as np
from typing import List, Tuple


def train(X: npt.NDArray,
          y: npt.NDArray,
          r: float,
          epochs: int):
    """
    Train an average perceptron

    :param X: training examples
    :param y: training labels
    :param r: learning rate
    :param epochs: number of epochs to perform

    :return: weight vector
    """

    w = np.zeros(X.shape[1])
    a = np.zeros_like(w)

    for _ in range(epochs):
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for i in indices:
            if y[i] * (w @ X[i]) <= 0:
                w += r * y[i] * X[i]

            a += w

    return a


def predict(x: npt.ArrayLike,
            a: npt.ArrayLike):
    """
    Predict the value of x using averaged perceptron

    :param x: x value
    :param a: averaged weight vector

    :return: predicted value
    """
    return np.sign(a @ x.T)
