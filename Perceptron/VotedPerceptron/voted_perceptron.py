import numpy.typing as npt
import numpy as np
from typing import List, Tuple


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

    m = 0
    w = np.zeros(X.shape[1])
    w_arr = [w]
    counts = [0]

    for _ in range(epochs):
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for i in indices:
            if y[i] * (w_arr[m] @ X[i]) <= 0:
                w_arr.append(w_arr[m] + r * y[i] * X[i])
                counts.append(1)
                m += 1
            else:
                counts[m] += 1

    return w_arr, counts


def predict(x: npt.ArrayLike,
            w_arr: List[npt.ArrayLike],
            counts: List[int]):
    """
    Predict the value of x using linear regression

    :param x: x value
    :param w_arr: List of weight vectors
    :param counts: list of counts

    :return: predicted value
    """
    return np.sign(np.sum(c * np.sign(w @ x.T) for w, c in zip(w_arr, counts)))
