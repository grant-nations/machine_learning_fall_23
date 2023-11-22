import numpy.typing as npt
import numpy as np
from typing import Union, Callable

def train(X: npt.NDArray,
          y: npt.NDArray,
          r: float,
          c: float,
          epochs: int = 100,
          lr_func: Union[Callable, None] = None,
          ):
    """
    Train an SVM classifier in the primal domain with stochastic gradient descent

    :param X: training examples
    :param y: training labels
    :param r: learning rate (initial lr if lr_func is not None)
    :param c: regularization parameter
    :param epochs: epochs to perform gradient descent
    :param lr_func: learning rate decay function
    """

    w = np.zeros(X.shape[1])
    N = len(X)

    lr = r
    loss = []

    for t in range(epochs):
        if lr_func is not None:
            lr = lr_func(lr, t)

        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for i in indices:
            if y[i] * (w @ X[i]) <= 1:
                w[:-1] -= lr * w[:-1] 
                w += lr * c * N * y[i] * X[i]
            else:
                w[:-1] = (1 - lr) * w[:-1]

            loss.append(1/2 * (w[:-1] @ w[:-1]) + c * N * np.sum(np.maximum(0, 1 - y * (w @ X.T))))
    
    return w, loss


def predict(x: npt.ArrayLike,
            w: npt.ArrayLike):
    """
    Predict the value of x using SVM in primal form

    :param x: x value
    :param w: weight vector

    :return: predicted value (sign of w @ x.T)
    """
    return np.sign(w @ x.T)