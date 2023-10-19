import numpy.typing as npt
import numpy as np


def _dw(X: npt.NDArray,
        Y: npt.NDArray,
        w: npt.ArrayLike) -> npt.ArrayLike:
    """
    The partial derivative of the squared error loss function with respect to w.

    :param X: the matrix of x training examples
    :param Y: the matrix of y training example labels
    :param w: the weight vector
    """
    return -1 * X.T @ (Y - w @ X.T)


def predict(x: npt.ArrayLike,
            w: npt.ArrayLike):
    """
    Predict the value of x using linear regression

    :param x: x value
    :param w: weight vector
    """
    return w @ x.T


def loss(X: npt.NDArray,
         Y: npt.NDArray,
         w: npt.ArrayLike) -> float:

    return 0.5 * np.sum(np.power(Y - w @ X.T, 2))


def train(X: npt.NDArray,
          Y: npt.NDArray,
          w: npt.ArrayLike,
          r: float,
          convergence_tol: float = 1e-6) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Performs batch gradient descent on the squared error loss function.

    :param x: the training examples
    :param y: the training labels
    :param w: the weight vector
    :param r: the learning rate
    :param num_iters: the number of iterations to perform
    """
    costs = []  # keep track of costs for plotting

    indices = np.arange(len(X))
    prev_cost = 0
    cost = convergence_tol + 1  # this 1 is arbitrary

    while np.abs(cost - prev_cost) > convergence_tol:
        w = w - r * _dw(X, Y, w)

        prev_cost = cost
        cost = loss(X, Y, w)
        costs.append(cost)
    return w, np.array(costs)
