import numpy.typing as npt
import numpy as np


def _dw(x_i: npt.ArrayLike,
        y_i: npt.ArrayLike,
        w: npt.ArrayLike) -> npt.ArrayLike:
    """
    The partial derivative of the squared error loss function with respect to w.

    :param x_i: the ith training example
    :param y_i: the ith training label
    :param w: the weight vector
    """
    return -1 * x_i * (y_i - predict(x_i, w))


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
    Performs stochastic gradient descent on the squared error loss function.

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
        i = np.random.choice(indices)

        w = w - r * _dw(X[i], Y[i], w)

        prev_cost = cost
        cost = loss(X, Y, w)
        costs.append(cost)
    return w, np.array(costs)
