import numpy.typing as npt
import numpy as np


def _dw(x_i: npt.ArrayLike,
        y_i: npt.ArrayLike,
        w: npt.ArrayLike,
        b: npt.ArrayLike) -> npt.ArrayLike:
    """
    The partial derivative of the squared error loss function with respect to w.

    :param x_i: the ith training example
    :param y_i: the ith training label
    :param w: the weight vector
    :param b: the bias
    """
    return -1 * x_i * (y_i - predict(x_i, w, b))


def _db(x_i: npt.ArrayLike,
        y_i: npt.ArrayLike,
        w: npt.ArrayLike,
        b: npt.ArrayLike) -> npt.ArrayLike:
    """
    The partial derivative of the squared error loss function with respect to b.

    :param x_i: the ith training example
    :param y_i: the ith training label
    :param w: the weight vector
    :param b: the bias
    """
    return -1 * (y_i - predict(x_i, w, b))


def predict(x: npt.ArrayLike,
            w: npt.ArrayLike,
            b: float):
    """
    Predict the value of x using linear regression

    :param x: x value
    :param w: weight vector
    :param b: bias term
    """
    return w @ x.T + b


def stochastic_grad_descent(x: npt.ArrayLike,
                            y: npt.ArrayLike,
                            w: npt.ArrayLike,
                            b: float,
                            r: float,
                            convergence_tol: float = 0.01) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Performs stochastic gradient descent on the squared error loss function.

    :param x: the training examples
    :param y: the training labels
    :param w: the weight vector
    :param b: the bias
    :param r: the learning rate
    :param num_iters: the number of iterations to perform
    """
    indices = np.arange(len(x))
    cost_delta = None

    # TODO: Do this

    while cost_delta > convergence_tol:
        i = np.random(indices)

        w = w - r * _dw(x[i], y[i], w, b)
        b = b - r * _db(x[i], y[i], w, b)

        new_cost = w @ x.T + b
    return w, b
