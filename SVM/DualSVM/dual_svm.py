from scipy.optimize import minimize
import numpy.typing as npt
import numpy as np

def train(X: npt.NDArray,
          y: npt.ArrayLike,
          c: int):
    """
    Train the dual form of an SVM classifier

    :param X: training examples
    :param y: training labels
    :param c: hyperparameter
    """
    
    def _equality_constraint(alpha):
        return np.sum(alpha * y)

    # below is the inefficient way of doing this

    # def _dual_svm_loss(alpha):

    #     loss = 0

    #     for i in range(len(X)):
    #         for j in range(len(X)):
    #             loss += (1/2) * y[i] * y[j] * alpha[i] * alpha[j] * X[i] @ X[i]
        
    #     loss -= np.sum(alpha)

    #     return loss
    
    def _dual_svm_loss(alpha):
        return (1/2) * alpha.T @ (y[:, np.newaxis] * X) @ (y[:, np.newaxis] * X).T @ alpha - np.sum(alpha)    
        
    alpha_guess = np.zeros_like(y)

    constraints = [{'type': 'eq', 'fun': _equality_constraint}]

    res = minimize(_dual_svm_loss,
                    alpha_guess,
                    method='SLSQP',
                    constraints=constraints,
                    bounds=[(0, c)] * len(y))
    
    alpha_optimal = res.x
    
    w_optimal = np.sum(alpha_optimal[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)

    b_optimal = 0
    b_optimal_count = 0

    for i, a in enumerate(alpha_optimal):
        if a > 0 and a < c:
            b_optimal += (y[i] - w_optimal @ X[i].T)
            b_optimal_count += 1

    b_optimal /= b_optimal_count

    return w_optimal, b_optimal

def predict(x: npt.ArrayLike,
            w: npt.ArrayLike,
            b: float):
    """
    Predict the value of x using SVM in primal form

    :param x: x value
    :param w: weight vector
    :param b: bias value

    :return: predicted value
    """
    
    return np.sign(w @ x.T + b)
