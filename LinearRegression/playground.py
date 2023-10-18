import numpy as np
from numpy.linalg import inv

X = np.array([[1, -1, 2, 1], [1, 1, 3, 1],
              [-1, 1, 0, 1], [1, 2, -4, 1], [3, -1, -1, 1]])
Y = np.array([[1], [4], [-1], [-2], [0]])

w_star = inv(X.T @ X) @ X.T @ Y
print(w_star)

print("\n-----------------------------------------------\n")


def dw(x_i, y_i, w, b):
    return -1 * x_i * (y_i - (w.T @ x_i + b))


def db(x_i, y_i, w, b):
    return -1 * (y_i - (w.T @ x_i + b))


def stochastic_grad_descent(x, y, w, b, r, convergence_tol):

    indices = np.arange(len(x))
    for _ in range(num_iters):
        i = np.random(indices)
        step_dw = dw(x[i], y[i], w, b)
        step_db = db(x[i], y[i], w, b)

        w = w - r * step_dw
        b = b - r * step_db

    return w, b


X = np.array([[1, -1, 2], [1, 1, 3], [-1, 1, 0], [1, 2, -4], [3, -1, -1]])
w = np.array([0, 0, 0])
b = 0

w_new, b_new = stochastic_grad_descent(X, Y, w, b, 0.1, 5)
# print(w_new, b_new)

print("\n-----------------------------------------------\n")

# below is check for problem 5b

# def grad_descent(X, Y, w, b, r, num_iters):
#     for _ in range(num_iters):
#         dw_sum = 0
#         db_sum = 0
#         for i in range(len(X)):
#             dw_sum += dw(X[i], Y[i], w, b)
#             db_sum += db(X[i], Y[i], w, b)

#         w = w - r * dw_sum
#         b = b - r * db_sum

#         print(f"dw: {dw_sum}, db: {db_sum}")
#     return w, b


# w = np.array([-1, 1, -1])
# b = -1

# w_new, b_new = grad_descent(X, Y, w, b, 0.1, 1)
