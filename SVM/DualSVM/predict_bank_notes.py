import os
import pandas as pd
from SVM.DualSVM.dual_svm import train, predict, predict_with_alpha
import numpy as np

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Data', 'bank_note')
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None)

X = train_data.iloc[:, :-1]
X = X.to_numpy()
y = train_data.iloc[:, -1].replace(0, -1)
y = y.to_numpy()


test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None)

X_test = test_data.iloc[:, :-1]
X_test = X_test.to_numpy()
y_test = test_data.iloc[:, -1].replace(0, -1)
y_test = y_test.to_numpy()

# PART A -----------------------

# print("------------- PART A ------------- \n")

# for c in [(100/873), (500/873), (700/873)]:
#     print(f"using c = {c}")
#     print("------------------------------\n")
#     w, b, _ = train(X, y, c)

#     print(f"Learned weight vector: {w}")
#     print(f"bias: {b}")

#     incorrect_predictions = 0

#     for X_i, y_i in zip(X, y):
#         if predict(X_i, w, b) != y_i:
#             incorrect_predictions += 1

#     print(f"Training error: {incorrect_predictions/len(X_test)}")

#     incorrect_predictions = 0

#     for X_i, y_i in zip(X_test, y_test):
#         if predict(X_i, w, b) != y_i:
#             incorrect_predictions += 1

#     print(f"Testing error: {incorrect_predictions/len(X_test)}\n")

# PART B -----------------------


def gaussian_kernel(x, y, gamma):
    N, _ = x.shape
    K = np.zeros((N, N))

    distances = np.abs(np.sum(x**2, axis=1, keepdims=True) - 2 * (x @ y.T) + np.sum(y.T**2, axis=0, keepdims=True))

    K = np.exp(-distances / gamma)

    return K


print("------------- PART B ------------- \n")

for c in [(100/873), (500/873), (700/873)]:
    print(f"using c = {c}")
    print("------------------------------\n")

    for gamma in [0.1, 0.5, 1, 5, 100]:

        knl = lambda x, y: gaussian_kernel(x, y, gamma)

        print(f"using gamma = {gamma}")
        print("------------------------------\n")

        w, b, alpha = train(X, y, c, kernel=knl)

        incorrect_predictions = 0

        for X_i, y_i in zip(X, y):
            if predict_with_alpha(X_i, X, y, alpha, b, kernel=knl) != y_i:
                incorrect_predictions += 1

        print(f"Training error: {incorrect_predictions/len(X_test)}")

        incorrect_predictions = 0

        for X_i, y_i in zip(X_test, y_test):
            if predict_with_alpha(X_i, X, y, alpha, b, kernel=knl) != y_i:
                incorrect_predictions += 1

        print(f"Testing error: {incorrect_predictions/len(X_test)}\n")
