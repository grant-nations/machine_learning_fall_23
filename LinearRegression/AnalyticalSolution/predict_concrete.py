import os
import numpy as np
from LinearRegression.BatchGradDescent import batch_grad_descent
import matplotlib.pyplot as plt
from numpy.linalg import inv

features = [
    "Cement",
    "Slag",
    "Fly ash",
    "Water",
    "SP",
    "Coarse Aggr",
    "Fine Aggr",
]

X = []
Y = []

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Data")
train_filename = os.path.join(data_dir, "concrete", "train.csv")
with open(train_filename) as f:
    for line in f:
        values = line.strip().split(',')
        # add all independent variables to x including a 1 for bias term
        X.append([float(val) for val in values[:-1]] + [1])
        Y.append(float(values[-1]))  # add dependent variable to y

X = np.array(X)
Y = np.array(Y)

w_star = inv(X.T @ X) @ X.T @ Y
loss = 0.5 * np.sum(np.power(Y - w_star @ X.T, 2))

print("weights: ", end='')
for _w in w_star:
    print(f"{_w}, ", end='')
print()
print(f"final cost: {loss}")