import os
import numpy as np
from LinearRegression.StochasticGradDescent import stochastic_grad_descent

features = [
    "Cement",
    "Slag",
    "Fly ash",
    "Water",
    "SP",
    "Coarse Aggr",
    "Fine Aggr",
]

x = []
y = []

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Data")
train_filename = os.path.join(data_dir, "concrete", "train.csv")
with open(train_filename) as f:
    for line in f:
        values = line.strip().split(',')
        x.append(values[:-1])  # add all independent variables to x
        y.append(values[-1])  # add dependent variable to y

x = np.array(x)
y = np.array(y)

w = np.zeros_like(x)
b = 0

learning_rate = 0.01
w, b = stochastic_grad_descent(x, y, w, b, learning_rate)

