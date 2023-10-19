import os
import numpy as np
from LinearRegression.StochasticGradDescent import stochastic_grad_descent
# import matplotlib.pyplot as plt

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

w = np.zeros_like(X[0])

learning_rate = 0.004
w, costs = stochastic_grad_descent.train(X, Y, w, learning_rate)
print("weights: ", end='')
for _w in w:
    print(f"{_w}, ", end='')
print()
print(f"final cost: {costs[-1]}")

iterations = list(range(len(costs)))

# Plot costs vs iterations
# plt.plot(iterations, costs, label='Cost')
# plt.xlabel('Iterations')
# plt.ylabel('Cost')
# plt.title('Stochastic Gradient Descent Cost vs Iterations')
# plt.legend()
# plt.show()

