import os
import pandas as pd
import numpy as np
from NeuralNetworks.nn import ThreeLayerNN
import matplotlib.pyplot as plt

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data', 'bank_note')
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None)

X = train_data.iloc[:, :-1]
X = X.to_numpy()
y = train_data.iloc[:, -1]
y = y.to_numpy()

test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None)

X_test = test_data.iloc[:, :-1]
X_test = X_test.to_numpy()
y_test = test_data.iloc[:, -1]
y_test = y_test.to_numpy()



print("------------- PART A ------------- \n")

w1 = np.array([[-2, 2],
               [-3, 3]])
w2 = np.array([[-2, 2],
               [-3, 3]])
w3 = np.array([2, -1.5])
b1 = np.array([-1, 1])
b2 = np.array([-1, 1])
b3 = -1

nn = ThreeLayerNN(w1, w2, w3, b1, b2, b3)

x_t = np.array([1, 1])
y_t = 1

nn.step(x_t, y_t, verbose=True)

# exit()
print("------------- PART B ------------- \n")

input_dim = len(X[0])
hidden_dims = [5, 10, 25, 50, 100]
epochs = 10
gamma_0 = 0.1
d = 1

for hidden_dim in hidden_dims:
    print(f"Hidden dim: {hidden_dim}")

    w1 = np.random.randn(input_dim, hidden_dim)
    w2 = np.random.randn(hidden_dim, hidden_dim)
    w3 = np.random.randn(hidden_dim)  # the "1" dim is implied here
    b1 = np.random.randn(hidden_dim)
    b2 = np.random.randn(hidden_dim)
    b3 = np.random.randn()

    nn = ThreeLayerNN(w1, w2, w3, b1, b2, b3)
    loss_history = nn.train(X, y, epochs, gamma_0, d)

    # plt.plot(loss_history)
    # plt.xlabel("Update")
    # plt.ylabel("Loss")
    # plt.title(f"Loss vs Update for hidden dim {hidden_dim}")
    # plt.show()

    incorrect_predictions = 0

    for X_i, y_i in zip(X, y):
        if nn.predict(X_i) != y_i:
            incorrect_predictions += 1

    print(f"Training error: {incorrect_predictions/len(X_test)}")

    incorrect_predictions = 0

    for X_i, y_i in zip(X_test, y_test):
        if nn.predict(X_i) != y_i:
            incorrect_predictions += 1

    print(f"Testing error: {incorrect_predictions/len(X_test)}\n")

print("------------- PART C ------------- \n")

for hidden_dim in hidden_dims:
    print(f"Hidden dim: {hidden_dim}")

    w1 = np.zeros((input_dim, hidden_dim))
    w2 = np.zeros((hidden_dim, hidden_dim))
    w3 = np.zeros(hidden_dim)  # the "1" dim is implied here
    b1 = np.zeros(hidden_dim)
    b2 = np.zeros(hidden_dim)
    b3 = 0

    nn = ThreeLayerNN(w1, w2, w3, b1, b2, b3)
    loss_history = nn.train(X, y, epochs, gamma_0, d)

    incorrect_predictions = 0

    for X_i, y_i in zip(X, y):
        if nn.predict(X_i) != y_i:
            incorrect_predictions += 1

    print(f"Training error: {incorrect_predictions/len(X_test)}")

    incorrect_predictions = 0

    for X_i, y_i in zip(X_test, y_test):
        if nn.predict(X_i) != y_i:
            incorrect_predictions += 1

    print(f"Testing error: {incorrect_predictions/len(X_test)}\n")