import os
import pandas as pd
import numpy as np
from NeuralNetworks.nn import ThreeLayerNN

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data', 'bank_note')
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

x = np.array([1, 1])
y = 1

nn.step(x, y, verbose=True)


# print("------------- PART B ------------- \n")

# input_dim = len(X[0])
# hidden_dims = [5, 10, 25, 50, 100]

# for hidden_dim in hidden_dims:
#     w1 = np.random.randn(input_dim, hidden_dims[0])
#     w2 = np.random.randn(hidden_dims[0], hidden_dims[1])
#     w3 = np.random.randn(hidden_dims[1])  # the "1" dim is implied here
#     b1 = np.random.randn()
#     b2 = np.random.randn()
#     b3 = np.random.randn()
