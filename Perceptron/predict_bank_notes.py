import os
import pandas as pd
from Perceptron.perceptron import train, predict

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data', 'bank_note')
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None)

X = train_data.iloc[:, :-1].to_numpy()
y = train_data.iloc[:, -1].replace(0, -1)
y = y.to_numpy()

r = 0.1 # learning rate

w = train(X, y, r, epochs=10)

print(f"Learned weight vector: {w}")

test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None)

X_test = test_data.iloc[:, :-1].to_numpy()
y_test = test_data.iloc[:, -1].replace(0, -1)
y_test = y_test.to_numpy()

incorrect_predictions = 0

for X_i, y_i in zip(X_test, y_test):
    if predict(X_i, w) != y_i:
        incorrect_predictions += 1

print(f"Testing error: {incorrect_predictions/len(X_test)}")