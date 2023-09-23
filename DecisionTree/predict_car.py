from DecisionTree.decision_tree import train, predict
import os

print("training decision tree... ", end="", flush=True)

label_values = ['unacc', 'acc', 'good', 'vgood']
attributes = [
    ('buying', ['vhigh', 'high', 'med', 'low'],),
    ('maint', ['vhigh', 'high', 'med', 'low'],),
    ('doors', ['2', '3', '4', '5more'],),
    ('persons', ['2', '4', 'more'],),
    ('lug_boot', ['small', 'med', 'big'],),
    ('safety', ['low', 'med', 'high'],),
]

x = []
y = []

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
train_filename = os.path.join(data_dir, "car", "train.csv")
with open(train_filename) as f:
    for line in f:
        values = line.strip().split(',')
        x.append(values[:-1])  # add all independent variables to x
        y.append(values[-1])  # add label to y

tree = train(x, y, attributes)

print(" done.")

tot_predictions = 0
incorrect_predictions = 0

print("prediction error (train): ", end="", flush=True)

for _x, _y in zip(x, y):
    if predict(_x, attributes, tree) != _y:
        incorrect_predictions += 1

    tot_predictions += 1

prediction_error = incorrect_predictions / tot_predictions
print(f"{round(prediction_error, 3)}")

print("prediction error (test): ", end="", flush=True)

tot_predictions = 0
incorrect_predictions = 0
x_test = []
y_test = []

test_filename = os.path.join(data_dir, "car", "test.csv")

with open(test_filename) as f:
    for line in f:
        values = line.strip().split(',')
        x_test.append(values[:-1]) 
        y_test.append(values[-1])

for _x_test, _y_test in zip(x_test, y_test):
    if predict(_x_test, attributes, tree) != _y_test:
        incorrect_predictions += 1

    tot_predictions += 1

prediction_error = incorrect_predictions / tot_predictions
print(f"{round(prediction_error, 3)}")
