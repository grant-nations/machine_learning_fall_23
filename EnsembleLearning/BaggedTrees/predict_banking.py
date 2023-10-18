from Preprocessing.preprocessing import preprocess_numerical_attributes
import os
from EnsembleLearning.BaggedTrees.bagged_trees import train, predict
from DecisionTree import decision_tree
import matplotlib.pyplot as plt

label_values = ['yes', 'no']
attributes = [
    ("age", "numeric"),
    ("job",
     ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar",
      "self-employed", "retired", "technician", "services"]),
    ("marital", ["married", "divorced", "single"]),
    ("education", ["unknown", "secondary", "primary", "tertiary"]),
    ("default", ["yes", "no"]),
    ("balance", "numeric"),
    ("housing", ["yes", "no"]),
    ("loan", ["yes", "no"]),
    ("contact", ["unknown", "telephone", "cellular"]),
    ("day", "numeric"),
    ("month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]),
    ("duration", "numeric"),
    ("campaign", "numeric"),
    ("pdays", "numeric"),
    ("previous", "numeric"),
    ("poutcome", ["unknown", "other", "failure", "success"])]

x = []
y = []

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Data")
train_filename = os.path.join(data_dir, "bank", "train.csv")
with open(train_filename) as f:
    for line in f:
        values = line.strip().split(',')
        x.append(values[:-1])  # add all independent variables to x
        y.append(values[-1])  # add label to y

x_proc, attributes_proc = preprocess_numerical_attributes(x, attributes)

x_test = []
y_test = []

test_filename = os.path.join(data_dir, "bank", "test.csv")

with open(test_filename) as f:
    for line in f:
        values = line.strip().split(',')
        x_test.append(values[:-1])
        y_test.append(values[-1])

x_test_proc, _ = preprocess_numerical_attributes(x_test, attributes)

training_errors = []
testing_errors = []


max_trees = 500
step = 100

ensemble = None
for num_trees in range(0, max_trees + 1, step):
    if num_trees == 0:
        num_trees = 1
    ensemble = train(x_proc, y, attributes_proc, num_trees)

    tot_predictions = 0
    incorrect_predictions = 0

    print("Prediction error (train): ", end="", flush=True)

    for _x, _y in zip(x_proc, y):
        if predict(_x, attributes_proc, ensemble) != _y:
            incorrect_predictions += 1

        tot_predictions += 1

    prediction_error = incorrect_predictions / tot_predictions
    print(f"{round(prediction_error, 3)}")

    training_errors.append(prediction_error)

    print("Prediction error (test): ", end="", flush=True)

    tot_predictions = 0
    incorrect_predictions = 0

    for _x_test, _y_test in zip(x_test_proc, y_test):
        if predict(_x_test, attributes_proc, ensemble) != _y_test:
            incorrect_predictions += 1

        tot_predictions += 1

    prediction_error = incorrect_predictions / tot_predictions
    print(f"{round(prediction_error, 3)}")

    testing_errors.append(prediction_error)

    # Now, let's plot the results
num_trees_range = list(range(0, max_trees + 1, step))
num_trees_range[0] = 1

plt.plot(num_trees_range, training_errors, label='Training Errors')
plt.plot(num_trees_range, testing_errors, label='Testing Errors')

plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.title('Training and Testing Errors vs Number of Trees')
plt.legend()
plt.savefig("bagged_trees_error.png")
plt.show()

single_tree = decision_tree.train(x_proc, y, attributes_proc)

tot_predictions = 0
incorrect_predictions = 0

print("Single tree prediction error (train): ", end="", flush=True)

for _x, _y in zip(x_proc, y):
    if decision_tree.predict(_x, attributes_proc, single_tree) != _y:
        incorrect_predictions += 1

    tot_predictions += 1

prediction_error = incorrect_predictions / tot_predictions
print(f"{round(prediction_error, 3)}")

print("Single tree prediction error (test): ", end="", flush=True)

tot_predictions = 0
incorrect_predictions = 0

for _x_test, _y_test in zip(x_test_proc, y_test):
    if decision_tree.predict(_x_test, attributes_proc, single_tree) != _y_test:
        incorrect_predictions += 1

    tot_predictions += 1

prediction_error = incorrect_predictions / tot_predictions
print(f"{round(prediction_error, 3)}")