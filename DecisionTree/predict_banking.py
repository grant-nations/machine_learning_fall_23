from DecisionTree.decision_tree import train, predict, preprocess_numerical_attributes
import os
from DecisionTree.gain import majority_error, entropy, gini

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

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
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

for chaos_function, name in [(majority_error, "majority error"), (entropy, "information gain"), (gini, "gini index")]:
    print(f"\n{name} -----------------------")

    test_errors = []
    train_errors = []

    for depth in range(1, 17):
        print(f"  max depth: {depth} -----------")
        print("    training decision tree... ", end="", flush=True)

        tree = train(x_proc, y, attributes_proc, max_depth=depth, chaos_evaluator=chaos_function)

        print(" done.")

        tot_predictions = 0
        incorrect_predictions = 0

        print("    prediction error (train): ", end="", flush=True)

        for _x, _y in zip(x_proc, y):
            if predict(_x, attributes_proc, tree) != _y:
                incorrect_predictions += 1

            tot_predictions += 1

        prediction_error = incorrect_predictions / tot_predictions
        train_errors.append(prediction_error)
        print(f"{round(prediction_error, 3)}")

        print("    prediction error (test): ", end="", flush=True)

        tot_predictions = 0
        incorrect_predictions = 0

        for _x_test, _y_test in zip(x_test_proc, y_test):
            if predict(_x_test, attributes_proc, tree) != _y_test:
                incorrect_predictions += 1

            tot_predictions += 1

        prediction_error = incorrect_predictions / tot_predictions
        test_errors.append(prediction_error)
        print(f"{round(prediction_error, 3)}")

    print(f"\n  average training error: {round(sum(train_errors)/len(train_errors), 3)}")
    print(f"  average testing error: {round(sum(test_errors)/len(test_errors), 3)}")
    print("-----------------------------------")
