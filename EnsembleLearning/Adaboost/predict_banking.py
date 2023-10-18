from Preprocessing.preprocessing import preprocess_numerical_attributes
import os
from EnsembleLearning.Adaboost.adaboost import train, predict
import matplotlib.pyplot as plt
from DecisionStump import decision_stump

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

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
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


max_iters = 500
step = 100

ensemble = None
for num_iters in range(0, max_iters + 1, step):
    if num_iters == 0:
        num_iters = 1
    ensemble = train(x_proc, y, attributes_proc, num_iters)

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


# this ensemble will have max_iters stumps
stumps = [e[1] for e in ensemble]
stump_test_errors = []
stump_train_errors = []

iterations = list(range(0, max_iters + 1, step))
iterations[0] = 1

for i in iterations:
    stump = stumps[i - 1]
    incorrect_predictions = 0
    for _x, _y in zip(x_proc, y):
        if decision_stump.predict(_x, attributes_proc, stump) != _y:
            incorrect_predictions += 1

    prediction_error = incorrect_predictions / len(x_proc)
    stump_train_errors.append(prediction_error)

    incorrect_predictions = 0
    for _x_test, _y_test in zip(x_test_proc, y_test):
        if decision_stump.predict(_x_test, attributes_proc, stump) != _y_test:
            incorrect_predictions += 1

    prediction_error = incorrect_predictions / len(x_test_proc)
    stump_test_errors.append(prediction_error)


# Plotting training and testing errors
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(iterations, training_errors, label='Training Error')
plt.plot(iterations, testing_errors, label='Testing Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Training and Testing Errors')
plt.legend()

# Plotting stump training and testing errors
plt.subplot(1, 2, 2)
plt.plot(iterations, stump_train_errors, label='Stump Training Error')
plt.plot(iterations, stump_test_errors, label='Stump Testing Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Stump Training and Testing Errors')
plt.legend()

# Adjust layout
plt.tight_layout()

# Show the plots
# plt.savefig("adaboost_error.png")
# plt.show()
