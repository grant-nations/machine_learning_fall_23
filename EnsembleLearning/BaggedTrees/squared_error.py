from Preprocessing.preprocessing import preprocess_numerical_attributes, convert_labels
import os
from EnsembleLearning.BaggedTrees import bagged_trees
from DecisionTree import decision_tree
import random
import statistics
from Utils.spinner import progress_spinner
import sys

num_predictors = None
num_trees = None
num_samples = None

if len(sys.argv) > 3:
    num_predictors = int(sys.argv[1])
    num_trees = int(sys.argv[2])
    num_samples = int(sys.argv[3])
else:
    num_predictors = 100
    num_trees = 500
    num_samples = 1000

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
y_proc = convert_labels(y, old_labels=label_values, new_labels=[1, -1])

x_test = []
y_test = []

test_filename = os.path.join(data_dir, "bank", "test.csv")

with open(test_filename) as f:
    for line in f:
        values = line.strip().split(',')
        x_test.append(values[:-1])
        y_test.append(values[-1])

x_test_proc, _ = preprocess_numerical_attributes(x_test, attributes)
y_test_proc = convert_labels(y_test, old_labels=label_values, new_labels=[1, -1])

indices_arr = list(range(len(x_proc)))


ensembles = []

for i in range(num_predictors):
    progress_spinner("generating decision tree ensembles...", num_predictors, i + 1)
    # sample 1000 examples without replacement from training dataset
    indices = random.sample(indices_arr, num_samples)
    x_samples = [x_proc[j] for j in indices]
    y_samples = [y_proc[j] for j in indices]

    # run bagged trees learning algorithm based on sampled values with 500 trees
    ensemble = bagged_trees.train(x_samples, y_samples, attributes_proc, num_trees)
    ensembles.append(ensemble)

# now we have 100 bagged predictors. Pick the first tree in each run to get 100 single trees.
single_trees = [e[0] for e in ensembles]

x_test_biases = []
x_test_sample_variances = []

print()


# for each test example, compute the predictions of the 100 single trees
for i, (_x, _y) in enumerate(zip(x_test_proc, y_test_proc)):
    progress_spinner("estimating single decision tree squared error...", len(x_test_proc), i + 1)
    predictions = [decision_tree.predict(_x, attributes_proc, t) for t in single_trees]

    # take average prediction across trees --> E_D(h(x*))
    expected_prediction = statistics.mean(predictions)

    # compute bias term
    bias = (_y - expected_prediction) ** 2

    # estimate variance as sample variance of predictions
    sample_variance = statistics.variance(predictions)

    x_test_biases.append(bias)
    x_test_sample_variances.append(sample_variance)

# averages biases and sample variances across all test instances
avg_bias = statistics.mean(x_test_biases)
avg_sample_var = statistics.mean(x_test_sample_variances)

# avg_bias + avg_sample_var = estimate of general squared error for single tree learner
single_tree_se = avg_bias + avg_sample_var

print()
print(f"\tsquared error = {single_tree_se}")
print(f"\tbias = {avg_bias}")
print(f"\tvariance = {avg_sample_var}\n")


# do it all over again for bagged trees

for i, (_x, _y) in enumerate(zip(x_test_proc, y_test_proc)):
    progress_spinner("estimating bagged trees squared error...", len(x_test_proc), i + 1)
    predictions = [bagged_trees.predict(_x, attributes_proc, e) for e in ensembles]

    expected_prediction = statistics.mean(predictions)

    # compute bias term
    bias = (_y - expected_prediction) ** 2

    # estimate variance as sample variance of predictions
    sample_variance = statistics.variance(predictions)

    x_test_biases.append(bias)
    x_test_sample_variances.append(sample_variance)

# averages biases and sample variances across all test instances
avg_bias = statistics.mean(x_test_biases)
avg_sample_var = statistics.mean(x_test_sample_variances)

# avg_bias + avg_sample_var = estimate of general squared error for single tree learner
bagged_trees_se = avg_bias + avg_sample_var

print()
print(f"\tsquared error = {bagged_trees_se}")
print(f"\tbias = {avg_bias}")
print(f"\tvariance = {avg_sample_var}\n")