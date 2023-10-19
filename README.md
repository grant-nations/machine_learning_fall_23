This is a machine learning library developed by Grant Nations for CS5350/6350 at the University of Utah


# Decision Tree Functions

The `decision_tree.py` module in the `DecisionTree` directory provides functions for training and using a decision tree for classification using the ID3 algorithm. The functions included are:

1. `preprocess_unknown_values(x, attributes)`

    This function preprocesses unknown values in the dataset `x` by replacing them with the most common value for that attribute.

    - `x`: List of data points.
    - `attributes`: List of tuples representing attributes and their possible values.

2. `preprocess_numerical_attributes(x, attributes)`

    This function preprocesses numerical attributes by converting them to binary attributes.

    - `x`: List of data points.
    - `attributes`: List of tuples representing attributes and their possible values.

3. `predict(x, attributes, tree)`

    This function predicts the label for a given data point `x` using the decision tree.

    - `x`: Data point to predict.
    - `attributes`: List of tuples representing attributes and their possible values.
    - `tree`: The decision tree (a dict or a label if it is a leaf node).

4. `train(x, y, attributes, max_depth=None, chaos_evaluator=entropy)`

    This function trains a decision tree using the ID3 algorithm.

    - `x`: List of training data points.
    - `y`: List of training labels.
    - `attributes`: List of tuples representing attributes and their possible values.
    - `max_depth`: Maximum depth of the tree (optional, default is `None`).
    - `chaos_evaluator`: The function to evaluate chaos of a set (e.g. entropy, gini, majority error). Default is `entropy`.

5. `_train(x, y, attributes, curr_depth, max_depth, chaos_evaluator)`

    This is a helper function for training the decision tree. It is not intended for direct use.

    - `x`: List of training data points.
    - `y`: List of training labels.
    - `attributes`: List of tuples representing attributes and their possible values.
    - `curr_depth`: Current depth of the tree.
    - `max_depth`: Maximum depth of the tree (optional, default is `None`).
    - `chaos_evaluator`: The function to evaluate chaos of a set (e.g. entropy, gini, majority error).

## Usage

Examples of how to use these functions can be found in the `predict_banking.py`, `predict_car.py`, and `predict_banking_proc_unknowns.py` files.



---
---
# Homework 2 Execution Guide

The `run.sh` script is designed to automate the execution of Python scripts related to homework 2. Follow the steps below to run the script:

## Instructions

1. **Clone the Repository**

   Clone or download the repository for homework 2.

   ```
   git clone https://github.com/grant-nations/machine_learning_fall_23.git
   ```

2. **Navigate to the Repository**

   Open a terminal or command prompt and navigate to the root directory of the repository.

3. **Give Execute Permissions**

   If necessary, grant execute permissions to the shell script using the following command:
   
   ```
   chmod +x run.sh
   ```

4. **Run the Shell Script**

   Execute the shell script by running the following command:

   ```
   ./run.sh
   ```

5. **Follow the Output**

   The script will start executing and you will see output indicating which Python scripts are being run. The scripts display the training and testing errors from each decision tree.

6. **Completion**

   Once all scripts are done running, you will see "Done." printed to the terminal.

## Additional Notes

- The `run.sh` script will only work from the root project directory, due to `PYTHONPATH` being set to the current working directory.
---