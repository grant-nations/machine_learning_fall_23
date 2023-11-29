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

# Adaboost Functions

The `adaboost.py` module in `EnsembleLearning/Adaboost` provides functions to train and use the Adaboost classifier.

1. `train(x, y, attributes, num_iters)`
   - `x`: the x values to train on
   - `y`: the corresponding labels to the training data
   - `attributes`: list of attributes as tuples of (attribute, possible values)
   - `num_iters`: number of iterations to run the algorithm for

   - **Returns:** `ensemble`: a list of tuples containing (alpha, stump) where alpha is the vote weight and stump is a weak classifier.

2. `predict(x, attributes, ensemble)`

   - `x`: example to predict label of
   - `attributes`: list of attributes as tuples of (attribute, possible values)
   - `ensemble`: a list of tuples containing (alpha, classifier) used to predict the label

   - **Returns:** The top-voted label by the ensemble.

## Usage

Examples of how to use these functions can be found in the `predict_banking.py` file.

---

# Bagged Trees Functions

The `bagged_trees.py` module in `EnsembleLearning/BaggedTrees` implements a bagged trees classifier.

1. `train(x: List[Any], y: List[str], attributes: List[Tuple[str, List[Union[str, int]]]], num_trees: int, tree_depth: Union[int, None] = None)`
     - `x`: The feature values to train on.
     - `y`: The corresponding labels for the training data.
     - `attributes`: A list of attributes as tuples of (attribute, possible values).
     - `num_trees`: The number of trees to use in the ensemble.
     - `tree_depth`: The maximum depth of each decision tree (default is `None`).

2. `predict(x: List[Any], attributes: List[Tuple[str, List[Union[str, int]]]], ensemble: List[Dict[Any, Any]]) -> Any`
     - `x`: The example to predict the label of.
     - `attributes`: A list of attributes as tuples of (attribute, possible values).
     - `ensemble`: A list of decision trees used for prediction.

## Usage

Examples of how to use these functions can be found in the `predict_banking.py` file.

---

# Random Forests Functions

The `random_forests.py` module in `EnsembleLearning/RandomForests` implements a random forests classifier.

1. `train(x: List[Any], y: List[str], attributes: List[Tuple[str, List[Union[str, int]]]], num_trees: int, tree_depth: Union[int, None] = None)`
     - `x`: The feature values to train on.
     - `y`: The corresponding labels for the training data.
     - `attributes`: A list of attributes as tuples of (attribute, possible values).
     - `num_trees`: The number of trees to use in the ensemble.
     - `feature_subset_size`: the number of features to use at each node.

2. `predict(x: List[Any], attributes: List[Tuple[str, List[Union[str, int]]]], ensemble: List[Dict[Any, Any]]) -> Any`
     - `x`: The example to predict the label of.
     - `attributes`: A list of attributes as tuples of (attribute, possible values).
     - `ensemble`: A list of decision trees used for prediction.

## Usage

Examples of how to use these functions can be found in the `predict_banking.py` file.

---

# Gradient Descent Functions

The `batch_grad_descent.py` module in `LinearRegression/BatchGradDescent` and `stochastic_grad_descent.py` in `LinearRegression/StochasticGradDescent` implement batch and stochastic gradient descent, respectively. They both implement the following functions:

1. `predict(x: npt.ArrayLike, w: npt.ArrayLike) -> npt.ArrayLike`
   
   Predicts the value of `x` using linear regression.
   
    - `x`: The `x` value.
    - `w`: The weight vector.

2. `loss(X: npt.NDArray, Y: npt.NDArray, w: npt.ArrayLike) -> float`

   Computes the loss using the squared error loss function.
   
     - `X`: The training examples.
     - `Y`: The training labels.
     - `w`: The weight vector.

3. `train(X: npt.NDArray, Y: npt.NDArray, w: npt.ArrayLike, r: float, convergence_tol: float = 1e-6) -> Tuple[npt.ArrayLike, npt.ArrayLike]`

   Performs gradient descent on the squared error loss function.
   
     - `X`: The training examples.
     - `Y`: The training labels.
     - `w`: The weight vector.
     - `r`: The learning rate.
     - `convergence_tol`: The tolerance for convergence (default is `1e-6`).

## Usage

Examples of how to use these functions can be found in the `predict_concrete.py` files.

---

# Average Perceptron Functions

The `average_perception.py` module in `Perceptron/AveragePerceptron` implements the following functions:


1. `train(X: npt.NDArray, y: npt.NDArray, r: float, epochs: int)`

   Trains an average perceptron
   
     - `X`: The training examples.
     - `Y`: The training labels.
     - `r`: The learning rate.
     - `epochs`: The number of epochs to perform

2. `predict(x: npt.ArrayLike, a: npt.ArrayLike)`
   
   Predicts the value of `x` using averaged perceptron.
   
    - `x`: The `x` value.
    - `w`: The averaged weight vector

## Usage

Examples of how to use these functions can be found in the `predict_bank_notes.py` file.

---

# Standard Perceptron Functions

The `perception.py` module in `Perceptron/StandardPerceptron` implements the following functions:


1. `train(X: npt.NDArray, y: npt.NDArray, r: float, epochs: int)`

   Trains a standard perceptron
   
     - `X`: The training examples.
     - `Y`: The training labels.
     - `r`: The learning rate.
     - `epochs`: The number of epochs to perform

2. `predict(x: npt.ArrayLike, w: npt.ArrayLike)`
   
   Predicts the value of `x` using weighted perceptron.
   
    - `x`: The `x` value.
    - `w`: The weight vector

## Usage

Examples of how to use these functions can be found in the `predict_bank_notes.py` file.

---

# Voted Perceptron Functions

The `voted_perception.py` module in `Perceptron/VotedPerceptron` implements the following functions:


1. `train(X: npt.NDArray, y: npt.NDArray, r: float, epochs: int)`

   Trains a voted perceptron
   
     - `X`: The training examples.
     - `Y`: The training labels.
     - `r`: The learning rate.
     - `epochs`: The number of epochs to perform

2. `predict(x: npt.ArrayLike, w_arr: List[npt.ArrayLike], counts: List[int])`
   
   Predicts the value of `x` using voted perceptron.
   
    - `x`: The `x` value.
    - `w_arr`: List of weight vectors
    - `counts`: List of counts for each weight vector

## Usage

Examples of how to use these functions can be found in the `predict_bank_notes.py` file.

---

# Primal SVM Functions

The `primal_svm.py` module in `SVM/PrimalSVM` implements the following functions:


1. `train(X: npt.NDArray, y: npt.NDArray, r: float, c: float, epochs: int, lr_func: Union[Callable, None] = None`

   Trains an SVM classifier in the primal domain
   
     - `X`: The training examples.
     - `Y`: The training labels.
     - `r`: Learning rate.
     - `c`: Tradeoff hyperparameter.
     - `epochs`: Epochs to perform gradient descent.
     - `lr_func`: Learning rate decay function.

   - **Returns:** A tuple containing the optimal weight parameters and loss values at each update

2. `predict(x: npt.ArrayLike, w: npt.ArrayLike)`
   
   Predicts the value of `x` using weights and biases.
   
    - `x`: The `x` value.
    - `w`: Optimal weights, including bias parameter.

## Usage

Examples of how to use these functions can be found in the `predict_bank_notes.py` file.

---

# Dual SVM Functions

The `dual_svm.py` module in `SVM/DualSVM` implements the following functions:


1. `train(X: npt.NDArray, y: npt.NDArray, c: float, kernel: Callable = lambda x, y: x @ y.T)`

   Trains an SVM classifier in the dual domain
   
     - `X`: The training examples.
     - `Y`: The training labels.
     - `c`: Tradeoff hyperparameter.
     - `kernel`: The kernel function to use. Defaults to `lambda x, y: x @ y.T`

   - **Returns:** A tuple containing the optimal weight, bias, and alpha parameters

2. `predict(x: npt.ArrayLike, w: npt.ArrayLike, b: float)`
   
   Predicts the value of `x` using weights and biases.
   
    - `x`: The `x` value.
    - `w`: Optimal weights.
    - `b`: Optimal bias parameter.

3. `predict_with_alpha(x: npt.ArrayLike, X: npt.NDArray, y: npt.ArrayLike, alpha: npt.ArrayLike, b: float, kernel: Callable = lambda x, y: x @ y.T)`

   Predicts the value of `x` using alpha parameters.

   - `x`: The `x` value.
   - `X`: All training examples.
   - `y`: All training labels.
   - `alpha`: Learned alpha parameters.
   - `b`: Bias parameter.
   - `kernel`: Kernel used in training.

## Usage

Examples of how to use these functions can be found in the `predict_bank_notes.py` file.

---


# Homework 5 Execution Guide

The `run.sh` script is designed to automate the execution of Python scripts related to homework 5. Follow the steps below to run the script:

## Instructions

1. **Clone the Repository**

   Clone or download the repository.

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

   The script will start executing and you will see output indicating which homework problem number a script is is being run for.

6. **Completion**

   Once all scripts are done running, you will see "Done." printed to the terminal.

## Additional Notes

- The `run.sh` script will only work from the root project directory, due to `PYTHONPATH` being set to the current working directory.
---