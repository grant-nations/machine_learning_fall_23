#!/bin/bash

# set PYTHONPATH
export PYTHONPATH=$(pwd)

# PROBLEM 2A
echo -e "\nRunning python script for problem 2a...\n"
python3 EnsembleLearning/Adaboost/predict_banking.py

# PROBLEM 2B
echo -e "\nRunning python script for problem 2b...\n"
python3 EnsembleLearning/BaggedTrees/predict_banking.py

# PROBLEM 2C
echo -e "\nRunning python script for problem 2c...\n"

# the first argument is the number of predictors
# the second argument is the number of trees per predictor
# the third argument is the number of samples per predictor
python3 EnsembleLearning/BaggedTrees/squared_error.py 100 500 1000


# PROBLEM 2D
echo -e "\nRunning python script for problem 2d...\n"
python3 EnsembleLearning/RandomForests/predict_banking.py

# PROBLEM 2E
echo -e "\nRunning python script for problem 2e...\n"

# the first argument is the number of predictors
# the second argument is the number of trees per predictor
# the third argument is the number of samples per predictor
python3 EnsembleLearning/RandomForests/squared_error.py 100 500 1000

# PROBLEM 4A
echo -e "\nRunning python script for problem 4a...\n"
python3 LinearRegression/BatchGradDescent/predict_concrete.py

# PROBLEM 4B
echo -e "\nRunning python script for problem 4b...\n"
python3 LinearRegression/StochasticGradDescent/predict_concrete.py

# PROBLEM 4C
echo -e "\nRunning python script for problem 4c...\n"
python3 LinearRegression/AnalyticalSolution/predict_concrete.py

echo -e "\nDone.\n"
