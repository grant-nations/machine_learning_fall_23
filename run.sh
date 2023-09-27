#!/bin/bash

# set PYTHONPATH
export PYTHONPATH=$(pwd)

# run the code for problem 2b
echo -e "\nRunning python script for problem 2b...\n"
python3 DecisionTree/predict_car.py

# run the code for problem 3a
echo -e "\nRunning python script for problem 3a...\n"
python3 DecisionTree/predict_banking.py

# run the code for problem 3b
echo -e "\nRunning python script for problem 3b...\n"
python3 DecisionTree/predict_banking_proc_unknowns.py

echo -e "\nDone.\n"
