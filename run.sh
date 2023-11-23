#!/bin/bash

# set PYTHONPATH
export PYTHONPATH=$(pwd)

# PROBLEM 2
echo -e "\nRunning python script for problem 2...\n"
python3 SVM/PrimalSVM/predict_bank_notes.py

# PROBLEM 3
echo -e "\nRunning python script for problem 3...\n"
python3 SVM/DualSVM/predict_bank_notes.py

echo -e "\nDone.\n"
