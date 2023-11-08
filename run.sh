#!/bin/bash

# set PYTHONPATH
export PYTHONPATH=$(pwd)

# PROBLEM 2A
echo -e "\nRunning python script for problem 2a...\n"
python3 Perceptron/StandardPerceptron/predict_bank_notes.py

# PROBLEM 2B
echo -e "\nRunning python script for problem 2b...\n"
python3 Perceptron/VotedPerceptron/predict_bank_notes.py

# PROBLEM 2C
echo -e "\nRunning python script for problem 2c...\n"
python3 Perceptron/AveragePerceptron/predict_bank_notes.py

echo -e "\nDone.\n"
