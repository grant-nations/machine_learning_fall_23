This is a machine learning library developed by Grant Nations for CS5350/6350 at the University of Utah

# Homework 1 Execution Guide

This shell script is designed to automate the execution of Python scripts related to problem sets 2b, 3a, and 3b using a Decision Tree model. Follow the steps below to run the script:

## Instructions

1. **Clone the Repository**

   Clone or download the repository for homework 1.

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
   source ./run.sh
   ```

5. **Follow the Output**

   The script will start executing and you will see output indicating which Python scripts are being run. The scripts display the training and testing errors from each decision tree.

6. **Completion**

   Once all scripts are done running, you will see "Done." printed to the terminal.

## Additional Notes

- The `run.sh` script will only work from the root project directory, due to `PYTHONPATH` being set to the current working directory.
---