import math

def bin_gini(num_pos: int, num_neg: int):
    total = num_pos + num_neg

    if total == 0:
        return 0

    return 1 - ((num_neg/total)**2 + (num_pos/total) ** 2)

if __name__ == "__main__":

    while True:
        num_pos = int(input("Number of positive class: "))
        num_neg = int(input("Number of negative class: "))
        print("Gini index: ", bin_gini(num_pos, num_neg))
        print()