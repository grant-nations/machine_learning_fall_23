import math

def bin_me(num_pos: int, num_neg: int):
    total = num_pos + num_neg

    if total == 0:
        return 0
    
    majority_num = max(num_neg, num_pos)

    return 1 - (majority_num / total)

if __name__ == "__main__":

    while True:
        num_pos = int(input("Number of positive class: "))
        num_neg = int(input("Number of negative class: "))
        print("Majority error: ", bin_me(num_pos, num_neg))
        print()