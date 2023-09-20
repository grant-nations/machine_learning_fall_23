import math


def bin_entropy(num_pos: int, num_neg: int):
    """
    Calculate the entropy of a set that contains two classes.

    :param num_pos: Number of positive class.
    :param num_neg: Number of negative class.

    :return: Entropy of the set.
    """
    total = num_pos + num_neg
    prop_pos = num_pos / total
    prop_neg = num_neg / total
    return -1 * prop_pos * math.log(prop_pos, 2) - prop_neg * math.log(prop_neg, 2)


if __name__ == "__main__":

    while True:
        num_pos = int(input("Number of positive class: "))
        num_neg = int(input("Number of negative class: "))
        print("Entropy: ", bin_entropy(num_pos, num_neg))
        print()
