from typing import Dict, List, Any, Callable
from math import log2


def entropy(s: List[str]):
    """
    calculate the entropy of this list of labels

    :param s: the set of labels to compute the entropy for
    """

    # get the number of items in this set
    num_labels = len(s)

    if num_labels == 0:
        return 0

    # label counts
    labels = {}

    for l in s:
        labels[l] = labels.get(l, 0) + 1
        # increase the count of this label

    entropy = 0

    for count in list(labels.values()):
        prop = count/num_labels
        entropy -= prop * log2(prop)

    return entropy

def gini(s: List[str]):
    """
    calculate the gini index of this set of labels

    :param s: the set of labels to compute gini index for
    """
    num_labels = len(s)

    if num_labels == 0:
        return 0

    # label counts
    labels = {}

    for l in s:
        labels[l] = labels.get(l, 0) + 1
        # increase the count of this label

    gini_sum_term = 0
    for count in list(labels.values()):
        gini_sum_term += (count/num_labels) ** 2

    return 1 - gini_sum_term

def majority_error(s: List[str]):

    """
    calculate the majority error of this set of labels

    :param s: the set of labels to compute majority error for
    """
   # get the number of items in this set
    num_labels = len(s)

    if num_labels == 0:
        return 0

    # label counts
    labels = {}

    for l in s:
        labels[l] = labels.get(l, 0) + 1
        # increase the count of this label

    max_label_count = 0

    for count in list(labels.values()):
        if count > max_label_count:
            max_label_count = count

    return 1 - (max_label_count / num_labels)

def gain(x, y, attributes, a_index, chaos_evaluator=entropy):
    """
    :param x: set to partition
    :param y: labels for x
    :param attributes: list of attributes as tuples of (attribute, possible values)
    :param a_index: index of attribute to partition on
    :param chaos_evaluator: function to evaluate chaos of a set
    """
    set_chaos = chaos_evaluator(y)

    partitions = {attr_val: [] for attr_val in attributes[a_index][1]}

    for _x, _y in zip(x, y):
        partitions[_x[a_index]].append(_y)

    partition_chaos = 0

    for partition_y in list(partitions.values()):
        proportion_term = len(partition_y) / len(y)
        partition_chaos += proportion_term * chaos_evaluator(partition_y)

    return set_chaos - partition_chaos