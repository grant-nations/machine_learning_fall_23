from typing import Dict, List, Any, Callable
from math import log2


def entropy(s: List[str],
            d: List[float]):
    """
    calculate the entropy of this list of labels

    :param s: the set of labels to compute the entropy for
    :param d: the weights of the labels
    """

    # get the number of items in this set
    tot_weight = sum(d)

    if tot_weight == 0:
        return 0

    # label counts
    labels = {}

    for l, w in zip(s, d):
        labels[l] = labels.get(l, 0) + w
        # increase the count of this label

    entropy = 0

    for label_w in list(labels.values()):
        prop = label_w/tot_weight
        if prop > 0:
            entropy -= prop * log2(prop)

    return entropy

def gain(x, y, d, attributes, a_index, chaos_evaluator=entropy):
    """
    :param x: set to partition
    :param y: labels for x
    :param d: weights of examples x
    :param attributes: list of attributes as tuples of (attribute, possible values)
    :param a_index: index of attribute to partition on
    :param chaos_evalutaor: function to evaluate chaos of a set
    """
    set_chaos = chaos_evaluator(y, d)

    partitions = {attr_val: ([], [],) for attr_val in attributes[a_index][1]}

    for _x, _y, _d in zip(x, y, d):
        partitions[_x[a_index]][0].append(_y)
        partitions[_x[a_index]][1].append(_d)

    partition_chaos = 0

    total_weight_sum = sum(d)

    for partition_y, partition_d in list(partitions.values()):
        weight_sum = sum(partition_d)

        proportion_term = weight_sum / total_weight_sum
        partition_chaos += proportion_term * chaos_evaluator(partition_y, partition_d)

    return set_chaos - partition_chaos