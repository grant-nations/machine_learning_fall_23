import random
from collections import Counter
from typing import List, Any, Union, Tuple, Callable, Dict
from DecisionTree.gain import entropy
from DecisionTree import decision_tree


def train(x: List[Any],
          y: List[str],
          attributes: List[Tuple[str, List[Union[str, int]]]],
          num_trees: int,
          feature_subset_size: int):
    """
    Train a random forest using ID3. 

    :param x: the set of training data
    :param y: the set of training labels
    :param attributes: a list of attributes as tuples of (attribute, possible values)
    :param num_trees: the number of trees to use in the forest
    :param feature_subset_size: the number of features to use at each node
    """

    ensemble = []
    indices = list(range(len(x)))

    for _ in range(num_trees):

        # bootstrap samples
        bootstrap_indices = random.choices(indices, k=len(x))
        x_bootstrap = [x[i] for i in bootstrap_indices]
        y_bootstrap = [y[i] for i in bootstrap_indices]

        # train decision tree
        tree = train_rand_tree(x_bootstrap, y_bootstrap, attributes, feature_subset_size)

        # add tree to ensemble
        ensemble.append(tree)

    return ensemble


def predict(x: List[Any],
            attributes: List[Tuple[str, List[Union[str, int]]]],
            ensemble: List[Dict[Any, Any]]):
    """
    :param x: example to predict label of
    :param attributes: list of attributes as tuples of (attribute, possible values)
    :param ensemble: a list of decision trees used to predict the label

    :return: the top-voted label by the ensemble
    """
    votes = [decision_tree.predict(x, attributes, tree) for tree in ensemble]

    labels = {}

    for l in votes:
        labels[l] = labels.get(l, 0) + 1

    return max(labels, key=labels.get)


def train_rand_tree(x: List[Any],
                    y: List[str],
                    attributes: List[Tuple[str, List[Union[str, int]]]],
                    feature_subset_size: int,
                    chaos_evaluator: Callable = entropy):
    """
    Train a random decision tree (one that uses a random subset of features at each node) using ID3.

    :param x: the set of training data
    :param y: the set of training labels
    :param attributes: a list of attributes as tuples of (attribute, possible values)
    :param feature_subset_size: the number of features to use at each node
    :param chaos_evaluator: the function to evaluate chaos of a set (e.g. entropy, gini, majority error)
    """

    if len(set(y)) == 1:
        # all labels are the same
        return y[0]

    most_common_label = Counter(y).most_common(1)[0][0]

    if len(attributes) == 0:
        # no more attributes to split on
        return most_common_label

    attributes_subset = None
    attributes_subset_indices = None
    if len(attributes) <= feature_subset_size:
        # use all attributes
        attributes_subset = attributes
        attributes_subset_indices = list(range(len(attributes)))
    else:
        # use a random subset of features
        attributes_subset_indices = random.sample(list(range(len(attributes))), feature_subset_size)
        # e.g. if attributes_subset_indices = [0, 2, 3], then attributes_subset = [attributes[0], attributes[2], attributes[3]]
        attributes_subset = [attributes[i] for i in attributes_subset_indices]

    attributes_list = [a[0] for a in attributes_subset]

    best_gain = -1
    best_attribute = None
    best_attribute_index = None

    for i, a in enumerate(attributes_list):
        g = gain(x, y, attributes_subset, attributes_subset_indices, i, chaos_evaluator=chaos_evaluator)
        if g >= best_gain:
            best_gain = g
            best_attribute = a
            # e.g. if attributes_subset_indices = [0, 2, 3], then best_attribute_index = 0, 2, or 3
            best_attribute_index = attributes_subset_indices[i]

    # create a node for a new tree
    root_node = {"attribute": best_attribute,
                 "branches": []}

    new_attributes = [a for i, a in enumerate(attributes) if i != best_attribute_index]

    for attribute_val in attributes[best_attribute_index][1]:
        # add a new branch for this attribute value
        branch = {'value': attribute_val, 'node': None}
        root_node['branches'].append(branch)

        x_v = []
        y_v = []

        for i, _x in enumerate(x):
            _x_attr_val = _x[best_attribute_index]
            if _x_attr_val == attribute_val:
                x_v.append(_x[:best_attribute_index] + _x[best_attribute_index + 1:])
                y_v.append(y[i])

        if len(x_v) == 0:
            # add leaf node
            branch['node'] = most_common_label
        else:
            branch['node'] = train_rand_tree(x_v, y_v, new_attributes, feature_subset_size, chaos_evaluator)

    return root_node


def gain(x, y, attributes, orig_attributes_indices, a_index, chaos_evaluator=entropy):
    """
    :param x: set to partition
    :param y: labels for x
    :param attributes: list of attributes as tuples of (attribute, possible values)
    :param orig_attributes_indices: the original indices of the attributes (before feature subset)
    :param a_index: index of attribute to partition on
    :param chaos_evaluator: function to evaluate chaos of a set
    """
    set_chaos = chaos_evaluator(y)

    partitions = {attr_val: [] for attr_val in attributes[a_index][1]}

    for _x, _y in zip(x, y):
        partitions[_x[orig_attributes_indices[a_index]]].append(_y)

    partition_chaos = 0

    for partition_y in list(partitions.values()):
        proportion_term = len(partition_y) / len(y)
        partition_chaos += proportion_term * chaos_evaluator(partition_y)

    return set_chaos - partition_chaos
