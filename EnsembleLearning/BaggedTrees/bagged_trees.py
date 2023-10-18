# for T iterations:
#   bootstrap samples to train decision tree with
#   train decision tree
#   save tree in ensemble
# return ensemble of trees as classifier

import random
from DecisionTree import decision_tree
from typing import Union, List, Any, Tuple, Dict


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


def train(x: List[Any],
          y: List[str],
          attributes: List[Tuple[str, List[Union[str, int]]]],
          num_trees,
          tree_depth: Union[int, None] = None):
    """
    :param x: the x values to train on
    :param y: the corresponding labels to the training data
    :param attributes: list of attributes as tuples of (attribute, possible values)
    :param num_trees: number of trees to use in ensemble
    """
    ensemble = []
    indices = list(range(len(x)))

    for _ in range(num_trees):

        # bootstrap samples
        bootstrap_indices = random.choices(indices, k=len(x))
        x_bootstrap = [x[i] for i in bootstrap_indices]
        y_bootstrap = [y[i] for i in bootstrap_indices]

        # train decision tree
        tree = decision_tree.train(
            x_bootstrap, y_bootstrap, attributes, max_depth=tree_depth)

        # add tree to ensemble
        ensemble.append(tree)

    return ensemble
