from DecisionStump.gain import gain, entropy
from typing import List, Any, Union, Tuple, Callable

def predict(x: List[Any],
            attributes: List[Tuple[str, List[Union[str, int]]]],
            tree: Union[dict, str]):
    """
    Predict the label for a data point x using the decision stump.

    :param x: the data point to predict
    :param attributes: the list of attributes as tuples of (attribute, possible values)
    :param tree: the decision tree (a dict or a label if it is a leaf node)
    """
    if not isinstance(tree, dict):
        return tree  # this is a label

    attribute = tree["attribute"]
    for i, a in enumerate(attributes):
        if attribute == a[0]:
            x_value = x[i]
            for branch in tree["branches"]:
                if branch["value"] == x_value:
                    return predict(x, attributes, branch["node"])


def train(x: List[Any],
          y: List[str],
          d: List[float],
          attributes: List[Tuple[str, List[Union[str, int]]]],
          chaos_evaluator: Callable = entropy):
    """
    train a decision stump using ID3.

    :param x: the set of training data
    :param y: the set of training labels
    :param d: the weights of each training example
    :param attributes: a list of attributes as tuples of (attribute, possible values)
    :param chaos_evaluator: the function to evaluate chaos of a set (e.g. entropy, gini, majority error)
    """
    return _train(x, y, d, attributes, 0, chaos_evaluator)


def _train(x: List[Any],
           y: List[str],
           d: List[float],
           attributes: List[Tuple[str, List[Union[str, int]]]],
           curr_depth: int,
           chaos_evaluator: Callable):
    max_depth = 2 # this is hardcoded for stump

    curr_depth += 1

    if len(set(y)) == 1:
        # all labels are the same
        return y[0]

    label_scores = {}
    for label, weight in zip(y, d):
        label_scores[label] = label_scores.get(label, 0) + weight

    highest_scoring_label = max(label_scores, key=label_scores.get)

    if len(attributes) == 0:
        # no more attributes to split on
        return highest_scoring_label

    if curr_depth >= max_depth:
        return highest_scoring_label

    attributes_list = [a[0] for a in attributes]

    best_gain = -1
    best_attribute = None
    best_attribute_index = None

    for i, a in enumerate(attributes_list):
        g = gain(x, y, d, attributes, i, chaos_evaluator=chaos_evaluator)
        if g >= best_gain:
            best_gain = g
            best_attribute = a
            best_attribute_index = i

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
        d_v = []

        for i, _x in enumerate(x):
            _x_attr_val = _x[best_attribute_index]
            if _x_attr_val == attribute_val:
                x_v.append(_x[:best_attribute_index] + _x[best_attribute_index + 1:])
                y_v.append(y[i])
                d_v.append(d[i])

        if len(x_v) == 0:
            # add leaf node
            branch['node'] = highest_scoring_label
        else:
            branch['node'] = _train(x_v, y_v, d_v, new_attributes, curr_depth, chaos_evaluator)

    return root_node
