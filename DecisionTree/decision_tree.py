from DecisionTree.gain import gain, entropy
from collections import Counter
from typing import List, Any, Union, Tuple, Callable
import statistics


def preprocess_unknown_values(x: List[Any],
                              attributes: List[Tuple[str, List[Union[str, int]]]],
                              ):
    """
    Preprocess unknown values in x by replacing them with the most common value for that attribute.

    :param x: the set of data points
    :param attributes: the list of attributes as tuples of (attribute, possible values)
    """

    new_x = x[:][:]
    for i, _ in enumerate(attributes):
        all_attr_vals = [_x[i] for _x in x]
        most_common_val = Counter(all_attr_vals).most_common(1)[0][0]

        for j in range(len(x)):
            if x[j][i] == "unknown":
                new_x[j][i] = most_common_val

    return new_x


def preprocess_numerical_attributes(x: List[Any],
                                    attributes: List[Tuple[str, List[Union[str, int]]]],
                                    ):
    """
    Preprocess numerical attributes by converting them to binary attributes.

    :param x: the set of data points
    :param attributes: the list of attributes as tuples of (attribute, possible values)
    """
    new_attributes = []
    new_x = x[:][:]  # make a copy of x

    for i, (attr_name, attribute_vals) in enumerate(attributes):
        if not isinstance(attribute_vals, list):
            # find median value across all data points
            median_val = statistics.median([float(_x[i]) for _x in x])
            new_attribute = (f"{attr_name} < {median_val}", ["yes", "no"],)
            new_attributes.append(new_attribute)

            # change all x values to be binary instead of numerical
            for j in range(len(new_x)):
                new_x[j][i] = "yes" if float(x[j][i]) < median_val else "no"

        else:
            new_attributes.append((attr_name, attribute_vals,))

    return new_x, new_attributes


def predict(x: List[Any],
            attributes: List[Tuple[str, List[Union[str, int]]]],
            tree: Union[dict, str]):
    """
    Predict the label for a data point x using the decision tree.

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
          attributes: List[Tuple[str, List[Union[str, int]]]],
          max_depth: Union[int, None] = None,
          chaos_evaluator: Callable = entropy):
    """
    train a decision tree using ID3.

    :param x: the set of training data
    :param y: the set of training labels
    :param attributes: a list of attributes as tuples of (attribute, possible values)
    :param max_depth: the maximum depth of the tree
    :param chaos_evaluator: the function to evaluate chaos of a set (e.g. entropy, gini, majority error)
    """
    return _train(x, y, attributes, 0, max_depth, chaos_evaluator)


def _train(x: List[Any],
           y: List[str],
           attributes: List[Tuple[str, List[Union[str, int]]]],
           curr_depth: int,
           max_depth: Union[int, None],
           chaos_evaluator: Callable):
    curr_depth += 1

    if len(set(y)) == 1:
        # all labels are the same
        return y[0]

    most_common_label = Counter(y).most_common(1)[0][0]

    if len(attributes) == 0:
        # no more attributes to split on
        return most_common_label

    if max_depth and curr_depth >= max_depth:
        return most_common_label

    attributes_list = [a[0] for a in attributes]

    best_gain = -1
    best_attribute = None
    best_attribute_index = None

    for i, a in enumerate(attributes_list):
        g = gain(x, y, attributes, i, chaos_evalutaor=chaos_evaluator)
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

        for i, _x in enumerate(x):
            _x_attr_val = _x[best_attribute_index]
            if _x_attr_val == attribute_val:
                x_v.append(_x[:best_attribute_index] + _x[best_attribute_index + 1:])
                y_v.append(y[i])

        if len(x_v) == 0:
            # add leaf node
            branch['node'] = most_common_label
        else:
            branch['node'] = _train(x_v, y_v, new_attributes, curr_depth, max_depth, chaos_evaluator)

    return root_node
