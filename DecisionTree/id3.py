from DecisionTree.gain import gain
from collections import Counter
from typing import List, Any, Union, Tuple

def id3(x: List[Any], y: List[str], attributes: List[Tuple[str, List[Union[str, int]]]]):
    """
    :param x: the set of training data
    :param y: the set of training labels
    :param attributes: a list of attributes as tuples of (attribute, possible values)
    """

    if len(set(y)) == 1:
        # all labels are the same
        return y[0]

    most_common_label = Counter(y).most_common(1)[0][0]

    if len(attributes) == 0:
        # no more attributes to split on
        return most_common_label

    attributes_list = [a[0] for a in attributes]

    best_gain = 0
    best_attribute = None
    best_attribute_index = None

    for i, a in enumerate(attributes_list):
        g = gain(x, i)
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

        for i in range(len(x)):
            _x = x[best_attribute_index]
            if _x == attribute_val:
                x_v.append(_x)
                y_v.append(y[best_attribute_index])

        if len(x_v) == 0:
            # add leaf node
            branch['node'] = most_common_label
        else:
            branch['node'] = id3(x_v, y_v, new_attributes)

    return root_node