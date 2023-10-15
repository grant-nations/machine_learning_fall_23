from collections import Counter
from typing import List, Any, Union, Tuple
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