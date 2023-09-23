from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable
from math import log2


# TODO: Get rid of the "Gain" class, make it all functions.
# (Make the gain function a passable function to e.g. information_gain)

def _gain(s: List[Dict[str, Any]], attribute: str, chaos_operator: Callable) -> float:
    pass

def gain():
    return NotImplemented

class Gain(ABC):

    @abstractmethod
    def __call__(s: List[Dict[str, Any]], attribute: str, chaos_operator: Callable) -> float:
        """
        calculate the gain from a partition of this set

        :param s: the set to partition as a list of dictionaries of the form
        {
            "attributes": {
                "attr_a": {
                    "val_1": True,
                    "val_2": False,
                    "val_3": False,
                    "val_4": False
                    },
                "attr_b": {
                    "val_1": False,
                    "val_2": True,
                    "val_3": False
                },
                ...
            },
            "label": 1
        }
        :param attribute: the attribute to partition on

        :return: the gain from this partition
        """
        pass


class InformationGain(Gain):

    def __init__():
        pass

    @staticmethod
    def evaluate(s: List[Dict[str, Any]], attribute: str) -> float:
        labels = [d["label"] for d in s]
        s_entropy = InformationGain.entropy(labels)

        partitions = {}

        for d in s:
            attr = d["attributes"][attribute]
            for key, val in attr.items():
                if val:
                    # check if this partition exists
                    if partitions.get(key) is None:
                        partitions[key] = []

                    # add this item to the correct partition set
                    partitions[key].append(d)
                    break

        partition_entropy = 0
        for part_set in list(partitions.values()):
            part_labels = [d["label"] for d in part_set]
            proportion_term = len(part_labels)/len(labels)
            partition_entropy += proportion_term * InformationGain.entropy(part_labels)

        return s_entropy - partition_entropy

    @staticmethod
    def entropy(s: List[str]):
        """
        calculate the entropy of this list of labels

        :param s: the set of labels to compute the entropy for
        """

        # get the number of items in this set
        num_labels = len(s)

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
