from abc import ABC, abstractmethod
from typing import Dict, List, Any
from math import log2

class Gain(ABC):
    
    @abstractmethod
    def __call__(s: Dict[str, List[Any]], attribute: str) -> float:
        
        """
        :param s: the set to partition
        :param attribute: the attribute to partition on

        :return: the information gain from this partition
        """

        pass

class InformationGain(Gain):

    def __init__():
        pass

    def __call__(s: Dict[str, List[Any]], attribute: str) -> float:
        pass

    @staticmethod
    def entropy(s: List[str]):
        """
        calculate the entropy of this set of labels

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