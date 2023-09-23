from DecisionTree.gain import InformationGain


def test_entropy():
    assert InformationGain.entropy(["a", "b"]) == 1
    assert InformationGain.entropy(["a", "a", "a", "a"]) == 0


def test_information_gain():
    assert False
