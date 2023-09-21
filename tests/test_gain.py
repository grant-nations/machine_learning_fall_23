from DecisionTree.gain import InformationGain

def test_entropy():
    assert InformationGain.entropy(["a", "b"]) == 1
    assert InformationGain.entropy(["a", "a", "a", "a"]) == 0