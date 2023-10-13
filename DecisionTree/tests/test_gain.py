from DecisionTree.gain import entropy, majority_error


def test_entropy():
    assert entropy(["a", "b"]) == 1
    assert entropy(["a", "a", "a", "a"]) == 0

def test_majority_error():
    assert majority_error(["a", "a", "a", "a", "b", "b", "c", "d"]) == 0.5
    assert majority_error(["a", "a", "a", "a"]) == 0