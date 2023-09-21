from DecisionTree.gain import InformationGain


def test_entropy():
    assert InformationGain.entropy(["a", "b"]) == 1
    assert InformationGain.entropy(["a", "a", "a", "a"]) == 0


def test_information_gain():
    s = [
        {
            "attributes": {
                "x1": {
                    0: True,
                    1: False
                },
                "x2": {
                    0: True,
                    1: False
                },
                "x3": {
                    0: False,
                    1: True
                }
            },
            "label": 0
        },
        {
            "attributes": {
                "x1": {
                    0: True,
                    1: False
                },
                "x2": {
                    0: True,
                    1: False
                },
                "x3": {
                    0: True,
                    1: False
                }
            },
            "label": 1
        }
    ]

    assert InformationGain.entropy([d['label'] for d in s]) == 1
    assert InformationGain.evaluate(s, "x1") == 0
    assert InformationGain.evaluate(s, "x2") == 0
    assert InformationGain.evaluate(s, "x3") == 1
