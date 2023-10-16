from DecisionStump import decision_stump
from math import exp, log

def train(x, y, attributes, num_iters):
    """
    :param x: the x values to train on
    :param y: the corresponding labels to the training data
    :param attributes: list of attributes as tuples of (attribute, possible values)
    :param num_iters: number of iterations to run the algorithm for
    """

    ensemble = []  # list of alpha, stump tuples

    d = [1/len(x)] * len(x)

    for _ in range(num_iters):

        stump = decision_stump.train(x, y, d, attributes)

        prediction_error = 0

        predictions = [decision_stump.predict(_x, attributes, stump) for _x in x]

        for i, _y in enumerate(y):
            if _y != predictions[i]:
                prediction_error += d[i]

        alpha = 0.5 * log((1 - prediction_error) / prediction_error)

        ensemble.append((alpha, stump,))  # add the weak classifier (stump) to the ensemble

        for i in range(len(d)):
            exp_coeff = exp(-alpha) if y[i] == predictions[i] else exp(alpha)
            d[i] *= exp_coeff

        total_weight = sum(d)

        for i in range(len(d)):
            d[i] /= total_weight  # normalize

    return ensemble


def predict(x, attributes, ensemble):
    """
    :param x: example to predict label of
    :param attributes: list of attributes as tuples of (attribute, possible values)
    :param ensemble: a list of alpha (vote weight), classifier (stump) tuples used to predict the label

    :return: the top-voted label by the ensemble
    """

    votes = [decision_stump.predict(x, attributes, ensemble[i][1]) for i in range(len(ensemble))]

    labels = {}

    for l, e in zip(votes, ensemble):
        alpha = e[0]

        labels[l] = labels.get(l, 0) + alpha

    return max(labels, key=labels.get)
