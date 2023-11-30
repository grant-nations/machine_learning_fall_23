import numpy.typing as npt
import numpy as np
from typing import Tuple


class ThreeLayerNN:

    def __init__(self,
                 w1: npt.ArrayLike,
                 w2: npt.ArrayLike,
                 w3: npt.ArrayLike,
                 b1: float,
                 b2: float,
                 b3: float) -> None:
        """
        Initialize a three layer neural network.

        :param input_dim: The dimension of the input data.
        :param w1: The weights of the first layer.
        :param w2: The weights of the second layer.
        :param w3: The weights of the third layer.
        :param b1: The bias of the first layer.
        :param b2: The bias of the second layer.
        :param b3: The bias of the third layer.
        """

        self.w1 = w1  # (D, H1)
        self.w2 = w2  # (H1, H2)
        self.w3 = w3  # (H2,)
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def step(self, x: npt.ArrayLike, y: int, verbose: bool = False) -> Tuple[float, npt.ArrayLike]:
        """
        Perform a forward and backward pass for a single data point, updating the weights.

        :param x: x value to pass through model.
        :param y: label for x value

        :return: loss and gradient of loss with respect to weights.
        """

        # ----------------------- FORWARD PASS -----------------------

        if verbose:
            print("------------- FORWARD PASS ------------- \n")

        layer_1 = x @ self.w1 + self.b1  # (D,) @ (D, H1) -> (H1,)
        if verbose:
            print(f"layer_1: {layer_1}")

        layer_1_sig = self._sigmoid(layer_1)  # (H1,)
        if verbose:
            print(f"layer_1_sig: {layer_1_sig}")

        layer_2 = layer_1_sig @ self.w2 + self.b2  # (H1,) @ (H1, H2) -> (H2,)
        if verbose:
            print(f"layer_2: {layer_2}")

        layer_2_sig = self._sigmoid(layer_2)  # (H2,)
        if verbose:
            print(f"layer_2_sig: {layer_2_sig}")

        layer_3 = layer_2_sig @ self.w3 + self.b3  # (H2,) @ (H2,) -> (1,)
        if verbose:
            print(f"layer_3: {layer_3}")

        # ----------------------- BACKWARD PASS -----------------------

        if verbose:
            print("\n------------- BACKWARD PASS ------------- \n")

        loss = self._loss(layer_3, y)
        if verbose:
            print(f"loss: {loss}")

        # _____ layer 3 _____

        # dlayer_3 = dprediction * self.w3  # (1,) * (H2,) -> (H2,)
        dlayer_3 = layer_3 - y # (1,)
        if verbose:
            print(f"dlayer_3: {dlayer_3}")

        # dw3 = dprediction * layer_2_sig  # (1,) * (H2,) -> (H2,)
        dw3 = layer_2_sig * dlayer_3 # (H2,) * (1,) -> (H2,)
        if verbose:
            print(f"dw3: {dw3}")

        # db3 = dprediction  # (1,)
        db3 = dlayer_3 # (1,)
        if verbose:
            print(f"db3: {db3}")

        # _____ layer 2 _____

        # dlayer_2_sig = dlayer_3 * layer_2_sig * (1 - layer_2_sig)  # (H2,) * (H2,) * (H2,) -> (H2,)
        dlayer_2_sig = dlayer_3 * self.w3 # (1,) * (H2,) -> (H2,)
        if verbose:
            print(f"dlayer_2_sig: {dlayer_2_sig}")

        # dlayer_2 = dlayer_2_sig * self.w2.T  # (H2,) * (H2, H1) -> 
        dlayer_2 = layer_2_sig * (1 - layer_2_sig) * dlayer_2_sig # (H2,) * (H2,) * (H2,) -> (H2,)
        if verbose:
            print(f"dlayer_2: {dlayer_2}")

        dw2 = np.outer(layer_1_sig, dlayer_2)  # (H1,) X (H2,) -> (H1, H2)
        if verbose:
            print(f"dw2: {dw2}")

        db2 = dlayer_2
        if verbose:
            print(f"db2: {db2}")

        # _____ layer 1 _____

        # dlayer_1_sig = dlayer_2 * layer_1_sig * (1 - layer_1_sig)  # (H1,) * (H1,) * (H1,) -> (H1,)
        dlayer_1_sig = dlayer_2 @ self.w2.T # (H2,) @ (H2, H1) -> (H1,)
        if verbose:
            print(f"dlayer_1_sig: {dlayer_1_sig}")

        # dlayer_1 = dlayer_1_sig @ self.w1.T  # (H1,) @ (H1, D) -> (D,)
        dlayer_1 = layer_1_sig * (1 - layer_1_sig) * dlayer_1_sig # (H1,) * (H1,) * (H1,) -> (H1,)
        if verbose:
            print(f"dlayer_1: {dlayer_1_sig}")

        dw1 = np.outer(x, dlayer_1)  # (D,) X (H1,) -> (D, H1)
        if verbose:
            print(f"dw1: {dw1}")

        db1 = dlayer_1
        if verbose:
            print(f"db1: {db1}")

        return loss, (dw1, db1, dw2, db2, dw3, db3)

    def train(self,
              X: npt.NDArray,
              y: npt.ArrayLike,
              epochs: int,
              gamma_0: float,
              d: float) -> None:

        loss_history = []

        gamma = gamma_0

        indices = np.arange(len(X))
        for epoch in range(epochs):
            np.random.shuffle(indices)

            gamma = gamma / (1 + (gamma / d) * epoch)

            for i in indices:

                _x = X[i]
                _y = y[i]

                loss, (dw1, db1, dw2, db2, dw3, db3) = self.step(_x, _y)
                self._update_weights(dw1, dw2, dw3, db1, db2, db3, gamma)

                loss_history.append(loss)

        return loss_history

    def predict(self, x: npt.ArrayLike) -> int:
        layer_1 = x @ self.w1 + self.b1  # (D,) @ (D, H1) -> (H1,)
        layer_1_sig = self._sigmoid(layer_1)  # (H1,)

        layer_2 = layer_1_sig @ self.w2 + self.b2  # (H1,) @ (H1, H2) -> (H2,)
        layer_2_sig = self._sigmoid(layer_2)  # (H2,)

        layer_3 = layer_2_sig @ self.w3 + self.b3  # (H2,) @ (H2,) -> (1,)

        if layer_3 >= 0.5:
            return 1
        else:
            return 0

    def _update_weights(self, dw1, dw2, dw3, db1, db2, db3, gamma):
        self.w1 -= dw1 * gamma
        self.w2 -= dw2 * gamma
        self.w3 -= dw3 * gamma
        self.b1 -= db1 * gamma
        self.b2 -= db2 * gamma
        self.b3 -= db3 * gamma

    @staticmethod
    def _loss(y: int, y_true: int) -> float:
        return (1/2) * (y_true - y) ** 2

    @staticmethod
    def _sigmoid(x: npt.ArrayLike) -> npt.ArrayLike:
        return 1 / (1 + np.exp(-x))
