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

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

        # self.w1 = np.random.randn(input_dim, hidden_dims[0])
        # self.w2 = np.random.randn(hidden_dims[0], hidden_dims[1])
        # self.w3 = np.random.randn(hidden_dims[1]) # the "1" dim is implied here
        # self.b1 = np.random.randn()
        # self.b2 = np.random.randn()
        # self.b3 = np.random.randn()

    def step(self, x: npt.ArrayLike, y: int) -> Tuple[float, npt.ArrayLike]:
        """
        Perform a forward and backward pass for a single data point, updating the weights.

        :param x: x value to pass through model.
        :param y: label for x value

        :return: loss and gradient of loss with respect to weights.
        """

        # ----------------------- FORWARD PASS -----------------------

        layer_1 = x @ self.w1 + self.b1  # (D,) @ (D, H1) -> (H1,)
        layer_1_sig = self._sigmoid(layer_1) # (H1,)

        layer_2 = layer_1_sig @ self.w2 + self.b2  # (H1,) @ (H1, H2) -> (H2,)
        layer_2_sig = self._sigmoid(layer_2) # (H2,)

        layer_3 = layer_2_sig @ self.w3 + self.b3  # (H2,) @ (H2,) -> (1,)
        layer_3_sig = self._sigmoid(layer_3) # (1,)

        prediction = np.rint(layer_3_sig) 

        # ----------------------- BACKWARD PASS -----------------------

        loss = self._loss(prediction, y)

        # _____ layer 3 _____

        dprediction = y - prediction  # (1,)
        dlayer_3_sig = dprediction * layer_3_sig * (1 - layer_3_sig)  # (1,)

        dw3 = dlayer_3_sig * layer_2_sig # (1,) * (H2,) -> (H2,)
        db3 = dlayer_3_sig # (1,)

        dlayer_3 = dlayer_3_sig * self.w3 # (1,) * (H2,) -> (H2,)
        
        # _____ layer 2 _____

        # TODO: This

        # _____ layer 1 _____

        

    def train(self) -> None:
        pass

    def evaluate(self) -> int:
        pass

    @staticmethod
    def _loss(y: int, y_true: int) -> float:
        return (1/2) * (y_true - y) ** 2

    @staticmethod
    def _sigmoid(x: npt.ArrayLike) -> npt.ArrayLike:
        return 1 / (1 + np.exp(-x))
