import random
from model.activation import ActivationFunction

class Perceptron:
    """Single-layer perceptron."""
    def __init__(self, input_size: int, activation: ActivationFunction, learning_rate: float) -> None:
        if input_size < 1:
            raise ValueError(f"Input size must be at least 1, got {input_size}")

        self._weights = [random.random() for _ in range(input_size)]
        self._activation = activation
        self._learning_rate = learning_rate

    #   Public API

    """ Compute the weighted sum of inputs and weights.
        For inputs [x0, x1, x2] and weights [w0, w1, w2], returns:
        w0*x0 + w1*x1 + w2*x2
    """
    def compute(self, inputs: list[float]) -> float:
        if len(inputs) != len(self._weights):
            raise ValueError(f"Expected {len(self._weights)} inputs, got {len(inputs)}")

        return sum(w * x for w, x in zip(self._weights, inputs))

    def predict(self, inputs: list[float]) -> float:
        """Full forward pass: compute + activate."""
        return self._activation.compute(self.compute(inputs))


    """ Update weights based on error and inputs using the learning rule:
        w_new = w_old + learning_rate * error * input
        For each weight w_i and corresponding input x_i.
    """
    def update_weights(self, error: float, inputs: list[float]) -> None:
        self._weights = [
            w + self._learning_rate * error * x
            for w, x in zip(self._weights, inputs)
        ]

    #   Properties
    @property
    def weights(self) -> list[float]:
        return list(self._weights)

    @weights.setter
    def weights(self, new_weights: list[float]) -> None:
        self._weights = list(new_weights)

    @property
    def bias(self) -> float:
        return self._weights[0]
