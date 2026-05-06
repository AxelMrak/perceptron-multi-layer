import random
from config import LEARNING_RATE

class Perceptron:
    """Single-layer perceptron with step activation function."""
    def __init__(self, input_size: int, learning_rate: float) -> None:
        if input_size < 1:
            raise ValueError(f"Input size must be at least 1, got {input_size}")

        self._weights = [random.random() for _ in range(input_size)]
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

    """
        Step (threshold) activation function:
        output = 1 if weighted_sum > 0 else -1
    """
    def activate(self, raw_output: float) -> float:
        return 1.0 if raw_output > 0 else -1.0


    def predict(self, inputs: list[float]) -> float:
        """Full forward pass: compute + activate."""
        return self.activate(self.compute(inputs))


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
            """
    Bias is the first weight (w0) which acts as the threshold in the activation function.
    By convention, we can treat it as the bias term. It allows the decision boundary to  shift away from the origin.
    """
        return self._weights[0]
