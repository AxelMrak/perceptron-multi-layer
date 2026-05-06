import random
from config import LEARNING_RATE

class Perceptron:
    """Single-layer perceptron with step activation function."""
    def __init__(self, weights: list[float] | None = None) -> None:
        if weights is not None:
            self._validate_weights(weights)
            self._weights = list(weights)
        else:
            self._weights = [random.random() for _ in range(3)]

    #   Public API
    def compute(self, inputs: list[float]) -> float:
        """Weighted sum: w0*x0 + w1*x1 + w2*x2."""
        return sum(w * x for w, x in zip(self._weights, inputs))
    def activate(self, raw_output: float) -> float:
        """Step activation: 1 if > 0, else -1."""
        return 1.0 if raw_output > 0 else -1.0
    def predict(self, inputs: list[float]) -> float:
        """Full forward pass: compute + activate."""
        return self.activate(self.compute(inputs))
    def update_weights(self, error: float, inputs: list[float],
                       learning_rate: float = LEARNING_RATE) -> None:
        """Delta rule: w_new = w_old + lr * error * input."""
        self._weights = [
            w + learning_rate * error * x
            for w, x in zip(self._weights, inputs)
        ]

    #   Properties
    @property
    def weights(self) -> list[float]:
        return list(self._weights)
    @weights.setter
    def weights(self, new_weights: list[float]) -> None:
        self._validate_weights(new_weights)
        self._weights = list(new_weights)

    @property
    def bias(self) -> float:
        return self._weights[0]
    @property
    def w1(self) -> float:
        return self._weights[1]
    @property
    def w2(self) -> float:
        return self._weights[2]

    #   Internal
    @staticmethod
    def _validate_weights(weights: list[float]) -> None:
        if len(weights) != 3:
            raise ValueError(f"Expected 3 weights, got {len(weights)}")
