import math
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """Abstract base for activation functions."""

    @abstractmethod
    def compute(self, x: float) -> float:
        """Apply activation to a weighted sum."""
        ...

    @abstractmethod
    def derivative(self, x: float) -> float:
        """Return the derivative at x."""
        ...


class StepActivation(ActivationFunction):
    """Step activation: returns 1.0 if x > 0, else -1.0."""

    def compute(self, x: float) -> float:
        return 1.0 if x > 0 else -1.0

    def derivative(self, x: float) -> float:
        return 0.0


class TanhActivation(ActivationFunction):
    """Hyperbolic tangent: output range (-1, 1). Suitable for targets in [-1, 1]."""

    def compute(self, x: float) -> float:
        if x < -700:
            return -1.0
        if x > 700:
            return 1.0
        return math.tanh(x)

    def derivative(self, x: float) -> float:
        tx = self.compute(x)
        return 1.0 - tx * tx


class SigmoidActivation(ActivationFunction):
    """Sigmoid activation: 1 / (1 + e^(-x)). Output range: (0, 1)."""

    def compute(self, x: float) -> float:
        if x < -700:
            return 0.0
        if x > 700:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def derivative(self, x: float) -> float:
        sx = self.compute(x)
        return sx * (1.0 - sx)
