import random
from model.activation import ActivationFunction


class Layer:
    """A layer of neurons sharing the same activation function."""

    def __init__(
        self,
        num_neurons: int,
        num_inputs: int,
        activation: ActivationFunction,
        learning_rate: float,
    ) -> None:
        self._num_inputs = num_inputs
        self._activation = activation
        self._learning_rate = learning_rate

        # Each neuron has its own weight vector: weights[i][j] = weight j of neuron i
        self._weights = [
            [random.random() for _ in range(num_inputs)]
            for _ in range(num_neurons)
        ]

        # Cache for backpropagation
        self._last_input: list[float] = []
        self._last_raw: list[float] = []
        self._last_output: list[float] = []

    @property
    def num_neurons(self) -> int:
        return len(self._weights)

    @property
    def num_inputs(self) -> int:
        return self._num_inputs

    @property
    def weights(self) -> list[list[float]]:
        return [list(row) for row in self._weights]

    def forward(self, inputs: list[float]) -> list[float]:
        """Compute outputs for all neurons. Caches values for backpropagation."""
        if len(inputs) != self._num_inputs:
            raise ValueError(f"Expected {self._num_inputs} inputs, got {len(inputs)}")

        self._last_input = inputs
        self._last_raw = []
        self._last_output = []

        for neuron_weights in self._weights:
            raw = sum(w * x for w, x in zip(neuron_weights, inputs))
            self._last_raw.append(raw)
            self._last_output.append(self._activation.compute(raw))

        return list(self._last_output)

    def backward(self, errors: list[float]) -> list[float]:
        """Backpropagate errors, update weights, return errors for previous layer."""
        if len(errors) != len(self._weights):
            raise ValueError(f"Expected {len(self._weights)} errors, got {len(errors)}")

        deltas = [
            errors[i] * self._activation.derivative(self._last_raw[i])
            for i in range(len(errors))
        ]

        for i, neuron_weights in enumerate(self._weights):
            for j in range(len(neuron_weights)):
                neuron_weights[j] += self._learning_rate * deltas[i] * self._last_input[j]

        prev_errors = [
            sum(self._weights[i][j] * deltas[i] for i in range(len(self._weights)))
            for j in range(1, self._num_inputs)
        ]

        return prev_errors
