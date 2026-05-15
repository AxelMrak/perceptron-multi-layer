from model.layer import Layer


class Network:
    """Multi-layer perceptron: a stack of Layers."""

    def __init__(self, layers: list[Layer]) -> None:
        if len(layers) < 1:
            raise ValueError("Network must have at least 1 layer")
        self._layers = layers

    @property
    def layers(self) -> list[Layer]:
        return self._layers

    def forward(self, inputs: list[float]) -> list[float]:
        """Pass inputs through all layers, return last layer output."""
        current = inputs
        for i, layer in enumerate(self._layers):
            if i > 0:
                current = [1.0] + current
            current = layer.forward(current)
        return current

    def train_sample(self, inputs: list[float], expected: list[float]) -> list[float]:
        """Forward pass + backpropagation for a single sample."""
        predicted = self.forward(inputs)

        output_errors = [exp - pred for exp, pred in zip(expected, predicted)]

        errors = output_errors
        for layer in reversed(self._layers):
            errors = layer.backward(errors)

        return predicted
