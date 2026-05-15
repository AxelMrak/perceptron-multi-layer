from model.perceptron import Perceptron
from model.dataset import Dataset


class TrainerController:
    """
    Orchestrates the training loop.
    - Iterate
    - call model
    - track state and expose it for the view to display.
    """

    def __init__(
        self,
        perceptron: Perceptron,
        dataset: Dataset,
        max_epochs: int = 100,
    ) -> None:
        """
        Args:
            perceptron: The model to train.
            dataset: Training samples (e.g., AND truth table).
            max_epochs: Safety limit to prevent infinite loops.
        """
        self._perceptron = perceptron
        self._dataset = dataset
        self._max_epochs = max_epochs

        # State tracking for UI
        self._current_epoch = 0
        self._current_sample_input = [0.0, 0.0, 0.0]
        self._current_sample_expected = 0.0
        self._current_sample_predicted = 0.0
        self._current_error = 0.0
        self._converged = False

    # PROPS
    @property
    def current_epoch(self) -> int:
        """Which epoch we're on (1-based)."""
        return self._current_epoch

    @property
    def current_sample_input(self) -> list[float]:
        """Inputs of the current/last training sample."""
        return self._current_sample_input

    @property
    def current_sample_expected(self) -> float:
        """Expected output of the current/last sample."""
        return self._current_sample_expected

    @property
    def current_sample_predicted(self) -> float:
        """Predicted output for the current/last sample."""
        return self._current_sample_predicted

    @property
    def current_error(self) -> float:
        """Error of the last processed sample."""
        return self._current_error

    @property
    def converged(self) -> bool:
        """True if the perceptron correctly classified all samples."""
        return self._converged

    @property
    def perceptron(self) -> Perceptron:
        """Expose perceptron so View can read its weights."""
        return self._perceptron

    #  TRAINING METHODS
    def train_epoch(self) -> bool:
        """
        Run one complete pass through all training samples.

        For each sample:
        1. Predict output using current weights
        2. Calculate error = expected - predicted
        3. If error != 0, update weights via delta rule

        Returns:
            True if ALL samples were correctly classified (converged).
        """
        self._converged = True

        for sample in self._dataset:
            self._current_sample_input = sample.inputs
            self._current_sample_expected = sample.expected
            self._current_sample_predicted = self._perceptron.predict(sample.inputs)
            self._current_error = sample.expected - self._current_sample_predicted

            if self._current_error != 0.0:
                self._perceptron.update_weights(self._current_error, sample.inputs)
                self._converged = False

        # Increment epoch count after processing all samples
        self._current_epoch += 1
        return self._converged

    def train_full(self) -> bool:
        """
        Run epochs until convergence or max_epochs reached.

        Returns:
            True if converged within max_epochs, False otherwise.
        """
        for _ in range(self._max_epochs):
            if self.train_epoch():
                return True
        return False

    def test_prediction(self, input1_str: str, input2_str: str) -> str:
        """
        Make a prediction from user input (for the UI "Test" button).

        Args:
            input1_str: First input as string (from tkinter Entry).
            input2_str: Second input as string.

        Returns:
            A human-readable result, e.g., "Result: 1.0 (AND is True)".
        """
        valid = {"1", "-1"}
        if input1_str not in valid or input2_str not in valid:
            raise ValueError("Only values 1 or -1 are accepted.")

        result = self._perceptron.predict(
            [1.0, int(input1_str), int(input2_str)]
        )
        return f"Result: {result} (AND is {'True' if result == 1.0 else 'False'})"

    #  MULTI-LAYER TRAINING

    def train_backprop(
        self,
        network,
        dataset: Dataset,
        max_epochs: int = 1000,
        target_error: float = 0.01,
    ) -> tuple[bool, int]:
        """
        Train a multi-layer network using backpropagation.

        Runs epochs until MSE < target_error or max_epochs is reached.

        Returns:
            (converged, epochs): whether it converged and how many epochs it took.
        """
        for epoch in range(1, max_epochs + 1):
            total_error = 0.0

            for sample in dataset:
                predicted = network.train_sample(sample.inputs, [sample.expected])
                error = sample.expected - predicted[0]
                total_error += error * error

            mse = total_error / len(dataset)
            self._current_epoch = epoch
            self._current_error = mse

            if mse < target_error:
                self._converged = True
                return True, epoch

        self._converged = False
        return False, max_epochs

    def predict_network(self, network, inputs: list[float]) -> float:
        """Predict using a multi-layer network (single output neuron)."""
        outputs = network.forward(inputs)
        return outputs[0]
