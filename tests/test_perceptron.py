import pytest
from model.perceptron import Perceptron
from model.activation import StepActivation


def _make_perceptron(input_size=3, learning_rate=0.6):
    return Perceptron(input_size=input_size, activation=StepActivation(), learning_rate=learning_rate)


class TestCompute:
    def test_weighted_sum(self):
        p = _make_perceptron()
        p.weights = [0.0, 1.0, 1.0]
        assert p.compute([1.0, 1.0, 1.0]) == 2.0

    def test_bias_only(self):
        p = _make_perceptron()
        p.weights = [0.5, 0.0, 0.0]
        assert p.compute([1.0, 0.0, 0.0]) == 0.5

    def test_input_weight_mismatch_raises(self):
        p = _make_perceptron(input_size=3)
        with pytest.raises(ValueError, match="Expected 3 inputs"):
            p.compute([1.0, 2.0])


class TestPredict:
    def test_positive_result(self):
        p = _make_perceptron()
        p.weights = [0.0, 1.0, 1.0]
        assert p.predict([1.0, 1.0, 1.0]) == 1.0

    def test_negative_result(self):
        p = _make_perceptron()
        p.weights = [0.0, -1.0, -1.0]
        assert p.predict([1.0, 1.0, 1.0]) == -1.0


class TestUpdateWeights:
    def test_positive_error(self):
        p = _make_perceptron(learning_rate=0.6)
        p.weights = [0.0, 0.0, 0.0]
        p.update_weights(error=1.0, inputs=[1.0, 1.0, 1.0])
        assert p.weights == [0.6, 0.6, 0.6]

    def test_negative_error(self):
        p = _make_perceptron(learning_rate=0.5)
        p.weights = [1.0, 1.0, 1.0]
        p.update_weights(error=-1.0, inputs=[1.0, 1.0, 1.0])
        assert p.weights == [0.5, 0.5, 0.5]

    def test_uses_injected_learning_rate(self):
        fast = _make_perceptron(learning_rate=1.0)
        fast.weights = [0.0, 0.0, 0.0]
        fast.update_weights(error=1.0, inputs=[1.0, 1.0, 1.0])
        assert fast.weights == [1.0, 1.0, 1.0]

        slow = _make_perceptron(learning_rate=0.1)
        slow.weights = [0.0, 0.0, 0.0]
        slow.update_weights(error=1.0, inputs=[1.0, 1.0, 1.0])
        assert slow.weights == [0.1, 0.1, 0.1]


class TestWeightsProperty:
    def test_get_returns_copy(self):
        p = _make_perceptron()
        p.weights = [0.1, 0.2, 0.3]
        w = p.weights
        w[0] = 999
        assert p.weights[0] == 0.1

    def test_set_replaces_all_weights(self):
        p = _make_perceptron(input_size=3)
        p.weights = [0.5, -0.2, 0.8]
        assert p.weights == [0.5, -0.2, 0.8]
        assert p.bias == 0.5


class TestInitValidation:
    def test_input_size_zero_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            Perceptron(input_size=0, activation=StepActivation(), learning_rate=0.6)

    def test_input_size_negative_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            Perceptron(input_size=-1, activation=StepActivation(), learning_rate=0.6)

    def test_input_size_one_works(self):
        p = Perceptron(input_size=1, activation=StepActivation(), learning_rate=0.6)
        assert len(p.weights) == 1

    def test_larger_input_size_works(self):
        p = Perceptron(input_size=5, activation=StepActivation(), learning_rate=0.6)
        assert len(p.weights) == 5


class TestBiasProperty:
    def test_bias_is_first_weight(self):
        p = _make_perceptron()
        p.weights = [0.7, 0.1, 0.2]
        assert p.bias == 0.7
