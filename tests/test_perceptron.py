import pytest
from model.perceptron import Perceptron

class TestCompute:
    def test_weighted_sum(self):
        p = Perceptron()
        p.weights = [0.0, 1.0, 1.0]
        assert p.compute([1.0, 1.0, 1.0]) == 2.0
    def test_bias_only(self):
        p = Perceptron()
        p.weights = [0.5, 0.0, 0.0]
        assert p.compute([1.0, 0.0, 0.0]) == 0.5

class TestActivate:
    def test_positive(self):
        assert Perceptron().activate(0.5) == 1.0
    def test_zero(self):
        assert Perceptron().activate(0.0) == -1.0
    def test_negative(self):
        assert Perceptron().activate(-0.5) == -1.0

class TestPredict:
    def test_positive_result(self):
        p = Perceptron()
        p.weights = [0.0, 1.0, 1.0]
        assert p.predict([1.0, 1.0, 1.0]) == 1.0
    def test_negative_result(self):
        p = Perceptron()
        p.weights = [0.0, -1.0, -1.0]
        assert p.predict([1.0, 1.0, 1.0]) == -1.0

class TestUpdateWeights:
    def test_positive_error(self):
        p = Perceptron()
        p.weights = [0.0, 0.0, 0.0]
        p.update_weights(error=1.0, inputs=[1.0, 1.0, 1.0], learning_rate=0.6)
        assert p.weights == [0.6, 0.6, 0.6]
    def test_negative_error(self):
        p = Perceptron()
        p.weights = [1.0, 1.0, 1.0]
        p.update_weights(error=-1.0, inputs=[1.0, 1.0, 1.0], learning_rate=0.5)
        assert p.weights == [0.5, 0.5, 0.5]

class TestWeightsProperty:
    def test_get_returns_copy(self):
        p = Perceptron()
        p.weights = [0.1, 0.2, 0.3]
        w = p.weights
        w[0] = 999
        assert p.weights[0] == 0.1  # Internal state unchanged
    def test_set_validates_length(self):
        p = Perceptron()
        with pytest.raises(ValueError):
            p.weights = [1.0, 2.0]
