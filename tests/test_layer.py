from model.layer import Layer
from model.activation import SigmoidActivation, StepActivation


class TestLayerInit:
    def test_creates_correct_number_of_neurons(self):
        layer = Layer(3, 4, StepActivation(), 0.5)
        assert layer.num_neurons == 3
        assert layer.num_inputs == 4
        assert len(layer.weights) == 3
        assert all(len(w) == 4 for w in layer.weights)


class TestLayerForward:
    def test_output_length_matches_neurons(self):
        layer = Layer(2, 3, StepActivation(), 0.5)
        output = layer.forward([1.0, 0.5, -0.3])
        assert len(output) == 2

    def test_known_weights_give_expected_output(self):
        layer = Layer(2, 3, StepActivation(), 0.5)
        layer._weights = [[0.5, 0.2, -0.3], [-0.1, 0.8, 0.4]]
        out = layer.forward([1.0, 1.0, 1.0])
        # [0.5+0.2-0.3=0.4 → 1.0], [-0.1+0.8+0.4=1.1 → 1.0]
        assert out == [1.0, 1.0]

    def test_input_size_mismatch_raises(self):
        import pytest
        layer = Layer(2, 3, StepActivation(), 0.5)
        with pytest.raises(ValueError):
            layer.forward([1.0, 2.0])


class TestLayerBackward:
    def test_weights_change_after_backward(self):
        layer = Layer(1, 2, SigmoidActivation(), 0.5)
        layer._weights = [[0.5, -0.3]]
        layer.forward([1.0, 2.0])
        old_w0 = layer._weights[0][0]
        layer.backward([0.2])
        assert layer._weights[0][0] != old_w0

    def test_backward_returns_errors_for_previous_layer(self):
        layer = Layer(2, 3, SigmoidActivation(), 0.5)
        layer.forward([1.0, 0.5, -0.3])
        prev_errors = layer.backward([0.1, -0.1])
        assert len(prev_errors) == 3
