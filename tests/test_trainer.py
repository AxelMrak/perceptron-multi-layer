import pytest
from model.perceptron import Perceptron
from model.activation import StepActivation
from model.dataset import LogicGateDataset
from controller.trainer import TrainerController


def _make_controller(max_epochs=100):
    p = Perceptron(input_size=3, activation=StepActivation(), learning_rate=0.6)
    return TrainerController(p, LogicGateDataset.and_gate(), max_epochs=max_epochs)


class TestInit:
    def test_epoch_starts_at_zero(self):
        c = _make_controller()
        assert c.current_epoch == 0

    def test_not_converged_initially(self):
        c = _make_controller()
        assert c.converged is False

    def test_error_starts_at_zero(self):
        c = _make_controller()
        assert c.current_error == 0.0

    def test_perceptron_property_returns_same_object(self):
        p = Perceptron(input_size=3, activation=StepActivation(), learning_rate=0.6)
        c = TrainerController(p, LogicGateDataset.and_gate())
        assert c.perceptron is p


class TestTrainEpoch:
    def test_increments_epoch_counter(self):
        c = _make_controller()
        c.train_epoch()
        assert c.current_epoch == 1
        c.train_epoch()
        assert c.current_epoch == 2

    def test_updates_sample_state(self):
        c = _make_controller()
        c.train_epoch()
        assert c.current_sample_input is not None
        assert c.current_sample_expected != 0.0
        assert c.current_sample_predicted != 0.0

    def test_converges_with_and_data(self):
        c = _make_controller(max_epochs=200)
        for _ in range(200):
            if c.train_epoch():
                break
        assert c.converged is True

    def test_returns_false_when_errors_exist(self):
        p = Perceptron(input_size=3, activation=StepActivation(), learning_rate=0.6)
        p.weights = [0.0, 0.0, 0.0]
        c = TrainerController(p, LogicGateDataset.and_gate())
        result = c.train_epoch()
        assert result is False

    def test_returns_true_when_no_errors(self):
        p = Perceptron(input_size=3, activation=StepActivation(), learning_rate=0.6)
        p.weights = [-1.0, 1.0, 1.0]
        c = TrainerController(p, LogicGateDataset.and_gate())
        result = c.train_epoch()
        assert result is True
        assert c.converged is True


class TestTrainFull:
    def test_converges_with_and_data(self):
        c = _make_controller(max_epochs=500)
        result = c.train_full()
        assert result is True
        assert c.converged is True
        assert c.current_epoch > 0

    def test_returns_false_when_max_epochs_exceeded(self):
        c = _make_controller(max_epochs=0)
        result = c.train_full()
        assert result is False
        assert c.current_epoch == 0

    def test_perceptron_learns_and_gate(self):
        c = _make_controller(max_epochs=500)
        c.train_full()
        assert c.perceptron.predict([1.0, 1.0, 1.0]) == 1.0
        assert c.perceptron.predict([1.0, 1.0, -1.0]) == -1.0
        assert c.perceptron.predict([1.0, -1.0, 1.0]) == -1.0
        assert c.perceptron.predict([1.0, -1.0, -1.0]) == -1.0


class TestTestPrediction:
    def test_valid_input_one_one(self):
        c = _make_controller()
        result = c.test_prediction("1", "1")
        assert result.startswith("Result:")

    def test_input_not_one_or_minus_one_raises(self):
        c = _make_controller()
        with pytest.raises(ValueError, match="Only values 1 or -1"):
            c.test_prediction("0", "1")

    def test_input_text_raises(self):
        c = _make_controller()
        with pytest.raises(ValueError, match="Only values 1 or -1"):
            c.test_prediction("a", "b")

    def test_single_invalid_raises(self):
        c = _make_controller()
        with pytest.raises(ValueError):
            c.test_prediction("1", "2")

    def test_after_training_correct_and_results(self):
        c = _make_controller(max_epochs=500)
        c.train_full()
        assert "True" in c.test_prediction("1", "1")
        assert "False" in c.test_prediction("1", "-1")
        assert "False" in c.test_prediction("-1", "1")
        assert "False" in c.test_prediction("-1", "-1")
