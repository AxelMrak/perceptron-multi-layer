from model.activation import StepActivation


class TestStepActivationCompute:
    def test_positive_returns_one(self):
        assert StepActivation().compute(0.5) == 1.0
        assert StepActivation().compute(100.0) == 1.0

    def test_zero_returns_minus_one(self):
        assert StepActivation().compute(0.0) == -1.0

    def test_negative_returns_minus_one(self):
        assert StepActivation().compute(-0.5) == -1.0
        assert StepActivation().compute(-100.0) == -1.0


class TestStepActivationDerivative:
    def test_derivative_always_zero(self):
        assert StepActivation().derivative(0.0) == 0.0
        assert StepActivation().derivative(1.0) == 0.0
        assert StepActivation().derivative(-1.0) == 0.0
