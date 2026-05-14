from model.activation import SigmoidActivation


class TestSigmoidActivationCompute:
    def test_zero_returns_half(self):
        assert abs(SigmoidActivation().compute(0.0) - 0.5) < 0.001

    def test_large_positive_approaches_one(self):
        assert SigmoidActivation().compute(10.0) > 0.999

    def test_large_negative_approaches_zero(self):
        assert SigmoidActivation().compute(-10.0) < 0.001

    def test_symmetric(self):
        sig = SigmoidActivation()
        assert abs(sig.compute(2.0) + sig.compute(-2.0) - 1.0) < 0.001


class TestSigmoidActivationDerivative:
    def test_derivative_at_zero_is_quarter(self):
        assert abs(SigmoidActivation().derivative(0.0) - 0.25) < 0.001

    def test_derivative_is_positive(self):
        assert SigmoidActivation().derivative(1.0) > 0

    def test_derivative_approaches_zero_at_extremes(self):
        sig = SigmoidActivation()
        assert sig.derivative(50.0) < 0.001
        assert sig.derivative(-50.0) < 0.001
