import math
import pytest
from nanograd.engine import Value
from nanograd.gradcheck import gradcheck, numerical_grad, analytic_grad

@pytest.mark.parametrize("f, xs", [
    (lambda v: v[0] * v[1] + v[0]**3, [1.5, -2.0]),
    (lambda v: v[0] / v[1] - v[1], [0.7, 1.3]),
    (lambda v: v[0].exp(), [0.5]),
    (lambda v: v[0].log(), [2.5]),
    (lambda v: v[0].tanh(), [-0.8]),
    (lambda v: (v[0] * v[1]).tanh().exp() + (v[0]**2 + 1).log(), [0.3, -1.2]),
    (lambda v: v[0] + v[0], [3.0]), # node reuse: gradients must accumulate
])
def test_gradcheck_passes(f, xs):
    assert gradcheck(f, xs)

def test_forward_values():
    assert Value(1.0).exp().data == pytest.approx(math.e)
    assert Value(math.e).log().data == pytest.approx(1.0)
    assert Value(0.5).tanh().data == pytest.approx(math.tanh(0.5))

def test_gradcheck_catches_a_wrong_backward():
    def bad_exp(v):
        out = Value(math.exp(v.data), (v,), 'exp')
        def _backward():
            v.grad += out.grad # forgot to multiply by e^x
        out._backward = _backward
        return out
    assert not gradcheck(lambda v: bad_exp(v[0]), [1.0])

def test_numerical_matches_analytic():
    f = lambda v: v[0]**2 * v[1]
    xs = [3.0, -2.0]
    for a, n in zip(analytic_grad(f, xs), numerical_grad(f, xs)):
        assert a == pytest.approx(n, rel=1e-6)
