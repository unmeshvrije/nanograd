import random
import pytest
from nanograd.engine import Value
from nanograd.nn import MLP, Neuron
from nanograd.datasets import make_moons, make_spiral
from nanograd import diagnostics

def test_mlp_options_and_shapes():
    random.seed(0)
    m = MLP(2, [4, 3], act='tanh', init='xavier')
    out = m([Value(0.1), Value(-0.2)])
    assert len(out) == 3
    assert "TanhNeuron" in repr(m)

def test_bad_init_raises():
    with pytest.raises(ValueError):
        Neuron(2, init='nope')

def test_datasets():
    X, y = make_moons(50, seed=0)
    assert len(X) == 50 and set(y) == {0, 1}
    X, y = make_spiral(10, 4, seed=0)
    assert len(X) == 40 and set(y) == {0, 1, 2, 3}

def test_backward_steps_matches_backward():
    def build():
        a, b = Value(2.0), Value(-3.0)
        return a, b, (a * b + a).tanh()
    a, b, out = build()
    for _ in diagnostics.backward_steps(out):
        pass
    a2, b2, out2 = build()
    out2.backward()
    assert (a.grad, b.grad) == pytest.approx((a2.grad, b2.grad))

def test_dead_fraction_and_counts():
    acts = [[0.0, 1.0], [0.0, 0.0]]
    assert diagnostics.dead_fraction(acts) == 0.5
    x = Value(1.0)
    assert diagnostics.count_nodes(x * 2 + 1) == 5
