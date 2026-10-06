import math
import pytest
from nanograd.engine import Value
from nanograd.gradcheck import gradcheck
from nanograd.losses import softmax, log_softmax, cross_entropy, mse, hinge

def test_softmax_sums_to_one():
    probs = softmax([Value(1.0), Value(2.0), Value(3.0)])
    assert sum(p.data for p in probs) == pytest.approx(1.0)
    assert probs[2].data > probs[1].data > probs[0].data

def test_softmax_is_stable_for_huge_logits():
    probs = softmax([Value(1000.0), Value(1001.0)])
    assert probs[1].data == pytest.approx(1 / (1 + math.exp(-1)))

def test_cross_entropy_value():
    logits = [Value(2.0), Value(0.5), Value(-1.0)]
    expected = -math.log(math.exp(2.0) / sum(math.exp(x) for x in (2.0, 0.5, -1.0)))
    assert cross_entropy(logits, 0).data == pytest.approx(expected)

def test_cross_entropy_gradient_is_probs_minus_onehot():
    xs = [2.0, 0.5, -1.0]
    logits = [Value(x) for x in xs]
    cross_entropy(logits, 1).backward()
    probs = [p.data for p in softmax([Value(x) for x in xs])]
    for k, l in enumerate(logits):
        assert l.grad == pytest.approx(probs[k] - (1 if k == 1 else 0))

@pytest.mark.parametrize("target", [0, 1, 2])
def test_cross_entropy_gradcheck(target):
    assert gradcheck(lambda v: cross_entropy(v, target), [0.3, -1.1, 2.0])

def test_log_softmax_gradcheck():
    assert gradcheck(lambda v: log_softmax(v)[0] + 2 * log_softmax(v)[2], [0.3, -1.1, 2.0])

def test_mse_and_hinge():
    assert mse([Value(1.0), Value(3.0)], [0.0, 1.0]).data == pytest.approx(2.5)
    assert hinge(Value(2.0), 1).data == 0
    assert hinge(Value(0.5), -1).data == pytest.approx(1.5)
