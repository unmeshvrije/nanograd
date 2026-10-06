import random
import pytest
from nanograd.engine import Value
from nanograd.optim import SGD, Adam

def quadratic_loss(params):
    # minimum at (3, -2)
    return (params[0] - 3)**2 + 10 * (params[1] + 2)**2

@pytest.mark.parametrize("make_opt", [
    lambda ps: SGD(ps, lr=0.04),
    lambda ps: SGD(ps, lr=0.02, momentum=0.9),
    lambda ps: Adam(ps, lr=0.1),
])
def test_optimizers_find_the_minimum(make_opt):
    params = [Value(0.0), Value(0.0)]
    opt = make_opt(params)
    for _ in range(500):
        loss = quadratic_loss(params)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert params[0].data == pytest.approx(3, abs=1e-2)
    assert params[1].data == pytest.approx(-2, abs=1e-2)

def test_sgd_single_step():
    p = Value(1.0)
    p.grad = 0.5
    SGD([p], lr=0.1).step()
    assert p.data == pytest.approx(0.95)

def test_adam_first_step_is_lr_sized():
    # thanks to bias correction, Adam's very first step is ~lr regardless of gradient scale
    for g in (1e-3, 1.0, 1e3):
        p = Value(0.0)
        p.grad = g
        Adam([p], lr=0.01).step()
        assert p.data == pytest.approx(-0.01, rel=1e-4)

def test_zero_grad():
    ps = [Value(1.0), Value(2.0)]
    for p in ps:
        p.grad = 5.0
    SGD(ps).zero_grad()
    assert all(p.grad == 0 for p in ps)
