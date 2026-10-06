"""
Numerical gradient checking: the ground truth for any backward pass.

The idea: the derivative is the slope, so we can estimate it by nudging an
input a tiny bit in each direction and measuring how much the output moves:

    df/dx ~= (f(x + h) - f(x - h)) / (2h)

This is slow (two forward passes per input) but needs no calculus, so it is
the perfect referee for the fast, analytic gradients that backward() computes.
"""
from nanograd.engine import Value

def numerical_grad(f, xs, h=1e-6):
    """ estimate df/dx_i for each float in xs using central differences.
    f takes a list of Values and returns a single Value. """
    grads = []
    for i in range(len(xs)):
        plus  = [Value(x + h if j == i else x) for j, x in enumerate(xs)]
        minus = [Value(x - h if j == i else x) for j, x in enumerate(xs)]
        grads.append((f(plus).data - f(minus).data) / (2 * h))
    return grads

def analytic_grad(f, xs):
    """ the gradients computed by backpropagation """
    vals = [Value(x) for x in xs]
    f(vals).backward()
    return [v.grad for v in vals]

def rel_error(a, b):
    """ relative error that behaves sensibly when both gradients are near zero """
    return abs(a - b) / max(1.0, abs(a), abs(b))

def gradcheck(f, xs, h=1e-6, tol=1e-5, verbose=False):
    """ compare backprop gradients against numerical ones; True if they all agree.
    Note: checks fail at kinks (e.g. relu at exactly 0) where the derivative is undefined. """
    analytic = analytic_grad(f, xs)
    numeric = numerical_grad(f, xs, h)
    errors = [rel_error(a, n) for a, n in zip(analytic, numeric)]
    ok = all(e < tol for e in errors)
    if verbose:
        print(f"{'input':>10} {'analytic':>14} {'numeric':>14} {'rel error':>10}")
        for x, a, n, e in zip(xs, analytic, numeric, errors):
            flag = '' if e < tol else '  <-- MISMATCH'
            print(f"{x:>10.4f} {a:>14.6f} {n:>14.6f} {e:>10.2e}{flag}")
        print("PASS" if ok else "FAIL")
    return ok
