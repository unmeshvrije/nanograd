"""
Optimizers: given parameters and their gradients, decide how to update them.

Autograd tells us *which direction* reduces the loss; the optimizer decides
*how far* to step, and whether to use any memory of previous steps.
"""
import math

class Optimizer:

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    """ stochastic gradient descent, optionally with momentum.
    momentum=0:  p -= lr * grad
    momentum>0:  v = momentum * v + grad;  p -= lr * v
    The velocity v is a running sum of past gradients, so steps speed up along
    directions that consistently point downhill and cancel out where they zig-zag. """

    def __init__(self, params, lr=0.1, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocity = [0.0 for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self.velocity[i] = self.momentum * self.velocity[i] + p.grad
            p.data -= self.lr * self.velocity[i]

class Adam(Optimizer):
    """ Adam: momentum (first moment m) plus a per-parameter step size that
    shrinks for parameters with large, noisy gradients (second moment v).
    The bias correction fixes m and v starting at zero during the first few steps. """

    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [0.0 for _ in self.params]
        self.v = [0.0 for _ in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
