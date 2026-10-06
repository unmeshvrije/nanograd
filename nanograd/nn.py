import math
import random
from nanograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

def init_weight(nin, init='uniform'):
    """ sample one initial weight for a neuron with `nin` inputs """
    if init == 'uniform':  # the original micrograd scheme, ignores fan-in
        return random.uniform(-1, 1)
    if init == 'xavier':   # std = 1/sqrt(nin): keeps variance stable for tanh
        return random.gauss(0, math.sqrt(1 / nin))
    if init == 'he':       # std = sqrt(2/nin): compensates for ReLU zeroing half its inputs
        return random.gauss(0, math.sqrt(2 / nin))
    if isinstance(init, (int, float)):  # plain gaussian with a given std, for experiments
        return random.gauss(0, init)
    raise ValueError(f"unknown init {init!r}")

class Neuron(Module):

    def __init__(self, nin, nonlin=True, act='relu', init='uniform'):
        assert act in ('relu', 'tanh'), "act must be 'relu' or 'tanh'"
        self.w = [Value(init_weight(nin, init)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
        self.act = act

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if not self.nonlin:
            return act
        return act.relu() if self.act == 'relu' else act.tanh()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.act.capitalize() if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, **kwargs):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1, **kwargs) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
