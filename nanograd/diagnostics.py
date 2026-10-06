"""
Tools for looking inside a network: the graph that backward() walks, and the
activations and gradients flowing through each layer.
"""
from nanograd.engine import Value

def trace(root):
    """ all nodes and edges of the graph that produced root """
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def count_nodes(root):
    """ how many Value objects (scalars) were created to compute root """
    return len(trace(root)[0])

def topo_order(root):
    """ the same topological ordering that Value.backward() uses """
    topo, visited = [], set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(root)
    return topo

def backward_steps(root):
    """ run backpropagation one node at a time, yielding after each step.
    Yields (node, order) where node is the Value whose _backward just ran and
    order is the full reversed topological order, so you can watch gradients
    arrive at each node in turn. Grads are reset to zero first. """
    order = list(reversed(topo_order(root)))
    for v in order:
        v.grad = 0
    root.grad = 1
    for v in order:
        v._backward()
        yield v, order

def _as_list(out):
    return out if isinstance(out, list) else [out]

def layer_activations(model, X):
    """ run each input in X through the model layer by layer.
    Returns acts[layer][sample][neuron] as plain floats. """
    acts = [[] for _ in model.layers]
    for xrow in X:
        x = [Value(xi) for xi in xrow]
        for li, layer in enumerate(model.layers):
            x = layer(x)
            acts[li].append([v.data for v in _as_list(x)])
            x = _as_list(x)
    return acts

def dead_fraction(acts_layer):
    """ fraction of neurons in a layer that output exactly 0 for every sample.
    A dead ReLU gets zero gradient forever, so it can never recover. """
    n_neurons = len(acts_layer[0])
    dead = sum(1 for j in range(n_neurons) if all(sample[j] == 0 for sample in acts_layer))
    return dead / n_neurons

def saturated_fraction(acts_layer, threshold=0.99):
    """ fraction of tanh activations stuck near +-1, where the local gradient ~ 0 """
    flat = [a for sample in acts_layer for a in sample]
    return sum(1 for a in flat if abs(a) > threshold) / len(flat)

def layer_grads(model):
    """ gradients of each layer's weights (call after backward()) as plain floats """
    return [[p.grad for n in layer.neurons for p in n.w] for layer in model.layers]
