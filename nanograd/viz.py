"""
Plotting helpers. These need extra packages, imported lazily:
    graphviz (pip install graphviz, plus the graphviz binaries) for draw_dot
    numpy + matplotlib for plot_decision_boundary
"""
from nanograd.engine import Value
from nanograd.diagnostics import trace

def draw_dot(root, format='svg', rankdir='LR', highlight=None, done=None):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    highlight: a node to draw in orange (e.g. the one whose _backward just ran)
    done: a set of nodes to draw in green (their _backward has already run)
    """
    from graphviz import Digraph
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    done = done or set()
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        style = {}
        if n is highlight:
            style = {'style': 'filled', 'fillcolor': 'orange'}
        elif n in done:
            style = {'style': 'filled', 'fillcolor': 'palegreen'}
        dot.node(name=str(id(n)), label="{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record', **style)
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def predict(model, xrow):
    """ class prediction: sign of the score for binary models, argmax for multi-class """
    out = model([Value(xi) for xi in xrow])
    if isinstance(out, list):
        return max(range(len(out)), key=lambda k: out[k].data)
    return int(out.data > 0)

def plot_decision_boundary(model, X, y, ax=None, h=0.1, title=None):
    """ shade the plane by the model's predicted class and overlay the data """
    import numpy as np
    import matplotlib.pyplot as plt
    X = np.asarray(X)
    ax = ax or plt.gca()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.array([predict(model, row) for row in np.c_[xx.ravel(), yy.ravel()]]).reshape(xx.shape)
    # fix the color range so a class keeps its color even if the model never predicts some class
    n_classes = max(max(y), Z.max()) + 1
    ax.contourf(xx, yy, Z, levels=np.arange(-0.5, n_classes), cmap=plt.cm.Spectral, vmin=0, vmax=n_classes - 1, alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral, vmin=0, vmax=n_classes - 1, edgecolors='k', linewidths=0.3)
    ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax
