
# nanograd

> **Credit:** nanograd is an educational fork of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). The autograd engine, the neural net library, the demo notebooks and the original README content are his work, used under the MIT license (see `LICENSE`). This fork only adds teaching material on top: the `lessons/` notebooks and the supporting modules (`gradcheck`, `losses`, `optim`, `datasets`, `diagnostics`, `viz`). All credit for the original design goes to him.

![awww](puppy.jpg)

A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

### Installation

From a clone of this repository:

```bash
pip install -e .
pip install numpy matplotlib jupyter   # for the notebooks and lessons
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from nanograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `nanograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Training a GPT

For a more advanced example, see [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), which trains and samples from a full GPT-2-like transformer in pure, dependency-free Python. It builds on a more efficient and better version of the autograd engine here (storing local gradients at forward time instead of per-op backward closures), and is the complete algorithm in a single file — everything else is just efficiency. See also the accompanying [explainer post](https://karpathy.github.io/2026/02/12/microgpt/) for a detailed walkthrough.

### Lessons

The `lessons/` folder holds four notebooks that build on the engine, each with exercises, automatic checks and collapsed solution cells. They are best done in this order:

| notebook | topic | library code it builds |
|---|---|---|
| `01_gradcheck_and_new_ops.ipynb` | numerical gradient checking, writing `exp` / `log` / `tanh` / `sigmoid`, why gradients accumulate with `+=` | `nanograd/gradcheck.py`, new ops in `nanograd/engine.py` |
| `02_softmax_cross_entropy.ipynb` | stable softmax, cross-entropy and its "prediction − truth" gradient, loss at init, 3-class spiral | `nanograd/losses.py` |
| `03_optimizers.ipynb` | SGD, momentum and Adam on a 2-D ravine and on a real network, max stable learning rate, schedules | `nanograd/optim.py` |
| `04_gradient_flow_diagnostics.ipynb` | step-by-step backprop, graph size, exploding / vanishing activations, He / Xavier init, dead ReLUs, watching a decision boundary form | `nanograd/diagnostics.py`, `nanograd/viz.py` |

Toy datasets (`make_moons`, `make_spiral`) live in `nanograd/datasets.py` and need no numpy or sklearn. `MLP` also takes `act='relu' | 'tanh'` and `init='uniform' | 'xavier' | 'he' | <std>`; the defaults keep the original behaviour.

Every exercise cell is followed by a check cell and a solution cell, so the notebooks run top to bottom as-is. Solution cells are tagged `solution`, so you can produce a student copy without them:

```bash
jupyter nbconvert --to notebook --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags solution \
    --output-dir student lessons/*.ipynb
```

(The markdown "Solution" heading above each solution cell will still be there; delete it or leave it as a marker.)

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from nanograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](gout.svg)

### Running tests

Most tests check gradients numerically and need only pytest. `test/test_engine.py` additionally compares against [PyTorch](https://pytorch.org/) and is skipped if it isn't installed. Then simply:

```bash
python -m pytest
```

### License

MIT
