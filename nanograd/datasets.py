"""
Tiny toy datasets in pure Python (no numpy / sklearn needed).
Each returns X as a list of [x0, x1] points and y as a list of int class labels.
"""
import math
import random

def make_moons(n_samples=100, noise=0.1, seed=None):
    """ two interleaving half circles, labels 0 and 1 """
    rng = random.Random(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    X, y = [], []
    for i in range(n_out):
        t = math.pi * i / max(1, n_out - 1)
        X.append([math.cos(t), math.sin(t)]); y.append(0)
    for i in range(n_in):
        t = math.pi * i / max(1, n_in - 1)
        X.append([1 - math.cos(t), 0.5 - math.sin(t)]); y.append(1)
    X = [[a + rng.gauss(0, noise), b + rng.gauss(0, noise)] for a, b in X]
    return X, y

def make_spiral(n_per_class=40, n_classes=3, noise=0.2, seed=None):
    """ n_classes interleaved spiral arms, labels 0 .. n_classes-1 """
    rng = random.Random(seed)
    X, y = [], []
    for c in range(n_classes):
        for i in range(n_per_class):
            r = i / n_per_class
            t = c * 2 * math.pi / n_classes + r * 4 + rng.gauss(0, noise)
            X.append([r * math.sin(t), r * math.cos(t)]); y.append(c)
    return X, y
