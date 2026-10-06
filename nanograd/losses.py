"""
Loss functions built purely out of Value operations, so backward() just works.
"""
from nanograd.engine import Value

def mse(pred, target):
    """ mean squared error for a list of predictions vs a list of numbers """
    return sum((p - t)**2 for p, t in zip(pred, target)) * (1.0 / len(pred))

def hinge(score, y):
    """ svm "max-margin" loss for one example, label y in {-1, +1} (as in demo.ipynb) """
    return (1 + -y * score).relu()

def softmax(logits):
    """ turn a list of raw scores into a list of probabilities that sum to 1.
    Subtracting the max logit doesn't change the result (it cancels in the ratio)
    but keeps exp() from overflowing on large scores. """
    m = max(l.data for l in logits)  # a plain float: a constant shift, no gradient needed
    exps = [(l - m).exp() for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    """ log of softmax, computed stably as (l - m) - log(sum(exp(l - m))) """
    m = max(l.data for l in logits)
    shifted = [l - m for l in logits]
    logsumexp = sum(s.exp() for s in shifted).log()
    return [s - logsumexp for s in shifted]

def cross_entropy(logits, target):
    """ negative log probability of the correct class (target is an int index).
    This is the loss used to train virtually every classifier and language model. """
    return -log_softmax(logits)[target]
