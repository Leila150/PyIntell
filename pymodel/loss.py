"""Loss functions."""

import numpy as np


def cross_entropy(logits, targets):
    """Mean categorical cross-entropy for logits shaped (..., classes)."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    flat = log_probs.reshape(-1, log_probs.shape[-1])
    target_flat = targets.reshape(-1)
    return float(-np.mean(flat[np.arange(target_flat.size), target_flat]))


def mse(predictions, targets):
    """Mean squared error."""
    return float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))


def loss(predictions, targets, kind="cross_entropy"):
    """Dispatch to a supported loss function."""
    if kind == "cross_entropy":
        return cross_entropy(predictions, targets)
    if kind == "mse":
        return mse(predictions, targets)
    raise ValueError(f"unknown loss: {kind}")
