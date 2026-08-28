"""Loss functions."""

import numpy as np


def cross_entropy(logits, targets):
    logits = np.asarray(logits, dtype=np.float64); targets = np.asarray(targets, dtype=np.int64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    flat = log_probs.reshape(-1, log_probs.shape[-1]); target_flat = targets.reshape(-1)
    return float(-np.mean(flat[np.arange(target_flat.size), target_flat]))

def binary_cross_entropy(predictions, targets, eps=1e-12):
    p = np.clip(np.asarray(predictions), eps, 1-eps); t = np.asarray(targets)
    return float(-np.mean(t*np.log(p) + (1-t)*np.log(1-p)))

def mse(predictions, targets): return float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))
def mae(predictions, targets): return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(targets))))
def huber_loss(predictions, targets, delta=1.0):
    error = np.abs(np.asarray(predictions) - np.asarray(targets)); quadratic = np.minimum(error, delta)
    linear = error - quadratic; return float(np.mean(0.5*quadratic**2 + delta*linear))

def kl_divergence(p, q, eps=1e-12):
    p = np.clip(np.asarray(p), eps, None); q = np.clip(np.asarray(q), eps, None)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def contrastive_loss(a, b, target, margin=1.0):
    distance = np.linalg.norm(np.asarray(a)-np.asarray(b), axis=-1)
    target = np.asarray(target); return float(np.mean(target*distance**2 + (1-target)*np.maximum(0, margin-distance)**2))

def label_smoothing(targets, classes, smoothing=0.1):
    result = np.full((len(targets), classes), smoothing/(classes-1), dtype=np.float32)
    result[np.arange(len(targets)), np.asarray(targets)] = 1-smoothing
    return result

def loss(predictions, targets, kind="cross_entropy"):
    functions = {"cross_entropy": cross_entropy, "mse": mse, "mae": mae, "binary_cross_entropy": binary_cross_entropy, "huber": huber_loss}
    if kind not in functions: raise ValueError(f"unknown loss: {kind}")
    return functions[kind](predictions, targets)
