"""Activation functions for neural-network layers."""

import numpy as np


def relu(x):
    return np.maximum(np.asarray(x), 0)


def gelu(x):
    x = np.asarray(x, dtype=np.float32)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def silu(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def tanh(x):
    return np.tanh(np.asarray(x))


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    return exp / np.maximum(np.sum(exp, axis=axis, keepdims=True), 1e-12)


def log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.maximum(np.sum(np.exp(shifted), axis=axis, keepdims=True), 1e-12))
