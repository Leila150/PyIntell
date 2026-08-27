"""Small neural-network layer primitives."""

import numpy as np


def linear(x, weights, bias=None):
    """Apply a linear transformation."""
    result = np.asarray(x) @ np.asarray(weights)
    return result if bias is None else result + bias


def relu(x):
    return np.maximum(x, 0)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def activation(x, name="gelu"):
    """Apply a named activation."""
    functions = {"relu": relu, "gelu": gelu, "tanh": np.tanh, "sigmoid": lambda v: 1 / (1 + np.exp(-v))}
    if name not in functions:
        raise ValueError(f"unknown activation: {name}")
    return functions[name](x)


def layer_norm(x, eps=1e-5):
    """Normalize the final dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps)


def dropout(x, probability=0.0, training=True):
    """Apply inverted dropout."""
    if not training or probability <= 0:
        return x
    if probability >= 1:
        raise ValueError("dropout probability must be less than 1")
    mask = (np.random.random(x.shape) >= probability) / (1 - probability)
    return x * mask


def feedforward(x, hidden_size=None, activation_name="gelu"):
    """Apply a compact two-layer feed-forward network."""
    width = x.shape[-1] if hidden_size is None else hidden_size
    input_size = x.shape[-1]
    w1 = np.random.randn(input_size, width).astype(np.float32) * 0.02
    w2 = np.random.randn(width, input_size).astype(np.float32) * 0.02
    return linear(activation(linear(x, w1), activation_name), w2)
