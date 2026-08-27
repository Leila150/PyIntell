"""Neural-network layer and activation primitives."""

import numpy as np


def linear(x, weights, bias=None):
    result = np.asarray(x) @ np.asarray(weights)
    return result if bias is None else result + bias


def relu(x): return np.maximum(x, 0)
def leaky_relu(x, negative_slope=0.01):
    x = np.asarray(x); return np.where(x >= 0, x, negative_slope * x)
def gelu(x):
    x = np.asarray(x); return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.asarray(x)))
def tanh(x): return np.tanh(x)
def softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
def silu(x): return np.asarray(x) * sigmoid(x)
def swish(x): return silu(x)
def mish(x): return np.asarray(x) * np.tanh(softplus(x))


def softmax(x, axis=-1):
    x = np.asarray(x); shifted = x - np.max(x, axis=axis, keepdims=True)
    values = np.exp(shifted); return values / np.sum(values, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x = np.asarray(x); shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def activation(x, name="gelu"):
    functions = {"relu": relu, "gelu": gelu, "tanh": tanh, "sigmoid": sigmoid,
                 "leaky_relu": leaky_relu, "softplus": softplus, "silu": silu,
                 "swish": swish, "mish": mish}
    if name not in functions: raise ValueError(f"unknown activation: {name}")
    return functions[name](x)


def layer_norm(x, eps=1e-5):
    x = np.asarray(x); mean = np.mean(x, axis=-1, keepdims=True); variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps)


def batch_norm(x, eps=1e-5):
    x = np.asarray(x); mean = np.mean(x, axis=0, keepdims=True); variance = np.var(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps)


def rms_norm(x, eps=1e-8):
    x = np.asarray(x); return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def dropout(x, probability=0.0, training=True):
    if not training or probability <= 0: return x
    if probability >= 1: raise ValueError("dropout probability must be less than 1")
    mask = (np.random.random(np.asarray(x).shape) >= probability) / (1 - probability)
    return np.asarray(x) * mask


def flatten_layer(x): return np.asarray(x).reshape(-1)


def feedforward(x, hidden_size=None, activation_name="gelu"):
    width = x.shape[-1] if hidden_size is None else hidden_size; input_size = x.shape[-1]
    w1 = np.random.randn(input_size, width).astype(np.float32) * 0.02
    w2 = np.random.randn(width, input_size).astype(np.float32) * 0.02
    return linear(activation(linear(x, w1), activation_name), w2)


def mlp(x, hidden_size=None, activation_name="gelu"): return feedforward(x, hidden_size, activation_name)
def residual(x, function): return np.asarray(x) + function(x)
def residual_block(x, function): return residual(x, function)
