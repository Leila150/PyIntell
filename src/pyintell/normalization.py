"""Normalization layers used by neural-network components."""

import numpy as np


class LayerNorm:
    """Layer normalization over the final dimension."""

    def __init__(self, features, eps=1e-5):
        self.features = int(features)
        self.eps = float(eps)
        self.gamma = np.ones(self.features, dtype=np.float32)
        self.beta = np.zeros(self.features, dtype=np.float32)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != self.features:
            raise ValueError("last dimension does not match features")
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(variance + self.eps) * self.gamma + self.beta


class RMSNorm:
    """Root-mean-square normalization without mean subtraction."""

    def __init__(self, features, eps=1e-6):
        self.features = int(features)
        self.eps = float(eps)
        self.weight = np.ones(self.features, dtype=np.float32)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != self.features:
            raise ValueError("last dimension does not match features")
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        y = y * np.asarray(gamma)
    if beta is not None:
        y = y + np.asarray(beta)
    return y


def rms_norm(x, weight=None, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if weight is None else y * np.asarray(weight)
