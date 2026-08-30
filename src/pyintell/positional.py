"""Positional representations for sequence models."""

import numpy as np


def sinusoidal(length, features, base=10000.0):
    """Create the classic sinusoidal positional encoding."""
    length, features = int(length), int(features)
    if length < 0 or features <= 0:
        raise ValueError("length must be non-negative and features must be positive")
    positions = np.arange(length, dtype=np.float32)[:, None]
    indices = np.arange(0, features, 2, dtype=np.float32)
    div = np.exp(-np.log(base) * indices / features)
    enc = np.zeros((length, features), dtype=np.float32)
    enc[:, 0::2] = np.sin(positions * div)
    if features > 1:
        enc[:, 1::2] = np.cos(positions * div[:enc[:, 1::2].shape[1]])
    return enc


def add_sinusoidal(x, base=10000.0):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim < 2:
        raise ValueError("x must have at least two dimensions")
    return x + sinusoidal(x.shape[-2], x.shape[-1], base=base)


def rotary_frequencies(length, dimension, base=10000.0):
    """Return inverse frequencies used by rotary positional embeddings."""
    length, dimension = int(length), int(dimension)
    if dimension <= 0 or dimension % 2:
        raise ValueError("dimension must be a positive even integer")
    inv_freq = 1.0 / (base ** (np.arange(0, dimension, 2, dtype=np.float32) / dimension))
    positions = np.arange(length, dtype=np.float32)
    return positions[:, None] * inv_freq[None, :]
