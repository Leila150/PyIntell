"""Embedding layers implemented with NumPy."""

import numpy as np


def embedding(tokens, dimensions=128, weights=None, vocab_size=None):
    """Map token IDs to vectors. If weights are supplied, use them directly."""
    ids = np.asarray(tokens, dtype=np.int64)
    if weights is None:
        if vocab_size is None:
            vocab_size = int(ids.max()) + 1 if ids.size else 1
        weights = np.random.randn(vocab_size, dimensions).astype(np.float32) * 0.02
    return weights[ids]


def positional_embedding(sequence_length, dimensions, weights=None):
    """Return learned positional vectors for a sequence."""
    if weights is None:
        weights = np.random.randn(sequence_length, dimensions).astype(np.float32) * 0.02
    return weights[:sequence_length]
