"""Embedding layers implemented with NumPy."""

import numpy as np


def embedding(tokens, dimensions=128, weights=None, vocab_size=None):
    ids = np.asarray(tokens, dtype=np.int64)
    if weights is None:
        if vocab_size is None: vocab_size = int(ids.max()) + 1 if ids.size else 1
        weights = np.random.randn(vocab_size, dimensions).astype(np.float32) * 0.02
    return weights[ids]


def positional_embedding(sequence_length, dimensions, weights=None):
    if weights is None: weights = np.random.randn(sequence_length, dimensions).astype(np.float32) * 0.02
    return weights[:sequence_length]


def sinusoidal_embedding(sequence_length, dimensions):
    positions = np.arange(sequence_length)[:, None]
    div = np.exp(np.arange(0, dimensions, 2) * (-np.log(10000.0) / dimensions))
    result = np.zeros((sequence_length, dimensions), dtype=np.float32)
    result[:, 0::2] = np.sin(positions * div)
    result[:, 1::2] = np.cos(positions * div[:result[:, 1::2].shape[1]])
    return result


def rotary_embedding(sequence_length, dimensions): return sinusoidal_embedding(sequence_length, dimensions)
def position_embedding(sequence_length, dimensions): return positional_embedding(sequence_length, dimensions)

def embedding_similarity(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
