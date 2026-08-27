"""Attention primitives."""

import numpy as np


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention(query, key, value, mask=None):
    """Scaled dot-product attention."""
    depth = query.shape[-1]
    scores = query @ np.swapaxes(key, -1, -2) / np.sqrt(depth)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = _softmax(scores)
    return weights @ value


def self_attention(x, mask=None):
    """Self-attention using the input as query, key, and value."""
    return attention(x, x, x, mask)


def causal_attention(x):
    """Self-attention with a causal mask that blocks future tokens."""
    length = x.shape[-2]
    mask = np.tril(np.ones((length, length), dtype=bool))
    return attention(x, x, x, mask)


def multihead_attention(x, heads=8, mask=None):
    """A compact multi-head self-attention implementation."""
    if x.shape[-1] % heads:
        raise ValueError("embedding dimension must be divisible by heads")
    depth = x.shape[-1] // heads
    pieces = np.split(x, heads, axis=-1)
    outputs = [attention(piece, piece, piece, mask) for piece in pieces]
    return np.concatenate(outputs, axis=-1)
