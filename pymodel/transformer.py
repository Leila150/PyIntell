"""Transformer building blocks."""

import numpy as np

from .attention import multihead_attention
from .layers import feedforward, layer_norm


def transformer_block(x, heads=8, hidden_size=None, causal=True):
    """Apply attention and feed-forward sublayers with residual connections."""
    length = x.shape[-2]
    mask = np.tril(np.ones((length, length), dtype=bool)) if causal else None
    attended = multihead_attention(layer_norm(x), heads=heads, mask=mask)
    x = x + attended
    x = x + feedforward(layer_norm(x), hidden_size=hidden_size)
    return x


def transformer(x, layers=4, heads=8, hidden_size=None, causal=True):
    """Stack Transformer blocks."""
    for _ in range(layers):
        x = transformer_block(x, heads=heads, hidden_size=hidden_size, causal=causal)
    return layer_norm(x)
