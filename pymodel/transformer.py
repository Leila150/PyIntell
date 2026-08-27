"""Transformer building blocks."""

import numpy as np

from .attention import multihead_attention, cross_attention, causal_mask
from .layers import feedforward, layer_norm


def transformer_block(x, heads=8, hidden_size=None, causal=True):
    """Apply attention and feed-forward sublayers with residual connections."""
    length = x.shape[-2]
    mask = causal_mask(length) if causal else None
    attended = multihead_attention(layer_norm(x), heads=heads, mask=mask)
    x = x + attended
    x = x + feedforward(layer_norm(x), hidden_size=hidden_size)
    return x


def transformer(x, layers=4, heads=8, hidden_size=None, causal=True):
    """Stack Transformer blocks."""
    for _ in range(layers):
        x = transformer_block(x, heads=heads, hidden_size=hidden_size, causal=causal)
    return layer_norm(x)


def encoder(x, layers=4, heads=8, hidden_size=None):
    """Run an encoder-only Transformer stack with bidirectional attention."""
    for _ in range(layers):
        x = transformer_block(x, heads=heads, hidden_size=hidden_size, causal=False)
    return layer_norm(x)


def decoder(x, context=None, memory=None, layers=4, heads=8, hidden_size=None):
    """Run a causal decoder stack with optional encoder context.

    ``memory`` is accepted as an alias for ``context`` for compatibility
    with encoder-decoder terminology and common Transformer APIs.
    """
    if context is not None and memory is not None:
        raise ValueError("provide either context or memory, not both")

    if memory is not None:
        context = memory

    for _ in range(layers):
        normalized = layer_norm(x)
        x = x + multihead_attention(
            normalized,
            heads=heads,
            mask=causal_mask(x.shape[-2]),
        )
        if context is not None:
            x = x + cross_attention(layer_norm(x), context)
        x = x + feedforward(layer_norm(x), hidden_size=hidden_size)

    return layer_norm(x)
