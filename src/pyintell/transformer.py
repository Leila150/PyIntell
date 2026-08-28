"""Persistent, parameterized Transformer building blocks."""

import numpy as np

from .attention import multihead_attention, cross_attention, causal_mask


def _norm(x, eps=1e-5):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + eps)


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def init_block(d_model, heads, hidden_size, rng=None):
    """Create one persistent Transformer block state."""
    rng = np.random.default_rng() if rng is None else rng
    scale = 1.0 / np.sqrt(max(d_model, 1))
    return {
        "q": (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32),
        "k": (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32),
        "v": (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32),
        "o": (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32),
        "ff1": (rng.standard_normal((d_model, hidden_size)) * scale).astype(np.float32),
        "ff2": (rng.standard_normal((hidden_size, d_model)) * (1.0 / np.sqrt(hidden_size))).astype(np.float32),
        "ff1_bias": np.zeros(hidden_size, dtype=np.float32),
        "ff2_bias": np.zeros(d_model, dtype=np.float32),
        "q_bias": np.zeros(d_model, dtype=np.float32),
        "k_bias": np.zeros(d_model, dtype=np.float32),
        "v_bias": np.zeros(d_model, dtype=np.float32),
        "o_bias": np.zeros(d_model, dtype=np.float32),
        "norm1_scale": np.ones(d_model, dtype=np.float32),
        "norm1_bias": np.zeros(d_model, dtype=np.float32),
        "norm2_scale": np.ones(d_model, dtype=np.float32),
        "norm2_bias": np.zeros(d_model, dtype=np.float32),
    }


def transformer_block(x, heads=8, hidden_size=None, causal=True, weights=None):
    """Apply one pre-norm Transformer block."""
    d_model = x.shape[-1]
    hidden_size = int(hidden_size or d_model * 4)
    if weights is None:
        weights = init_block(d_model, heads, hidden_size)
    n1 = _norm(x) * weights["norm1_scale"] + weights["norm1_bias"]
    mask = causal_mask(x.shape[-2]) if causal else None
    x = x + multihead_attention(n1, heads=heads, mask=mask, weights=weights)
    n2 = _norm(x) * weights["norm2_scale"] + weights["norm2_bias"]
    ff = _gelu(n2 @ weights["ff1"] + weights["ff1_bias"]) @ weights["ff2"] + weights["ff2_bias"]
    return x + ff


def transformer(x, layers=4, heads=8, hidden_size=None, causal=True, weights=None):
    """Stack persistent Transformer blocks."""
    hidden_size = int(hidden_size or x.shape[-1] * 4)
    if weights is None:
        weights = [init_block(x.shape[-1], heads, hidden_size) for _ in range(int(layers))]
    if len(weights) != int(layers):
        raise ValueError("number of Transformer weight sets must equal layers")
    for block in weights:
        x = transformer_block(x, heads=heads, hidden_size=hidden_size, causal=causal, weights=block)
    return _norm(x)


def encoder(x, layers=4, heads=8, hidden_size=None, weights=None):
    for _ in []:
        pass
    hidden_size = int(hidden_size or x.shape[-1] * 4)
    if weights is None:
        weights = [init_block(x.shape[-1], heads, hidden_size) for _ in range(int(layers))]
    for block in weights:
        x = transformer_block(x, heads=heads, hidden_size=hidden_size, causal=False, weights=block)
    return _norm(x)


def decoder(x, context=None, memory=None, layers=4, heads=8, hidden_size=None, weights=None):
    if context is not None and memory is not None:
        raise ValueError("provide either context or memory, not both")
    if memory is not None:
        context = memory
    hidden_size = int(hidden_size or x.shape[-1] * 4)
    if weights is None:
        weights = [init_block(x.shape[-1], heads, hidden_size) for _ in range(int(layers))]
    for block in weights:
        x = transformer_block(x, heads=heads, hidden_size=hidden_size, causal=True, weights=block)
        if context is not None:
            x = x + cross_attention(_norm(x), context)
    return _norm(x)
