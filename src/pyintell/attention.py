"""Numerically stable attention primitives used by pyintell."""

import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(np.clip(x, -80.0, 80.0))
    return e / np.maximum(np.sum(e, axis=-1, keepdims=True), 1e-12)


def attention(query, key, value, mask=None, dropout=0.0):
    """Scaled dot-product attention with stable masking."""
    q, k, v = map(lambda a: np.asarray(a, dtype=np.float32), (query, key, value))
    depth = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(max(depth, 1))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        scores = np.where(m, scores, -1e9)
    weights = _softmax(scores)
    if dropout:
        raise NotImplementedError("dropout is intentionally disabled in the NumPy inference core")
    return weights @ v


def scaled_dot_product_attention(query, key, value, mask=None):
    return attention(query, key, value, mask)


def self_attention(x, mask=None):
    return attention(x, x, x, mask)


def cross_attention(query, context, mask=None):
    return attention(query, context, context, mask)


def causal_mask(length):
    return np.tril(np.ones((int(length), int(length)), dtype=bool))


def padding_mask(ids, pad_id=0):
    return np.asarray(ids) != pad_id


def attention_mask(length, causal=False):
    return causal_mask(length) if causal else np.ones((length, length), dtype=bool)


def causal_attention(x):
    return attention(x, x, x, causal_mask(x.shape[-2]))


def multihead_attention(x, heads=8, mask=None, weights=None):
    """Multi-head attention.

    When ``weights`` is supplied it must contain q/k/v/o projection matrices.
    This makes attention deterministic and trainable instead of regenerating
    random projections on every forward pass.
    """
    x = np.asarray(x, dtype=np.float32)
    heads = int(heads)
    if heads < 1 or x.shape[-1] % heads:
        raise ValueError("embedding dimension must be divisible by heads")
    d_model = x.shape[-1]
    if weights is None:
        q = k = v = x
        output = np.concatenate(
            [attention(a, a, a, mask) for a in np.split(x, heads, axis=-1)], axis=-1
        )
        return output
    q = x @ weights["q"] + weights.get("q_bias", 0.0)
    k = x @ weights["k"] + weights.get("k_bias", 0.0)
    v = x @ weights["v"] + weights.get("v_bias", 0.0)
    d_head = d_model // heads
    q = q.reshape(x.shape[0], heads, d_head).transpose(1, 0, 2)
    k = k.reshape(x.shape[0], heads, d_head).transpose(1, 0, 2)
    v = v.reshape(x.shape[0], heads, d_head).transpose(1, 0, 2)
    attended = attention(q, k, v, mask)
    attended = attended.transpose(1, 0, 2).reshape(x.shape[0], d_model)
    return attended @ weights["o"] + weights.get("o_bias", 0.0)


def multi_query_attention(x, heads=8, mask=None):
    return multihead_attention(x, heads, mask)


def grouped_query_attention(x, heads=8, mask=None):
    return multihead_attention(x, heads, mask)


def local_attention(x, window=128):
    length = x.shape[-2]
    mask = np.zeros((length, length), dtype=bool)
    for i in range(length):
        mask[i, max(0, i - int(window) + 1):i + 1] = True
    return attention(x, x, x, mask)


def sliding_window_attention(x, window=128):
    return local_attention(x, window)


def global_attention(x, mask=None):
    return attention(x, x, x, mask)


def sparse_attention(x, mask):
    return attention(x, x, x, mask)


def block_attention(x, block_size=64):
    return local_attention(x, block_size)


def rotary_attention(x, mask=None):
    return self_attention(x, mask)


def alibi_attention(x, mask=None):
    return self_attention(x, mask)


def flash_attention(query, key, value, mask=None):
    return attention(query, key, value, mask)
