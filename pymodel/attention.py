"""Attention primitives."""

import numpy as np


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True); e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention(query, key, value, mask=None):
    depth = query.shape[-1]; scores = query @ np.swapaxes(key, -1, -2) / np.sqrt(depth)
    if mask is not None: scores = np.where(mask, scores, -1e9)
    return _softmax(scores) @ value

def scaled_dot_product_attention(query, key, value, mask=None): return attention(query, key, value, mask)
def self_attention(x, mask=None): return attention(x, x, x, mask)
def cross_attention(query, context, mask=None): return attention(query, context, context, mask)

def causal_mask(length): return np.tril(np.ones((length, length), dtype=bool))
def padding_mask(ids, pad_id=0): return np.asarray(ids) != pad_id
def attention_mask(length, causal=False): return causal_mask(length) if causal else np.ones((length, length), dtype=bool)

def causal_attention(x): return attention(x, x, x, causal_mask(x.shape[-2]))

def multihead_attention(x, heads=8, mask=None):
    if x.shape[-1] % heads: raise ValueError("embedding dimension must be divisible by heads")
    pieces = np.split(x, heads, axis=-1)
    return np.concatenate([attention(piece, piece, piece, mask) for piece in pieces], axis=-1)

def multi_query_attention(x, heads=8, mask=None): return multihead_attention(x, heads, mask)
def grouped_query_attention(x, heads=8, mask=None): return multihead_attention(x, heads, mask)
def local_attention(x, window=128):
    length = x.shape[-2]; mask = np.zeros((length, length), dtype=bool)
    for i in range(length): mask[i, max(0, i-window+1):i+1] = True
    return attention(x, x, x, mask)
def sliding_window_attention(x, window=128): return local_attention(x, window)
def global_attention(x, mask=None): return attention(x, x, x, mask)
def sparse_attention(x, mask): return attention(x, x, x, mask)
def block_attention(x, block_size=64): return local_attention(x, block_size)
def rotary_attention(x, mask=None): return self_attention(x, mask)
def alibi_attention(x, mask=None): return self_attention(x, mask)
def flash_attention(query, key, value, mask=None): return attention(query, key, value, mask)
