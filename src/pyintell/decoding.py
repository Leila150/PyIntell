"""Token decoding strategies for language-model inference."""

import numpy as np

from .activations import softmax


def greedy(logits):
    """Return the highest-logit token."""
    x = np.asarray(logits)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("logits must be a non-empty one-dimensional array")
    return int(np.argmax(x))


def top_p(logits, p=0.9, temperature=1.0, rng=None):
    """Nucleus sampling: sample from the smallest set reaching probability p."""
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("logits must be a non-empty one-dimensional array")
    probs = softmax(x / temperature).astype(np.float64)
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    keep = cumulative <= p
    keep[0] = True
    candidates = order[keep]
    candidate_probs = probs[candidates]
    candidate_probs /= candidate_probs.sum()
    generator = rng or np.random.default_rng()
    return int(generator.choice(candidates, p=candidate_probs))


def top_k_top_p(logits, k=50, p=0.9, temperature=1.0, rng=None):
    """Apply both top-k and nucleus filtering before sampling."""
    x = np.asarray(logits, dtype=np.float64).copy()
    if x.ndim != 1 or x.size == 0:
        raise ValueError("logits must be a non-empty one-dimensional array")
    if k <= 0:
        raise ValueError("k must be greater than zero")
    k = min(int(k), x.size)
    indices = np.argpartition(x, -k)[-k:]
    filtered = np.full_like(x, -np.inf)
    filtered[indices] = x[indices]
    return top_p(filtered, p=p, temperature=temperature, rng=rng)


def repetition_penalty(logits, token_ids, penalty=1.1):
    """Return logits adjusted for tokens already generated."""
    if penalty <= 0:
        raise ValueError("penalty must be greater than zero")
    x = np.asarray(logits, dtype=np.float64).copy()
    for token_id in set(int(i) for i in token_ids):
        if token_id < 0 or token_id >= x.size:
            continue
        x[token_id] = x[token_id] / penalty if x[token_id] > 0 else x[token_id] * penalty
    return x
