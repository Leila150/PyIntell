"""Generation and sampling utilities."""

import numpy as np


def sample(logits, temperature=1.0, top_k=None, rng=None):
    """Sample one token ID from a logits vector."""
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    scaled = logits / temperature
    if top_k is not None and top_k < scaled.size:
        indices = np.argpartition(scaled, -top_k)[-top_k:]
        filtered = np.full_like(scaled, -np.inf)
        filtered[indices] = scaled[indices]
        scaled = filtered
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    generator = rng or np.random.default_rng()
    return int(generator.choice(len(probabilities), p=probabilities))


def generate(model, prompt, max_tokens=50, temperature=1.0, top_k=None, **kwargs):
    """Generate text through a model's generate() method."""
    if not hasattr(model, "generate"):
        raise TypeError("model must provide a generate() method")
    return model.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_k=top_k, **kwargs)
