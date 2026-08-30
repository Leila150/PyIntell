"""High-level inference runtime built from PyIntell's existing model primitives."""

import numpy as np

from .activations import softmax
from .decoding import greedy, top_p, top_k_top_p, repetition_penalty
from .serialization import set_current_model
from .tokenization import encode, decode


class InferenceSession:
    """Reusable inference session around an existing PyIntell model.

    This layer deliberately does not implement a second model. It composes the
    model's existing ``forward``/``logits`` methods with PyIntell's tokenization,
    activation, and decoding primitives.
    """

    def __init__(self, model):
        if not hasattr(model, "logits") or not hasattr(model, "forward"):
            raise TypeError("model must provide forward() and logits()")
        self.model = model
        self.rng = getattr(model, "rng", np.random.default_rng())
        self.last_token_ids = []

    def tokenize(self, prompt):
        if not isinstance(prompt, str):
            ids = list(prompt)
        else:
            unknown_token = "<unk>" if "<unk>" in self.model.vocab else None
            ids = encode(prompt, self.model.vocab, unknown_token=unknown_token)
            if any(token_id is None for token_id in ids):
                raise ValueError("prompt contains unknown tokens and vocabulary has no '<unk>' token")
        if not ids:
            raise ValueError("prompt must contain at least one known token")
        return [int(i) for i in ids]

    def next_logits(self, token_ids):
        ids = list(token_ids)
        if not ids:
            raise ValueError("token_ids must not be empty")
        return np.asarray(self.model.logits(ids)[-1], dtype=np.float64)

    def probabilities(self, token_ids):
        """Return the next-token probability distribution."""
        return softmax(self.next_logits(token_ids))

    def predict_next(self, token_ids, strategy="greedy", temperature=1.0,
                     top_k=50, top_p_value=0.9, repetition_penalty_value=1.0):
        """Select one next token using an existing PyIntell decoder."""
        ids = list(token_ids)
        logits = self.next_logits(ids)
        if repetition_penalty_value != 1.0:
            logits = repetition_penalty(logits, ids, penalty=repetition_penalty_value)

        strategy = str(strategy).lower()
        if strategy == "greedy":
            return greedy(logits)
        if strategy in {"top_p", "nucleus"}:
            return top_p(logits, p=top_p_value, temperature=temperature, rng=self.rng)
        if strategy in {"top_k_top_p", "sampling"}:
            return top_k_top_p(logits, k=top_k, p=top_p_value,
                               temperature=temperature, rng=self.rng)
        raise ValueError("strategy must be 'greedy', 'top_p', or 'top_k_top_p'")

    def generate(self, prompt, max_tokens=50, strategy="top_k_top_p",
                 temperature=1.0, top_k=50, top_p_value=0.9,
                 repetition_penalty_value=1.0):
        """Generate text while reusing the model's existing forward pass."""
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, (int, np.integer)):
            raise TypeError("max_tokens must be an integer")
        if int(max_tokens) < 0:
            raise ValueError("max_tokens must not be negative")

        ids = self.tokenize(prompt)
        self.last_token_ids = ids.copy()
        for _ in range(int(max_tokens)):
            context = ids[-int(self.model.context_length):]
            token_id = self.predict_next(
                context,
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                top_p_value=top_p_value,
                repetition_penalty_value=repetition_penalty_value,
            )
            ids.append(token_id)
        self.last_token_ids = ids.copy()
        return decode(ids, self.model.reverse_vocab) if isinstance(prompt, str) else ids

    def stream(self, prompt, max_tokens=50, strategy="top_k_top_p",
               temperature=1.0, top_k=50, top_p_value=0.9,
               repetition_penalty_value=1.0):
        """Yield generated token IDs incrementally."""
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, (int, np.integer)):
            raise TypeError("max_tokens must be an integer")
        ids = self.tokenize(prompt)
        for _ in range(int(max_tokens)):
            context = ids[-int(self.model.context_length):]
            token_id = self.predict_next(
                context,
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                top_p_value=top_p_value,
                repetition_penalty_value=repetition_penalty_value,
            )
            ids.append(token_id)
            self.last_token_ids = ids.copy()
            yield token_id


def run(model):
    """Make an existing model the active PyIntell model and return it."""
    return set_current_model(model)


def inference(model):
    """Create a reusable inference session for an existing model."""
    return InferenceSession(model)
