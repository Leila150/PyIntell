"""High-level inference runtime built from PyIntell's existing model primitives."""
import numpy as np
from .activations import softmax
from .decoding import greedy, top_p, top_k_top_p, repetition_penalty
from .serialization import set_current_model
from .tokenization import encode, decode

class InferenceSession:
    """Reusable inference session around an existing PyIntell model."""
    def __init__(self, model):
        if not hasattr(model, "logits") or not hasattr(model, "forward"):
            raise TypeError("model must provide forward() and logits()")
        if not getattr(model, "vocab", None) or not getattr(model, "reverse_vocab", None):
            raise ValueError("model must have a non-empty vocabulary")
        self.model = model
        self.rng = getattr(model, "rng", np.random.default_rng())
        self.last_token_ids = []

    def tokenize(self, prompt):
        if isinstance(prompt, str):
            unknown_token = "<unk>" if "<unk>" in self.model.vocab else None
            ids = encode(prompt, self.model.vocab, unknown_token=unknown_token)
            if any(token_id is None for token_id in ids):
                raise ValueError("prompt contains unknown tokens and vocabulary has no '<unk>' token")
        else:
            try: ids = list(prompt)
            except TypeError: raise TypeError("prompt must be text or an iterable of token IDs")
        if not ids: raise ValueError("prompt must contain at least one known token")
        ids = [int(i) for i in ids]
        if any(i < 0 or i >= len(self.model.vocab) for i in ids):
            raise ValueError("prompt contains a token ID outside the model vocabulary")
        return ids

    def next_logits(self, token_ids):
        ids = list(token_ids)
        if not ids: raise ValueError("token_ids must not be empty")
        if len(ids) > int(self.model.context_length): ids = ids[-int(self.model.context_length):]
        return np.asarray(self.model.logits(ids)[-1], dtype=np.float64)

    def probabilities(self, token_ids): return softmax(self.next_logits(token_ids))

    def predict_next(self, token_ids, strategy="greedy", temperature=1.0,
                     top_k=50, top_p_value=0.9, repetition_penalty_value=1.0):
        ids = list(token_ids)
        logits = self.next_logits(ids)
        strategy = str(strategy).lower()
        temperature = float(temperature)
        if strategy != "greedy" and temperature <= 0: raise ValueError("temperature must be > 0")
        if repetition_penalty_value is not None and float(repetition_penalty_value) != 1.0:
            penalty = float(repetition_penalty_value)
            if penalty <= 0: raise ValueError("repetition_penalty_value must be > 0")
            logits = repetition_penalty(logits, ids, penalty=penalty)
        if strategy == "greedy": return int(greedy(logits))
        if strategy in {"top_p", "nucleus"}:
            p = float(top_p_value)
            if not 0 < p <= 1: raise ValueError("top_p_value must be in (0, 1]")
            return int(top_p(logits, p=p, temperature=temperature, rng=self.rng))
        if strategy in {"top_k_top_p", "sampling"}:
            k = int(top_k); p = float(top_p_value)
            if k < 1: raise ValueError("top_k must be at least 1")
            if not 0 < p <= 1: raise ValueError("top_p_value must be in (0, 1]")
            return int(top_k_top_p(logits, k=k, p=p, temperature=temperature, rng=self.rng))
        raise ValueError("strategy must be 'greedy', 'top_p', or 'top_k_top_p'")

    def generate(self, prompt, max_tokens=50, strategy="top_k_top_p", temperature=1.0,
                 top_k=50, top_p_value=0.9, repetition_penalty_value=1.0):
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, (int, np.integer)):
            raise TypeError("max_tokens must be an integer")
        if int(max_tokens) < 0: raise ValueError("max_tokens must not be negative")
        ids = self.tokenize(prompt)
        self.last_token_ids = ids.copy()
        for _ in range(int(max_tokens)):
            ids.append(self.predict_next(ids[-int(self.model.context_length):], strategy, temperature,
                                         top_k, top_p_value, repetition_penalty_value))
        self.last_token_ids = ids.copy()
        return decode(ids, self.model.reverse_vocab) if isinstance(prompt, str) else ids

    def stream(self, prompt, max_tokens=50, strategy="top_k_top_p", temperature=1.0,
               top_k=50, top_p_value=0.9, repetition_penalty_value=1.0):
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, (int, np.integer)):
            raise TypeError("max_tokens must be an integer")
        if int(max_tokens) < 0: raise ValueError("max_tokens must not be negative")
        ids = self.tokenize(prompt)
        self.last_token_ids = ids.copy()
        for _ in range(int(max_tokens)):
            token_id = self.predict_next(ids[-int(self.model.context_length):], strategy, temperature,
                                         top_k, top_p_value, repetition_penalty_value)
            ids.append(token_id); self.last_token_ids = ids.copy(); yield token_id

def run(model):
    """Make an existing model the active PyIntell model and return it."""
    return set_current_model(model)

def inference(model): return InferenceSession(model)
