"""Model object returned by pymodel.build."""

import numpy as np

from .embeddings import embedding, positional_embedding
from .layers import linear
from .transformer import transformer
from .tokenization import encode, decode, tokenizer
from .generation import sample


class Model:
    """Compact Transformer language-model container.

    The 0.1 release provides model construction and forward primitives.
    Automatic differentiation and optimizer-driven training are not yet
    implemented.
    """

    def __init__(self, vocab, reverse_vocab, dataset, parameters, focus, dtype, settings):
        self.vocab = vocab; self.reverse_vocab = reverse_vocab; self.dataset = dataset
        self.parameters = int(parameters)
        self.focus = tuple(focus) if isinstance(focus, (list, tuple, set)) else (focus,)
        self.requested_dtype = str(dtype)
        storage_dtype = {"bfloat16": "float16", "int4": "uint8"}.get(self.requested_dtype, self.requested_dtype)
        self.dtype = np.dtype(storage_dtype); self.settings = dict(settings)
        self.layers = int(self.settings["layers"]); self.heads = int(self.settings["heads"])
        self.embedding_size = int(self.settings["embedding_size"]); self.context_length = int(self.settings["context_length"])
        self.hidden_size = int(self.settings["hidden_size"])
        self.embedding_weights = embedding(np.arange(len(vocab)), dimensions=self.embedding_size, vocab_size=len(vocab)).astype(self.dtype)
        self.position_weights = positional_embedding(self.context_length, self.embedding_size).astype(self.dtype)
        self.output_weights = np.random.randn(self.embedding_size, len(vocab)).astype(np.float32) * 0.02

    def forward(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1: raise ValueError("token_ids must be a one-dimensional sequence")
        if len(ids) > self.context_length: raise ValueError("sequence exceeds model context length")
        if ids.size and (ids.min() < 0 or ids.max() >= len(self.vocab)): raise ValueError("token ID outside vocabulary")
        x = self.embedding_weights[ids] + self.position_weights[:len(ids)]
        return transformer(x, layers=self.layers, heads=self.heads, hidden_size=self.hidden_size, causal=True)

    def logits(self, token_ids):
        return linear(self.forward(token_ids), self.output_weights)

    def train(self, dataset=None, **kwargs):
        raise NotImplementedError("automatic-differentiation training is not implemented in pymodel 0.1.0")

    def evaluate(self, dataset=None, **kwargs):
        raise NotImplementedError("full evaluation is not implemented in pymodel 0.1.0")

    def generate(self, prompt, max_tokens=50, temperature=1.0, top_k=None, **kwargs):
        if isinstance(prompt, str): ids = encode(prompt, self.vocab)
        else: ids = list(prompt)
        if not ids: raise ValueError("prompt must contain at least one known token")
        for _ in range(max_tokens):
            context = ids[-self.context_length:]
            next_id = sample(self.logits(context)[-1], temperature=temperature, top_k=top_k)
            ids.append(next_id)
            if next_id not in self.reverse_vocab: break
        return decode(ids, self.reverse_vocab) if isinstance(prompt, str) else ids

    def summary(self):
        return {"parameters": self.parameters, "focus": list(self.focus), "dtype": self.requested_dtype,
                "layers": self.layers, "heads": self.heads, "embedding_size": self.embedding_size,
                "hidden_size": self.hidden_size, "context_length": self.context_length, "vocab_size": len(self.vocab)}
