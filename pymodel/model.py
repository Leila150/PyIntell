"""Model object returned by pymodel.build."""

import numpy as np

from .embeddings import embedding, positional_embedding
from .transformer import transformer


class Model:
    """A compact Transformer language-model container.

    This initial implementation focuses on architecture construction and
    inference primitives. Automatic differentiation and full optimization
    are planned for later releases.
    """

    def __init__(self, vocab, reverse_vocab, dataset, parameters, focus, dtype, settings):
        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        self.dataset = dataset
        self.parameters = int(parameters)
        self.focus = tuple(focus) if isinstance(focus, (list, tuple, set)) else (focus,)
        self.dtype = np.dtype(dtype)
        self.settings = dict(settings)
        self.layers = int(self.settings["layers"])
        self.heads = int(self.settings["heads"])
        self.embedding_size = int(self.settings["embedding_size"])
        self.context_length = int(self.settings["context_length"])
        self.hidden_size = int(self.settings["hidden_size"])
        self.embedding_weights = embedding(
            np.arange(len(vocab)),
            dimensions=self.embedding_size,
            vocab_size=len(vocab),
        ).astype(self.dtype)
        self.position_weights = positional_embedding(
            self.context_length, self.embedding_size
        ).astype(self.dtype)

    def forward(self, token_ids):
        """Run token IDs through embeddings and the Transformer stack."""
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional sequence")
        if len(ids) > self.context_length:
            raise ValueError("sequence exceeds model context length")
        x = self.embedding_weights[ids] + self.position_weights[:len(ids)]
        return transformer(
            x,
            layers=self.layers,
            heads=self.heads,
            hidden_size=self.hidden_size,
            causal=True,
        )

    def train(self, dataset=None, **kwargs):
        """Training placeholder; autograd/optimizer training is not enabled yet."""
        raise NotImplementedError(
            "full automatic-differentiation training is not implemented in pymodel 0.1.0"
        )

    def evaluate(self, dataset=None, **kwargs):
        raise NotImplementedError(
            "full evaluation is not implemented in pymodel 0.1.0"
        )

    def generate(self, prompt, **kwargs):
        raise NotImplementedError(
            "text generation requires a trained output head and tokenizer in pymodel 0.1.0"
        )

    def summary(self):
        return {
            "parameters": self.parameters,
            "focus": list(self.focus),
            "dtype": str(self.dtype),
            "layers": self.layers,
            "heads": self.heads,
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "context_length": self.context_length,
            "vocab_size": len(self.vocab),
        }
