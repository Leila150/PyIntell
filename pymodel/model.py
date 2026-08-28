"""Model object returned by pymodel.build."""

import numpy as np

from .embeddings import embedding, positional_embedding
from .layers import linear
from .transformer import transformer
from .tokenization import encode, decode
from .generation import sample


class Model:
    """Compact Transformer language-model container.

    Provides model construction, forward inference, generation, evaluation,
    and lightweight gradient training of the output projection.
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

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / np.sum(probabilities, axis=-1, keepdims=True)

    def _examples(self, dataset):
        for sequence in dataset:
            ids = list(sequence)
            if len(ids) < 2:
                continue
            for end in range(1, len(ids)):
                yield ids[:end], ids[end]

    def evaluate(self, dataset=None, **kwargs):
        """Return average next-token cross-entropy and accuracy."""
        data = self.dataset if dataset is None else dataset
        total_loss = 0.0; correct = 0; count = 0
        for inputs, target in self._examples(data):
            logits = self.logits(inputs)[-1]
            probabilities = self._softmax(logits[None, :])[0]
            target = int(target)
            if target < 0 or target >= len(self.vocab): raise ValueError("target token ID outside vocabulary")
            total_loss -= float(np.log(np.clip(probabilities[target], 1e-12, 1.0)))
            correct += int(np.argmax(probabilities) == target); count += 1
        if count == 0: raise ValueError("dataset must contain sequences with at least two token IDs")
        return {"loss": total_loss / count, "accuracy": correct / count, "samples": count}

    def train(self, dataset=None, epochs=1, learning_rate=1e-2, **kwargs):
        """Train the output projection using exact softmax cross-entropy gradients."""
        data = self.dataset if dataset is None else dataset
        epochs = int(epochs); learning_rate = float(learning_rate)
        if epochs < 1: raise ValueError("epochs must be at least 1")
        if learning_rate <= 0: raise ValueError("learning_rate must be positive")
        examples = list(self._examples(data))
        if not examples: raise ValueError("dataset must contain sequences with at least two token IDs")
        for _, target in examples:
            target = int(target)
            if target < 0 or target >= len(self.vocab): raise ValueError("target token ID outside vocabulary")
        before = self.evaluate(data); history = []
        for _ in range(epochs):
            gradient = np.zeros_like(self.output_weights, dtype=np.float32); total_loss = 0.0
            for inputs, target in examples:
                hidden = self.forward(inputs)[-1].astype(np.float32)
                logits = hidden @ self.output_weights
                probabilities = self._softmax(logits[None, :])[0]
                target = int(target); total_loss -= float(np.log(np.clip(probabilities[target], 1e-12, 1.0)))
                probabilities[target] -= 1.0
                gradient += np.outer(hidden, probabilities).astype(np.float32)
            gradient /= len(examples); self.output_weights -= learning_rate * gradient
            history.append(total_loss / len(examples))
        after = self.evaluate(data)
        return {"loss": history[-1], "loss_history": history, "epochs": epochs, "samples": len(examples),
                "before": before, "after": after, "loss_decreased": after["loss"] < before["loss"],
                "accuracy_improved": after["accuracy"] > before["accuracy"]}

    def generate(self, prompt, max_tokens=50, temperature=1.0, top_k=None, **kwargs):
        """Generate text or token IDs from a prompt.

        Unknown text tokens use ``<unk>`` when that token exists in the
        vocabulary. Otherwise a clear ValueError is raised instead of allowing
        ``None`` token IDs to reach NumPy.
        """
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, (int, np.integer)):
            raise TypeError("max_tokens must be an integer")
        max_tokens = int(max_tokens)
        if max_tokens < 0: raise ValueError("max_tokens must not be negative")
        if isinstance(prompt, str):
            unknown_token = "<unk>" if "<unk>" in self.vocab else None
            ids = encode(prompt, self.vocab, unknown_token=unknown_token)
            if any(token_id is None for token_id in ids):
                unknown = [token for token in prompt.split() if token not in self.vocab]
                raise ValueError(f"unknown token(s) in prompt: {unknown}; add '<unk>' to the vocabulary or use known tokens")
        else:
            ids = list(prompt)
            if any(token_id is None for token_id in ids):
                raise ValueError("prompt token IDs must not contain None")
        if not ids: raise ValueError("prompt must contain at least one known token")
        for _ in range(max_tokens):
            context = ids[-self.context_length:]
            next_id = sample(self.logits(context)[-1], temperature=temperature, top_k=top_k)
            ids.append(next_id)
        return decode(ids, self.reverse_vocab) if isinstance(prompt, str) else ids

    def summary(self):
        return {"parameters": self.parameters, "focus": list(self.focus), "dtype": self.requested_dtype,
                "layers": self.layers, "heads": self.heads, "embedding_size": self.embedding_size,
                "hidden_size": self.hidden_size, "context_length": self.context_length, "vocab_size": len(self.vocab)}