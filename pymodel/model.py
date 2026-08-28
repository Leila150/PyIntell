"""Trainable language-model object returned by pymodel.build."""

import time
import numpy as np

from .embeddings import embedding, positional_embedding
from .transformer import transformer, init_block
from .tokenization import encode, decode
from .generation import sample
from .optim import Optimizer
from .focus import build_focus_config, normalize_focus


class Model:
    """A compact, persistent NumPy Transformer language model.

    Alpha note: this class provides the framework and introspection APIs. Its
    training implementation is intentionally simple and is not production AI
    training yet.
    """
    def __init__(self, vocab, reverse_vocab, dataset, parameters, focus, dtype, settings):
        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        self.dataset = dataset
        self.requested_parameters = int(parameters)
        self.parameters = self.requested_parameters
        self.focus = tuple(normalize_focus(focus))
        self.requested_dtype = str(dtype)
        storage_dtype = {"bfloat16": "float16", "int4": "uint8"}.get(self.requested_dtype, self.requested_dtype)
        self.dtype = np.dtype(storage_dtype)
        self.settings = dict(settings)
        self.focus_config = dict(self.settings.get("focus_config", build_focus_config(self.focus)))
        self.focus_priorities = dict(self.focus_config.get("priorities", {}))
        self.model_name = str(self.settings.get("model_name", "pymodel_model"))
        self.platform = str(self.settings.get("platform", "auto"))
        self.device = self.settings.get("device")
        self.layers = int(self.settings["layers"])
        self.heads = int(self.settings["heads"])
        self.embedding_size = int(self.settings["embedding_size"])
        self.context_length = int(self.settings["context_length"])
        self.hidden_size = int(self.settings["hidden_size"])
        seed = self.settings.get("seed", None)
        self.rng = np.random.default_rng(seed)
        self.embedding_weights = embedding(np.arange(len(vocab)), dimensions=self.embedding_size,
                                           vocab_size=len(vocab)).astype(self.dtype)
        self.position_weights = positional_embedding(self.context_length, self.embedding_size).astype(self.dtype)
        self.blocks = [init_block(self.embedding_size, self.heads, self.hidden_size, self.rng)
                       for _ in range(self.layers)]
        self.output_weights = (self.rng.standard_normal((self.embedding_size, len(vocab))) *
                               (1.0 / np.sqrt(max(self.embedding_size, 1)))).astype(np.float32)
        self.output_bias = np.zeros(len(vocab), dtype=np.float32)
        self.training_steps = 0
        self.training_history = []
        self.optimizer = None
        self.created_at = time.time()

    def _named_parameters(self):
        params = {"embedding": self.embedding_weights, "position": self.position_weights,
                  "output": self.output_weights, "output_bias": self.output_bias}
        for i, block in enumerate(self.blocks):
            for name, value in block.items():
                params[f"blocks.{i}.{name}"] = value
        return params

    def named_parameters(self):
        return self._named_parameters()

    def parameter_count(self):
        return int(sum(value.size for value in self._named_parameters().values()))

    def parameters_info(self):
        return {name: {"shape": tuple(value.shape), "dtype": str(value.dtype), "count": int(value.size)}
                for name, value in self._named_parameters().items()}

    def get_focus(self):
        """Return the normalized capabilities selected for this model."""
        return list(self.focus)

    def set_focus(self, focus):
        """Change focus metadata without pretending to retrain the model.

        In 0.1.x this does not modify existing weights. Future training systems
        can use the resulting configuration to implement specialization.
        """
        config = build_focus_config(focus)
        self.focus = tuple(config["focuses"])
        self.focus_config = config
        self.focus_priorities = dict(config["priorities"])
        self.settings["focus_config"] = config
        return self.get_focus()

    def focus_scores(self):
        """Return normalized capability priorities produced by the builder."""
        return dict(self.focus_priorities)

    def focus_info(self):
        """Return focus names and their current alpha priority metadata."""
        ranked = sorted(self.focus_priorities.items(), key=lambda item: (-item[1], item[0]))
        return {
            "focuses": self.get_focus(),
            "priorities": self.focus_scores(),
            "top_capabilities": [name for name, _ in ranked[:5]],
        }

    def forward(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional sequence")
        if len(ids) == 0:
            raise ValueError("token_ids must not be empty")
        if len(ids) > self.context_length:
            raise ValueError("sequence exceeds model context length")
        if ids.min() < 0 or ids.max() >= len(self.vocab):
            raise ValueError("token ID outside vocabulary")
        x = self.embedding_weights[ids].astype(np.float32) + self.position_weights[:len(ids)].astype(np.float32)
        return transformer(x, layers=self.layers, heads=self.heads, hidden_size=self.hidden_size,
                           causal=True, weights=self.blocks)

    def logits(self, token_ids):
        return self.forward(token_ids) @ self.output_weights + self.output_bias

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(np.clip(shifted, -80.0, 80.0))
        return probabilities / np.maximum(np.sum(probabilities, axis=-1, keepdims=True), 1e-12)

    def _examples(self, dataset):
        for sequence in dataset:
            ids = list(sequence)
            if len(ids) < 2:
                continue
            for end in range(1, len(ids)):
                start = max(0, end - self.context_length)
                yield ids[start:end], ids[end]

    def evaluate(self, dataset=None, **kwargs):
        data = self.dataset if dataset is None else dataset
        total_loss = 0.0; correct = 0; count = 0
        for inputs, target in self._examples(data):
            probabilities = self._softmax(self.logits(inputs)[-1])
            target = int(target)
            if target < 0 or target >= len(self.vocab):
                raise ValueError("target token ID outside vocabulary")
            total_loss -= float(np.log(np.clip(probabilities[target], 1e-12, 1.0)))
            correct += int(np.argmax(probabilities) == target)
            count += 1
        if count == 0:
            raise ValueError("dataset must contain sequences with at least two token IDs")
        loss = total_loss / count
        return {"loss": loss, "accuracy": correct / count, "samples": count,
                "perplexity": float(np.exp(min(loss, 20.0)))}

    def train(self, dataset=None, epochs=1, learning_rate=None, optimizer=None, **kwargs):
        """Train the model's currently supported trainable output head."""
        data = self.dataset if dataset is None else dataset
        epochs = int(epochs if epochs is not None else self.settings.get("epochs", 1))
        learning_rate = float(learning_rate if learning_rate is not None else self.settings.get("learning_rate", 3e-4))
        optimizer = optimizer or self.settings.get("optimizer", "adamw")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        examples = list(self._examples(data))
        if not examples:
            raise ValueError("dataset must contain sequences with at least two token IDs")
        trainable = {"output": self.output_weights, "output_bias": self.output_bias}
        if self.optimizer is None or self.optimizer.kind != str(optimizer).lower() or self.optimizer.lr != learning_rate:
            self.optimizer = Optimizer(trainable, kind=optimizer, learning_rate=learning_rate,
                                       weight_decay=float(kwargs.get("weight_decay", self.settings.get("weight_decay", 0.0))),
                                       clip_norm=kwargs.get("clip_norm", self.settings.get("gradient_clip")))
        before = self.evaluate(data)
        history = []
        for _ in range(epochs):
            gradients = {"output": np.zeros_like(self.output_weights), "output_bias": np.zeros_like(self.output_bias)}
            total_loss = 0.0
            for inputs, target in examples:
                hidden = self.forward(inputs)[-1].astype(np.float32)
                probabilities = self._softmax(hidden @ self.output_weights + self.output_bias)
                target = int(target)
                total_loss -= float(np.log(np.clip(probabilities[target], 1e-12, 1.0)))
                probabilities[target] -= 1.0
                gradients["output"] += np.outer(hidden, probabilities)
                gradients["output_bias"] += probabilities
            gradients = {k: v / len(examples) for k, v in gradients.items()}
            self.optimizer.step(gradients)
            self.training_steps += 1
            history.append(total_loss / len(examples))
        after = self.evaluate(data)
        record = {"loss": history[-1], "loss_history": history, "epochs": epochs, "samples": len(examples),
                  "steps": self.training_steps, "before": before, "after": after,
                  "loss_decreased": after["loss"] < before["loss"],
                  "accuracy_improved": after["accuracy"] > before["accuracy"]}
        self.training_history.append(record)
        return record

    def generate(self, prompt, max_tokens=50, temperature=1.0, top_k=None, **kwargs):
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, (int, np.integer)):
            raise TypeError("max_tokens must be an integer")
        max_tokens = int(max_tokens)
        if max_tokens < 0:
            raise ValueError("max_tokens must not be negative")
        if isinstance(prompt, str):
            unknown_token = "<unk>" if "<unk>" in self.vocab else None
            ids = encode(prompt, self.vocab, unknown_token=unknown_token)
            if any(token_id is None for token_id in ids):
                raise ValueError("prompt contains unknown tokens and vocabulary has no '<unk>' token")
        else:
            ids = list(prompt)
        if not ids:
            raise ValueError("prompt must contain at least one known token")
        for _ in range(max_tokens):
            context = ids[-self.context_length:]
            ids.append(sample(self.logits(context)[-1], temperature=temperature, top_k=top_k))
        return decode(ids, self.reverse_vocab) if isinstance(prompt, str) else ids

    def summary(self):
        actual = self.parameter_count()
        return {"model_name": self.model_name, "platform": self.platform, "device": self.device,
                "parameters": actual, "requested_parameters": self.requested_parameters,
                "focus": list(self.focus), "focus_scores": self.focus_scores(),
                "dtype": self.requested_dtype, "layers": self.layers,
                "heads": self.heads, "embedding_size": self.embedding_size, "hidden_size": self.hidden_size,
                "context_length": self.context_length, "vocab_size": len(self.vocab),
                "training_steps": self.training_steps, "training_history_length": len(self.training_history),
                "settings": dict(self.settings)}
