"""High-level model builder."""

import math
import numpy as np

from .model import Model
from .system import ram, storage_info

_SUPPORTED_FOCUS = {
    "intelligence", "natural", "coding", "reasoning", "math", "knowledge",
    "creativity", "conversation", "instruction", "accuracy", "speed", "memory",
    "context", "language", "translation", "summarization", "classification",
    "roleplay",
}
_SUPPORTED_DTYPES = {"float64", "float32", "float16", "bfloat16", "int8", "int4"}


def _dtype_bytes(dtype):
    return {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}[dtype]


def _estimate_architecture(target, vocab_size, settings):
    """Choose a compact Transformer shape whose estimated size is near target."""
    layers = int(settings.get("layers", max(1, round((target / 2.0e6) ** 0.45))))
    layers = max(1, layers)
    heads = int(settings.get("heads", 4 if target < 10_000_000 else 8))
    heads = max(1, heads)
    embedding_size = int(settings.get("embedding_size", max(32, round((target / max(vocab_size, 1)) ** 0.5 * 8))))
    embedding_size = max(heads, (embedding_size // heads) * heads)
    hidden_size = int(settings.get("hidden_size", embedding_size * 4))
    # Embedding + two FFN projections + four attention projections per layer + LM head.
    estimated = (
        vocab_size * embedding_size
        + layers * (4 * embedding_size * embedding_size + 2 * embedding_size * hidden_size)
        + embedding_size * vocab_size
    )
    return {
        "layers": layers,
        "heads": heads,
        "embedding_size": embedding_size,
        "hidden_size": hidden_size,
        "estimated_parameters": estimated,
    }


def build(vocab, reverse_vocab, dataset, parameters, focus, dtype=None, settings=None):
    """Build a Transformer model from vocabulary, data, size, focus, and settings.

    ``parameters`` is the requested approximate number of model parameters.
    ``dtype`` is optional; when omitted, float32 is used. ``settings`` is an
    optional dictionary for architecture/training preferences.
    """
    if not isinstance(vocab, dict) or not isinstance(reverse_vocab, dict):
        raise TypeError("vocab and reverse_vocab must be dictionaries")
    if not isinstance(parameters, int) or parameters <= 0:
        raise ValueError("parameters must be a positive integer")
    settings = {} if settings is None else dict(settings)
    if dtype is None:
        dtype = "float32"
    dtype = str(dtype).lower()
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")
    focuses = [focus] if isinstance(focus, str) else list(focus)
    invalid = [item for item in focuses if item not in _SUPPORTED_FOCUS]
    if invalid:
        raise ValueError(f"unsupported focus values: {invalid}")

    architecture = _estimate_architecture(parameters, len(vocab), settings)
    bytes_per_parameter = _dtype_bytes(dtype)
    weight_bytes = parameters * bytes_per_parameter
    required_storage = int(math.ceil(weight_bytes * 1.25))
    required_ram = int(math.ceil(weight_bytes * (2.0 if dtype in {"int4", "int8"} else 4.0)))

    available_ram = ram().get("available")
    available_storage = storage_info().get("free")
    if available_ram is not None and required_ram > available_ram:
        raise MemoryError(
            f"requested model may require about {required_ram / 2**30:.2f} GiB RAM, "
            f"but only {available_ram / 2**30:.2f} GiB is available"
        )
    if required_storage > available_storage:
        raise OSError(
            f"requested model may require about {required_storage / 2**30:.2f} GiB storage, "
            f"but only {available_storage / 2**30:.2f} GiB is free"
        )

    final_settings = {
        "layers": architecture["layers"],
        "heads": architecture["heads"],
        "embedding_size": architecture["embedding_size"],
        "hidden_size": architecture["hidden_size"],
        "context_length": int(settings.get("context_length", 512)),
        "batch_size": int(settings.get("batch_size", 1)),
        "learning_rate": float(settings.get("learning_rate", 3e-4)),
        "epochs": int(settings.get("epochs", 1)),
    }
    return Model(vocab, reverse_vocab, dataset, parameters, focuses, dtype, final_settings)
