"""High-level model builder."""

import math

from .model import Model
from .system import ram, storage_info
from .serialization import set_current_model

_SUPPORTED_FOCUS = {
    "intelligence", "natural", "coding", "reasoning", "math", "knowledge",
    "creativity", "conversation", "instruction", "accuracy", "speed", "memory",
    "context", "language", "translation", "summarization", "classification", "roleplay",
}
_SUPPORTED_DTYPES = {"float64", "float32", "float16", "bfloat16", "int8", "int4"}


def _dtype_bytes(dtype):
    return {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}[dtype]


def _estimate_architecture(target, vocab_size, settings):
    layers = max(1, int(settings.get("layers", round((target / 2.0e6) ** 0.45))))
    heads = max(1, int(settings.get("heads", 4 if target < 10_000_000 else 8)))
    embedding_size = max(32, int(settings.get("embedding_size", round((target / max(vocab_size, 1)) ** 0.5 * 8))))
    embedding_size = max(heads, (embedding_size // heads) * heads)
    hidden_size = max(embedding_size, int(settings.get("hidden_size", embedding_size * 4)))
    estimated = vocab_size * embedding_size + layers * (4 * embedding_size**2 + 2 * embedding_size * hidden_size) + embedding_size * vocab_size
    return {"layers": layers, "heads": heads, "embedding_size": embedding_size, "hidden_size": hidden_size, "estimated_parameters": estimated}


def build(vocab, reverse_vocab, dataset, parameters, focus, dtype=None, settings=None):
    """Build a Transformer model and make it the active pymodel model.

    ``parameters`` is the requested approximate parameter count. ``dtype`` is
    optional and defaults to float32. ``settings`` is an optional dictionary.
    The builder checks available RAM and free storage before construction.
    """
    if not isinstance(vocab, dict) or not isinstance(reverse_vocab, dict):
        raise TypeError("vocab and reverse_vocab must be dictionaries")
    if not isinstance(parameters, int) or parameters <= 0:
        raise ValueError("parameters must be a positive integer")
    if settings is None:
        settings = {}
    elif not isinstance(settings, dict):
        raise TypeError("settings must be a dictionary or None")
    else:
        settings = dict(settings)
    dtype = "float32" if dtype is None else str(dtype).lower()
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")
    focuses = [focus] if isinstance(focus, str) else list(focus)
    invalid = [item for item in focuses if item not in _SUPPORTED_FOCUS]
    if invalid:
        raise ValueError(f"unsupported focus values: {invalid}")
    if not focuses:
        raise ValueError("focus must contain at least one focus")

    architecture = _estimate_architecture(parameters, len(vocab), settings)
    weight_bytes = parameters * _dtype_bytes(dtype)
    required_storage = int(math.ceil(weight_bytes * 1.25))
    required_ram = int(math.ceil(weight_bytes * (2.0 if dtype in {"int4", "int8"} else 4.0)))
    available_ram = ram().get("available")
    available_storage = storage_info().get("free")
    if available_ram is not None and required_ram > available_ram:
        raise MemoryError(
            f"requested model may require about {required_ram / 2**30:.2f} GiB RAM, "
            f"but only {available_ram / 2**30:.2f} GiB is available"
        )
    if available_storage is not None and required_storage > available_storage:
        raise OSError(
            f"requested model may require about {required_storage / 2**30:.2f} GiB storage, "
            f"but only {available_storage / 2**30:.2f} GiB is free"
        )

    final_settings = dict(settings)
    final_settings.update({
        "layers": architecture["layers"],
        "heads": architecture["heads"],
        "embedding_size": architecture["embedding_size"],
        "hidden_size": architecture["hidden_size"],
        "context_length": int(settings.get("context_length", 512)),
        "batch_size": int(settings.get("batch_size", 1)),
        "learning_rate": float(settings.get("learning_rate", 3e-4)),
        "epochs": int(settings.get("epochs", 1)),
        "estimated_parameters": architecture["estimated_parameters"],
        "estimated_weight_bytes": int(weight_bytes),
        "estimated_ram_bytes": required_ram,
        "estimated_storage_bytes": required_storage,
    })
    model = Model(vocab, reverse_vocab, dataset, parameters, focuses, dtype, final_settings)
    set_current_model(model)
    return model
