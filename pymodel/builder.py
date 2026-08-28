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
_SUPPORTED_PLATFORMS = {"auto", "cpu", "gpu", "cuda", "mps", "rocm", "tpu", "android", "mobile"}


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


def _validate_positive_int(settings, key, default, minimum=1):
    value = int(settings.get(key, default))
    if value < minimum:
        raise ValueError(f"settings['{key}'] must be at least {minimum}")
    return value


def _validate_positive_float(settings, key, default):
    value = float(settings.get(key, default))
    if value <= 0:
        raise ValueError(f"settings['{key}'] must be positive")
    return value


def build(vocab, reverse_vocab, dataset, parameters, focus, dtype=None, settings=None, model_name=None):
    """Build a configurable Transformer model.

    ``parameters`` is the requested approximate parameter count. ``dtype`` is
    optional and defaults to float32. ``settings`` controls model/training
    behavior. ``model_name`` gives the resulting model a persistent human-readable
    name.

    Supported settings include::

        {
            "platform": "auto",          # auto/cpu/gpu/cuda/mps/rocm/tpu/android/mobile
            "device": None,              # optional explicit device identifier
            "seed": None,                # deterministic initialization seed
            "layers": 4,
            "heads": 8,
            "embedding_size": 256,
            "hidden_size": 1024,
            "context_length": 512,
            "batch_size": 1,
            "learning_rate": 3e-4,
            "epochs": 1,
            "dropout": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "gradient_clip": 1.0,
            "shuffle": True,
            "mixed_precision": False,
            "gradient_accumulation": 1,
            "checkpoint_interval": 0,
            "save_checkpoints": True,
            "bos_token": None,
            "eos_token": None,
            "pad_token": None,
        }
    """
    if not isinstance(vocab, dict) or not isinstance(reverse_vocab, dict):
        raise TypeError("vocab and reverse_vocab must be dictionaries")
    if not isinstance(parameters, int) or isinstance(parameters, bool) or parameters <= 0:
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

    if model_name is not None:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string or None")
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("model_name must not be empty")

    platform = str(settings.get("platform", "auto")).lower()
    if platform not in _SUPPORTED_PLATFORMS:
        raise ValueError(f"unsupported platform: {platform}")
    device = settings.get("device")
    seed = settings.get("seed")
    if seed is not None:
        if isinstance(seed, bool):
            raise TypeError("settings['seed'] must be an integer or None")
        seed = int(seed)
    batch_size = _validate_positive_int(settings, "batch_size", 1)
    context_length = _validate_positive_int(settings, "context_length", 512)
    epochs = _validate_positive_int(settings, "epochs", 1)
    gradient_accumulation = _validate_positive_int(settings, "gradient_accumulation", 1)
    learning_rate = _validate_positive_float(settings, "learning_rate", 3e-4)
    weight_decay = float(settings.get("weight_decay", 0.0))
    gradient_clip = float(settings.get("gradient_clip", 1.0))
    dropout = float(settings.get("dropout", 0.0))
    if weight_decay < 0:
        raise ValueError("settings['weight_decay'] must not be negative")
    if gradient_clip < 0:
        raise ValueError("settings['gradient_clip'] must not be negative")
    if not 0 <= dropout < 1:
        raise ValueError("settings['dropout'] must be in [0, 1)")
    optimizer = str(settings.get("optimizer", "adamw")).lower()
    if optimizer not in {"sgd", "adam", "adamw", "rmsprop", "adagrad", "adadelta"}:
        raise ValueError(f"unsupported optimizer: {optimizer}")

    architecture = _estimate_architecture(parameters, len(vocab), settings)
    weight_bytes = parameters * _dtype_bytes(dtype)
    required_storage = int(math.ceil(weight_bytes * 1.25))
    required_ram = int(math.ceil(weight_bytes * (2.0 if dtype in {"int4", "int8"} else 4.0)))
    available_ram = ram().get("available")
    available_storage = storage_info().get("free")
    if available_ram is not None and required_ram > available_ram:
        raise MemoryError(f"requested model may require about {required_ram / 2**30:.2f} GiB RAM, but only {available_ram / 2**30:.2f} GiB is available")
    if available_storage is not None and required_storage > available_storage:
        raise OSError(f"requested model may require about {required_storage / 2**30:.2f} GiB storage, but only {available_storage / 2**30:.2f} GiB is free")

    final_settings = dict(settings)
    final_settings.update({
        "model_name": model_name or settings.get("model_name", "pymodel_model"),
        "platform": platform,
        "device": device,
        "seed": seed,
        "layers": architecture["layers"],
        "heads": architecture["heads"],
        "embedding_size": architecture["embedding_size"],
        "hidden_size": architecture["hidden_size"],
        "context_length": context_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "optimizer": optimizer,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "gradient_accumulation": gradient_accumulation,
        "dropout": dropout,
        "estimated_parameters": architecture["estimated_parameters"],
        "estimated_weight_bytes": int(weight_bytes),
        "estimated_ram_bytes": required_ram,
        "estimated_storage_bytes": required_storage,
    })
    model = Model(vocab, reverse_vocab, dataset, parameters, focuses, dtype, final_settings)
    set_current_model(model)
    return model
