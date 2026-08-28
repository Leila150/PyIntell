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

# Focus is a model specialization, not merely a label. These profiles describe
# the capabilities the builder should prioritize when choosing defaults.
_FOCUS_PROFILES = {
    "intelligence": {"reasoning": 1.0, "knowledge": 0.9, "accuracy": 0.9},
    "natural": {"language": 1.0, "conversation": 0.8, "creativity": 0.6},
    "coding": {"reasoning": 1.0, "instruction": 0.9, "accuracy": 0.9},
    "reasoning": {"reasoning": 1.0, "accuracy": 0.9, "math": 0.8},
    "math": {"math": 1.0, "reasoning": 0.9, "accuracy": 0.9},
    "knowledge": {"knowledge": 1.0, "memory": 0.8, "accuracy": 0.8},
    "creativity": {"creativity": 1.0, "language": 0.8, "natural": 0.7},
    "conversation": {"conversation": 1.0, "natural": 0.9, "instruction": 0.7},
    "instruction": {"instruction": 1.0, "accuracy": 0.9, "conversation": 0.6},
    "accuracy": {"accuracy": 1.0, "reasoning": 0.8, "knowledge": 0.7},
    "speed": {"speed": 1.0, "efficiency": 0.9},
    "memory": {"memory": 1.0, "knowledge": 0.8, "context": 0.7},
    "context": {"context": 1.0, "memory": 0.8, "language": 0.6},
    "language": {"language": 1.0, "natural": 0.8, "translation": 0.7},
    "translation": {"translation": 1.0, "language": 0.9, "accuracy": 0.8},
    "summarization": {"summarization": 1.0, "language": 0.8, "accuracy": 0.7},
    "classification": {"classification": 1.0, "accuracy": 0.9, "instruction": 0.6},
    "roleplay": {"roleplay": 1.0, "conversation": 0.9, "creativity": 0.8},
}


def _dtype_bytes(dtype):
    return {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}[dtype]


def _normalize_focus(focus):
    focuses = [focus] if isinstance(focus, str) else list(focus)
    if not focuses:
        raise ValueError("focus must contain at least one focus")
    normalized = []
    for item in focuses:
        if not isinstance(item, str):
            raise TypeError("focus values must be strings")
        value = item.strip().lower()
        if value not in _SUPPORTED_FOCUS:
            raise ValueError(f"unsupported focus values: {[value]}")
        if value not in normalized:
            normalized.append(value)
    return normalized


def _build_focus_config(focuses):
    priorities = {}
    for focus in focuses:
        for capability, weight in _FOCUS_PROFILES[focus].items():
            priorities[capability] = priorities.get(capability, 0.0) + weight
    scale = max(priorities.values(), default=1.0)
    priorities = {key: round(value / scale, 4) for key, value in priorities.items()}
    return {"focuses": list(focuses), "priorities": priorities}


def _estimate_architecture(target, vocab_size, settings, focus_config):
    # Focus influences defaults only; explicit architecture settings always win.
    priorities = focus_config["priorities"]
    reasoning_boost = max(priorities.get("reasoning", 0.0), priorities.get("math", 0.0))
    context_boost = priorities.get("context", 0.0)
    speed_boost = priorities.get("speed", 0.0)
    default_layers = round((target / 2.0e6) ** 0.45)
    default_layers += int(round(reasoning_boost * 2))
    default_layers -= int(round(speed_boost))
    layers = max(1, int(settings.get("layers", default_layers)))

    heads_default = 4 if target < 10_000_000 else 8
    heads_default += int(round(reasoning_boost))
    heads = max(1, int(settings.get("heads", heads_default)))

    embedding_default = round((target / max(vocab_size, 1)) ** 0.5 * 8)
    if context_boost:
        embedding_default += int(round(embedding_default * 0.05 * context_boost))
    embedding_size = max(32, int(settings.get("embedding_size", embedding_default)))
    embedding_size = max(heads, (embedding_size // heads) * heads)

    hidden_multiplier = 4
    if reasoning_boost:
        hidden_multiplier += int(round(reasoning_boost))
    hidden_size = max(embedding_size, int(settings.get("hidden_size", embedding_size * hidden_multiplier)))
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
    """Build a configurable, focus-specialized Transformer model.

    ``focus`` selects the capabilities the model should prioritize. Multiple
    focuses can be supplied and are merged into a normalized priority map.
    Focus changes architecture defaults when explicit architecture settings are
    omitted and is persisted in the model for training/inference tooling.
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

    focuses = _normalize_focus(focus)
    focus_config = _build_focus_config(focuses)

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

    architecture = _estimate_architecture(parameters, len(vocab), settings, focus_config)
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
        "model_name": model_name or settings.get("model_name", "pyintell_model"),
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
        "focus_config": focus_config,
        "estimated_parameters": architecture["estimated_parameters"],
        "estimated_weight_bytes": int(weight_bytes),
        "estimated_ram_bytes": required_ram,
        "estimated_storage_bytes": required_storage,
    })
    model = Model(vocab, reverse_vocab, dataset, parameters, focuses, dtype, final_settings)
    set_current_model(model)
    return model
