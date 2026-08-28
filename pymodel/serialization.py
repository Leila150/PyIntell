"""Model serialization and named-model management."""

import json
import os
import pickle
from pathlib import Path

_REGISTRY_PATH = Path(os.path.expanduser("~/.pymodel/models.json"))
_CURRENT_MODEL = None


def state_dict(model):
    return {
        name: value.copy()
        for name, value in vars(model).items()
        if hasattr(value, "shape") and hasattr(value, "copy")
    }


def load_state_dict(model, state):
    for name, value in state.items():
        setattr(model, name, value)
    return model


def save_weights(model, path):
    import numpy as np
    np.savez(path, **state_dict(model))


def load_weights(model, path):
    import numpy as np
    with np.load(path) as data:
        return load_state_dict(model, {key: data[key] for key in data.files})


def _read_registry():
    try:
        with _REGISTRY_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_registry(registry):
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _REGISTRY_PATH.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(registry, file, ensure_ascii=False, indent=2)
    os.replace(temporary, _REGISTRY_PATH)


def _validate_model_name(model_name):
    if not isinstance(model_name, str) or not model_name.strip():
        raise TypeError("model_name must be a non-empty string")
    name = model_name.strip()
    if name in {".", ".."} or any(char in name for char in "\\/"):
        raise ValueError("model_name must not contain path separators")
    return name


def save_model(model_name, path=None, model=None):
    """Save a model under a unique name.

    The most recently built/loaded model is used when ``model`` is omitted.
    ``path`` is the directory where ``<model_name>.pymodel`` is stored.
    Names are globally tracked in the user's pymodel registry and cannot be
    silently overwritten.
    """
    global _CURRENT_MODEL
    name = _validate_model_name(model_name)
    if model is None:
        model = _CURRENT_MODEL
    if model is None:
        raise RuntimeError("no active model; build or load a model first")
    if path is None:
        raise TypeError("path is required")

    registry = _read_registry()
    if name in registry:
        raise FileExistsError(f"model name '{name}' already exists")

    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"{name}.pymodel"
    if model_path.exists():
        raise FileExistsError(f"model file already exists: {model_path}")

    with model_path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    registry[name] = {"path": str(model_path.resolve())}
    try:
        _write_registry(registry)
    except Exception:
        try:
            model_path.unlink()
        except OSError:
            pass
        raise

    _CURRENT_MODEL = model
    return str(model_path)


def load_model(model_name):
    """Load a previously named model and make it the active model."""
    global _CURRENT_MODEL
    name = _validate_model_name(model_name)
    registry = _read_registry()
    if name not in registry:
        raise FileNotFoundError(f"model '{name}' was not found")

    model_path = Path(registry[name]["path"]).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"model file for '{name}' no longer exists: {model_path}")

    with model_path.open("rb") as file:
        model = pickle.load(file)
    _CURRENT_MODEL = model
    return model


def get_model(model_name):
    """Return a named model without changing the active model."""
    name = _validate_model_name(model_name)
    registry = _read_registry()
    if name not in registry:
        raise FileNotFoundError(f"model '{name}' was not found")
    model_path = Path(registry[name]["path"]).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"model file for '{name}' no longer exists: {model_path}")
    with model_path.open("rb") as file:
        return pickle.load(file)


def save(model, path):
    """Backward-compatible anonymous model save."""
    with open(path, "wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def load(path):
    """Backward-compatible anonymous model load."""
    with open(path, "rb") as file:
        return pickle.load(file)


def checkpoint(model, path):
    return save(model, path)


def save_checkpoint(model, path):
    return save(model, path)


def load_checkpoint(path):
    return load(path)


def save_vocab(vocabulary, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(vocabulary, file, ensure_ascii=False, indent=2)


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def serialize(value):
    return pickle.dumps(value)


def deserialize(value):
    return pickle.loads(value)


def export_model(model, path):
    return save(model, path)


def import_model(path):
    return load(path)
