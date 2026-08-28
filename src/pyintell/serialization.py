"""Model serialization and named-model management."""

import json
import os
import pickle
from pathlib import Path

_REGISTRY_PATH = Path(os.path.expanduser("~/.pyintell/models.json"))
_CURRENT_MODEL = None


def set_current_model(model):
    """Set the model used by save_model when no explicit model is supplied."""
    global _CURRENT_MODEL
    _CURRENT_MODEL = model
    return model


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


def _model_file(directory, name):
    """Return the canonical saved-model path inside a directory."""
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise NotADirectoryError(f"path must be a directory: {directory}")
    return directory / f"{name}.pyintell"


def save_model(model_name, path=None, model=None):
    """Save a model under a unique name.

    ``path`` is the directory where ``<model_name>.pyintell`` is created.
    If ``model`` is omitted, the most recently built/loaded model is used.
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

    model_path = _model_file(path, name)
    if model_path.exists():
        raise FileExistsError(f"model file already exists: {model_path}")

    # Store the name on the model so the active-model state is unambiguous.
    model.model_name = name

    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    try:
        with temporary.open("wb") as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, model_path)

        registry[name] = {"path": str(model_path.resolve())}
        try:
            _write_registry(registry)
        except Exception:
            try:
                model_path.unlink()
            except OSError:
                pass
            raise
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass

    _CURRENT_MODEL = model
    return str(model_path)


def load_model(model_name):
    """Load a named model and make it the active model."""
    global _CURRENT_MODEL
    name = _validate_model_name(model_name)
    registry = _read_registry()
    if name not in registry:
        raise FileNotFoundError(f"model '{name}' was not found")

    model_path = Path(registry[name].get("path", "")).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"model file for '{name}' no longer exists: {model_path}")

    with model_path.open("rb") as file:
        model = pickle.load(file)
    model.model_name = name
    _CURRENT_MODEL = model
    return model


def get_model(model_name):
    """Load a named model without changing the active model."""
    name = _validate_model_name(model_name)
    registry = _read_registry()
    if name not in registry:
        raise FileNotFoundError(f"model '{name}' was not found")
    model_path = Path(registry[name].get("path", "")).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"model file for '{name}' no longer exists: {model_path}")
    with model_path.open("rb") as file:
        model = pickle.load(file)
    model.model_name = name
    return model


def delete_model(model_name):
    """Delete a named model from disk and remove it from the registry."""
    global _CURRENT_MODEL
    name = _validate_model_name(model_name)
    registry = _read_registry()
    if name not in registry:
        raise FileNotFoundError(f"model '{name}' was not found")

    model_path = Path(registry[name].get("path", "")).expanduser()
    if not model_path.is_file():
        del registry[name]
        _write_registry(registry)
        raise FileNotFoundError(f"model file for '{name}' no longer exists: {model_path}")

    model_path.unlink()
    del registry[name]
    _write_registry(registry)

    if _CURRENT_MODEL is not None and getattr(_CURRENT_MODEL, "model_name", None) == name:
        _CURRENT_MODEL = None
    return True


def edit_model(model_name, **changes):
    """Edit supported metadata of a saved model."""
    global _CURRENT_MODEL
    name = _validate_model_name(model_name)
    registry = _read_registry()
    if name not in registry:
        raise FileNotFoundError(f"model '{name}' was not found")
    if not changes:
        raise ValueError("at least one model field must be provided")

    model_path = Path(registry[name].get("path", "")).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"model file for '{name}' no longer exists: {model_path}")

    with model_path.open("rb") as file:
        model = pickle.load(file)

    allowed = {"focus", "parameters", "settings", "model_name"}
    unknown = set(changes) - allowed
    if unknown:
        raise TypeError(f"unsupported model field(s): {', '.join(sorted(unknown))}")

    if "focus" in changes:
        focus = changes["focus"]
        if isinstance(focus, str):
            model.focus = (focus,)
        elif isinstance(focus, (list, tuple, set)) and focus:
            model.focus = tuple(focus)
        else:
            raise TypeError("focus must be a non-empty string, list, tuple, or set")

    if "parameters" in changes:
        parameters = changes["parameters"]
        if isinstance(parameters, bool) or not isinstance(parameters, (int, float)):
            raise TypeError("parameters must be a number")
        if parameters < 0:
            raise ValueError("parameters must not be negative")
        model.parameters = int(parameters)

    if "settings" in changes:
        settings = changes["settings"]
        if not isinstance(settings, dict):
            raise TypeError("settings must be a dictionary")
        protected = {"layers", "heads", "embedding_size", "hidden_size", "context_length"}
        changed_architecture = protected.intersection(settings)
        if changed_architecture:
            raise ValueError("cannot edit architecture settings in place; rebuild the model instead")
        model.settings.update(settings)

    new_name = name
    if "model_name" in changes:
        new_name = _validate_model_name(changes["model_name"])
        if new_name != name and new_name in registry:
            raise FileExistsError(f"model name '{new_name}' already exists")
        new_path = model_path.with_name(f"{new_name}.pymodel")
        if new_path != model_path and new_path.exists():
            raise FileExistsError(f"model file already exists: {new_path}")
    else:
        new_path = model_path

    model.model_name = new_name
    temporary = new_path.with_suffix(new_path.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, new_path)

    if new_name != name:
        del registry[name]
        registry[new_name] = {"path": str(new_path.resolve())}
        if model_path != new_path and model_path.exists():
            model_path.unlink()
    else:
        registry[name] = {"path": str(new_path.resolve())}

    _write_registry(registry)
    _CURRENT_MODEL = model
    return model


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
