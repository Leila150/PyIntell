"""Model and weight serialization helpers."""

import json
import pickle


def state_dict(model):
    return {name: value.copy() for name, value in vars(model).items() if hasattr(value, "shape") and hasattr(value, "copy")}


def load_state_dict(model, state):
    for name, value in state.items(): setattr(model, name, value)
    return model

def save_weights(model, path):
    import numpy as np
    np.savez(path, **state_dict(model))

def load_weights(model, path):
    import numpy as np
    with np.load(path) as data: return load_state_dict(model, {key: data[key] for key in data.files})

def save_model(model, path):
    with open(path, "wb") as file: pickle.dump(model, file)

def load_model(path):
    with open(path, "rb") as file: return pickle.load(file)

def save(model, path): return save_model(model, path)
def load(path): return load_model(path)
def checkpoint(model, path): return save_model(model, path)
def save_checkpoint(model, path): return save_model(model, path)
def load_checkpoint(path): return load_model(path)

def save_vocab(vocabulary, path):
    with open(path, "w", encoding="utf-8") as file: json.dump(vocabulary, file, ensure_ascii=False, indent=2)

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as file: return json.load(file)

def serialize(value): return pickle.dumps(value)
def deserialize(value): return pickle.loads(value)
def export_model(model, path): return save_model(model, path)
def import_model(path): return load_model(path)
