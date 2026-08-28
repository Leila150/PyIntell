"""General-purpose tensor, data, metrics, and model utility functions."""

import json
import os
import numpy as np


def tensor(data, dtype=None):
    return np.asarray(data, dtype=dtype)


def zeros(shape, dtype=np.float32):
    return np.zeros(shape, dtype=dtype)


def ones(shape, dtype=np.float32):
    return np.ones(shape, dtype=dtype)


def full(shape, value, dtype=None):
    return np.full(shape, value, dtype=dtype)


def random(shape, dtype=np.float32):
    return np.random.random(shape).astype(dtype)


def randn(shape, dtype=np.float32):
    return np.random.randn(*shape).astype(dtype)


def randint(low, high=None, size=None, dtype=np.int64):
    return np.random.randint(low, high, size=size, dtype=dtype)


def arange(*args, **kwargs):
    return np.arange(*args, **kwargs)


def reshape(x, shape):
    return np.reshape(x, shape)


def flatten(x):
    return np.asarray(x).reshape(-1)


def squeeze(x, axis=None):
    return np.squeeze(x, axis=axis)


def unsqueeze(x, axis):
    return np.expand_dims(x, axis=axis)


def transpose(x, axes=None):
    return np.transpose(x, axes=axes)


def permute(x, axes):
    return np.transpose(x, axes=axes)


def matmul(a, b):
    return np.matmul(a, b)


def dot(a, b):
    return np.dot(a, b)


def sum(x, axis=None, keepdims=False):
    return np.sum(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    return np.mean(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False):
    return np.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    return np.min(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=None):
    return np.argmax(x, axis=axis)


def argmin(x, axis=None):
    return np.argmin(x, axis=axis)


def clip(x, minimum, maximum):
    return np.clip(x, minimum, maximum)


def sqrt(x):
    return np.sqrt(x)


def exp(x):
    return np.exp(x)


def log(x):
    return np.log(x)


def abs(x):
    return np.abs(x)


def power(x, exponent):
    return np.power(x, exponent)


def cat(tensors, axis=0):
    return np.concatenate(tensors, axis=axis)


def stack(tensors, axis=0):
    return np.stack(tensors, axis=axis)


def split(x, sections, axis=0):
    return np.array_split(x, sections, axis=axis)


def chunk(x, chunks, axis=0):
    return np.array_split(x, chunks, axis=axis)


def repeat(x, repeats, axis=None):
    return np.repeat(x, repeats, axis=axis)


def expand(x, shape):
    return np.broadcast_to(x, shape)


def pad(x, padding, constant=0):
    return np.pad(x, padding, constant_values=constant)


def roll(x, shift, axis=None):
    return np.roll(x, shift, axis=axis)


def gather(x, indices, axis=0):
    return np.take(x, indices, axis=axis)


def scatter(x, indices, values, axis=0):
    result = np.array(x, copy=True)
    np.put_along_axis(result, np.asarray(indices), np.asarray(values), axis=axis)
    return result


def where(condition, x, y):
    return np.where(condition, x, y)


def masked_fill(x, mask, value):
    return np.where(mask, x, value)


def einsum(expression, *operands):
    return np.einsum(expression, *operands)


def norm(x, axis=None, keepdims=False):
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def normalize(x, axis=-1, eps=1e-12):
    x = np.asarray(x)
    return x / np.maximum(norm(x, axis=axis, keepdims=True), eps)


def device():
    return "cpu"


def cpu():
    return "cpu"


def gpu():
    return None


def to_device(x, target="cpu"):
    if str(target).lower() != "cpu":
        raise RuntimeError("GPU backends are not included in the NumPy-only release")
    return x


def seed(value):
    np.random.seed(value)


def set_seed(value):
    seed(value)


def random_seed(value=None):
    if value is None:
        value = int.from_bytes(os.urandom(8), "little")
    seed(value)
    return value


def accuracy(predictions, targets):
    p = np.asarray(predictions)
    t = np.asarray(targets)
    if p.ndim > t.ndim:
        p = np.argmax(p, axis=-1)
    return float(np.mean(p == t))


def precision(predictions, targets):
    p, t = np.asarray(predictions).astype(bool), np.asarray(targets).astype(bool)
    tp = np.sum(p & t)
    fp = np.sum(p & ~t)
    return float(tp / (tp + fp)) if tp + fp else 0.0


def recall(predictions, targets):
    p, t = np.asarray(predictions).astype(bool), np.asarray(targets).astype(bool)
    tp = np.sum(p & t)
    fn = np.sum(~p & t)
    return float(tp / (tp + fn)) if tp + fn else 0.0


def f1_score(predictions, targets):
    p = precision(predictions, targets)
    r = recall(predictions, targets)
    return 2 * p * r / (p + r) if p + r else 0.0


def perplexity(loss_value):
    return float(np.exp(loss_value))


def model_size(parameters, dtype="float32"):
    sizes = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}
    return int(np.ceil(int(parameters) * sizes[str(dtype).lower()]))


def count_parameters(model):
    return int(getattr(model, "parameters", 0))


def trainable_parameters(model):
    return count_parameters(model)


def parameter_shapes(model):
    return {name: value.shape for name, value in vars(model).items() if isinstance(value, np.ndarray)}


def memory_usage(model):
    return sum(value.nbytes for value in vars(model).values() if isinstance(value, np.ndarray))


def model_info(model):
    return model.summary() if hasattr(model, "summary") else {"parameters": count_parameters(model)}


def summary(model):
    return model_info(model)


def save_config(config, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_text(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as file:
        return file.read()


def write_text(path, text, encoding="utf-8"):
    with open(path, "w", encoding=encoding) as file:
        file.write(text)


def load_text(path, encoding="utf-8"):
    return read_text(path, encoding)


def save_text(path, text, encoding="utf-8"):
    return write_text(path, text, encoding)


def normalize_text(text):
    return " ".join(str(text).split())


def clean_text(text):
    return normalize_text(text)


def lower_text(text):
    return str(text).lower()


def uppercase(text):
    return str(text).upper()


def split_sentences(text):
    return [part.strip() for part in str(text).replace("!", ".").replace("?", ".").split(".") if part.strip()]


def split_words(text):
    return str(text).split()


def chunk_text(text, size):
    words = split_words(text)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def truncate_text(text, max_words):
    return " ".join(split_words(text)[:max_words])


def pad_sequence(sequence, length, value=0):
    result = list(sequence)[:length]
    return result + [value] * max(0, length - len(result))


def create_mask(length, causal=False):
    if causal:
        return np.tril(np.ones((length, length), dtype=bool))
    return np.ones((length, length), dtype=bool)


def shuffle(data, seed_value=None):
    items = list(data)
    rng = np.random.default_rng(seed_value)
    rng.shuffle(items)
    return items


def split_dataset(data, train_ratio=0.8, validation_ratio=0.1):
    items = list(data)
    n = len(items)
    a = int(n * train_ratio)
    b = a + int(n * validation_ratio)
    return items[:a], items[a:b], items[b:]


def batch(data, batch_size):
    items = list(data)
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def dataset(data):
    return list(data)


def load_dataset(data):
    return dataset(data)


def save_dataset(data, path):
    np.save(path, np.asarray(data, dtype=object), allow_pickle=True)


def cache(data):
    return list(data)


def filter_dataset(data, predicate):
    return [item for item in data if predicate(item)]


def map_dataset(data, function):
    return [function(item) for item in data]


def temperature(logits, value=1.0):
    if value <= 0:
        raise ValueError("temperature must be greater than zero")
    return np.asarray(logits) / value


def top_k(logits, k):
    values = np.asarray(logits)
    if k <= 0:
        raise ValueError("k must be positive")
    indices = np.argpartition(values, -min(k, values.size))[-min(k, values.size):]
    return indices


def top_p(logits, p=0.9):
    values = np.asarray(logits, dtype=np.float64)
    if not 0 < p <= 1:
        raise ValueError("p must be in the range (0, 1]")
    order = np.argsort(values)[::-1]
    shifted = values[order] - np.max(values)
    probs = np.exp(shifted)
    probs /= probs.sum()
    cumulative = np.cumsum(probs)
    return order[cumulative <= p] if np.any(cumulative <= p) else order[:1]


def repetition_penalty(logits, token_ids, penalty=1.1):
    values = np.array(logits, dtype=np.float64, copy=True)
    for token_id in set(token_ids):
        if 0 <= token_id < len(values):
            values[token_id] = values[token_id] / penalty if values[token_id] > 0 else values[token_id] * penalty
    return values


def frequency_penalty(logits, token_ids, penalty=0.0):
    values = np.array(logits, dtype=np.float64, copy=True)
    for token_id in token_ids:
        if 0 <= token_id < len(values):
            values[token_id] -= penalty
    return values


def presence_penalty(logits, token_ids, penalty=0.0):
    values = np.array(logits, dtype=np.float64, copy=True)
    for token_id in set(token_ids):
        if 0 <= token_id < len(values):
            values[token_id] -= penalty
    return values


def stop_at_token(token_id, stop_tokens):
    return token_id in set(stop_tokens)
