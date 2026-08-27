"""Lightweight optimizer definitions and update helpers."""

import numpy as np


def optimizer(parameters=None, kind="adamw", learning_rate=3e-4, weight_decay=0.0):
    return {"kind": kind.lower(), "learning_rate": float(learning_rate), "weight_decay": float(weight_decay), "parameters": parameters}

def sgd(parameters=None, learning_rate=1e-2, **kwargs): return optimizer(parameters, "sgd", learning_rate, kwargs.get("weight_decay", 0.0))
def adam(parameters=None, learning_rate=3e-4, **kwargs): return optimizer(parameters, "adam", learning_rate, kwargs.get("weight_decay", 0.0))
def adamw(parameters=None, learning_rate=3e-4, **kwargs): return optimizer(parameters, "adamw", learning_rate, kwargs.get("weight_decay", 0.0))
def rmsprop(parameters=None, learning_rate=1e-3, **kwargs): return optimizer(parameters, "rmsprop", learning_rate, kwargs.get("weight_decay", 0.0))
def adagrad(parameters=None, learning_rate=1e-2, **kwargs): return optimizer(parameters, "adagrad", learning_rate, kwargs.get("weight_decay", 0.0))
def adadelta(parameters=None, learning_rate=1.0, **kwargs): return optimizer(parameters, "adadelta", learning_rate, kwargs.get("weight_decay", 0.0))

def update_weights(weights, gradients, learning_rate=3e-4, weight_decay=0.0):
    return np.asarray(weights) - learning_rate * (np.asarray(gradients) + weight_decay * np.asarray(weights))

def step(weights, gradients, learning_rate=3e-4, weight_decay=0.0): return update_weights(weights, gradients, learning_rate, weight_decay)
def zero_grad(gradients): return {k: np.zeros_like(v) for k, v in gradients.items()} if isinstance(gradients, dict) else np.zeros_like(gradients)
def learning_rate(optimizer_state): return float(optimizer_state.get("learning_rate", 0.0))
