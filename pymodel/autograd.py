"""Small numerical automatic-differentiation helpers.

This module intentionally stays lightweight; it provides finite-difference
helpers for experiments rather than a production reverse-mode engine.
"""

import numpy as np


def gradient(function, x, epsilon=1e-5):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        plus = x.copy(); minus = x.copy()
        plus[index] += epsilon; minus[index] -= epsilon
        result[index] = (function(plus) - function(minus)) / (2 * epsilon)
    return result


def compute_gradients(function, x, epsilon=1e-5):
    return gradient(function, x, epsilon)


def numerical_gradient(function, x, epsilon=1e-5):
    return gradient(function, x, epsilon)


def backward(function, x, epsilon=1e-5):
    return gradient(function, x, epsilon)


def requires_grad(x, value=True):
    array = np.asarray(x)
    setattr(array, "requires_grad", value) if hasattr(array, "__dict__") else None
    return array


def detach(x):
    return np.array(x, copy=True)


def no_grad(function):
    return function


def zero_grad(gradients):
    if isinstance(gradients, dict):
        for key, value in gradients.items(): gradients[key] = np.zeros_like(value)
        return gradients
    return np.zeros_like(gradients)
