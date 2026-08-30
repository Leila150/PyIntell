"""Residual connection helpers used by transformer blocks."""

import numpy as np


def residual(x, update, scale=1.0):
    """Add a transformed branch back to its input."""
    x = np.asarray(x)
    update = np.asarray(update)
    if x.shape != update.shape:
        raise ValueError("residual inputs must have identical shapes")
    return x + float(scale) * update


class Residual:
    """Small callable wrapper around a residual connection."""

    def __init__(self, scale=1.0):
        self.scale = float(scale)

    def __call__(self, x, update):
        return residual(x, update, self.scale)
