"""Lightweight optimizer definitions."""


def optimizer(parameters=None, kind="adamw", learning_rate=3e-4):
    """Create an optimizer state description for a model."""
    return {"kind": kind.lower(), "learning_rate": float(learning_rate), "parameters": parameters}


def sgd(parameters=None, learning_rate=1e-2):
    return optimizer(parameters, "sgd", learning_rate)


def adam(parameters=None, learning_rate=3e-4):
    return optimizer(parameters, "adam", learning_rate)


def adamw(parameters=None, learning_rate=3e-4):
    return optimizer(parameters, "adamw", learning_rate)
