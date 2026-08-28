"""Practical NumPy optimizers for pyintell parameters."""

import numpy as np


class Optimizer:
    """Stateful optimizer supporting SGD, Adam, AdamW, RMSProp and Adagrad."""
    def __init__(self, parameters, kind="adamw", learning_rate=3e-4, weight_decay=0.0,
                 beta1=0.9, beta2=0.999, eps=1e-8, momentum=0.0, clip_norm=None):
        self.parameters = parameters
        self.kind = str(kind).lower()
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.momentum, self.clip_norm = float(momentum), clip_norm
        self.step_count = 0
        self.m = {}
        self.v = {}

    def step(self, gradients):
        """Apply one update to a dict of named NumPy arrays."""
        self.step_count += 1
        total = 0.0
        for name, grad in gradients.items():
            if name not in self.parameters:
                continue
            g = np.asarray(grad, dtype=np.float32)
            if self.clip_norm is not None:
                total += float(np.sum(g * g))
        scale = 1.0
        if self.clip_norm is not None:
            norm = np.sqrt(total)
            if norm > float(self.clip_norm):
                scale = float(self.clip_norm) / max(norm, 1e-12)
        for name, grad in gradients.items():
            if name not in self.parameters:
                continue
            p = self.parameters[name]
            g = np.asarray(grad, dtype=np.float32) * scale
            if self.kind == "sgd":
                if self.momentum:
                    self.m[name] = self.momentum * self.m.get(name, np.zeros_like(g)) + g
                    g = self.m[name]
                update = g
            elif self.kind in ("adam", "adamw"):
                m = self.m[name] = self.beta1 * self.m.get(name, np.zeros_like(g)) + (1 - self.beta1) * g
                v = self.v[name] = self.beta2 * self.v.get(name, np.zeros_like(g)) + (1 - self.beta2) * (g * g)
                mh = m / (1 - self.beta1 ** self.step_count)
                vh = v / (1 - self.beta2 ** self.step_count)
                update = mh / (np.sqrt(vh) + self.eps)
                if self.kind == "adamw" and self.weight_decay:
                    p *= (1.0 - self.lr * self.weight_decay)
            elif self.kind == "rmsprop":
                v = self.v[name] = 0.99 * self.v.get(name, np.zeros_like(g)) + 0.01 * (g * g)
                update = g / (np.sqrt(v) + self.eps)
            elif self.kind == "adagrad":
                v = self.v[name] = self.v.get(name, np.zeros_like(g)) + g * g
                update = g / (np.sqrt(v) + self.eps)
            else:
                raise ValueError(f"unknown optimizer: {self.kind}")
            if self.kind != "adamw" and self.weight_decay:
                update = update + self.weight_decay * p
            p[...] = p - self.lr * update
        return self

    def zero_grad(self):
        return {name: np.zeros_like(value, dtype=np.float32) for name, value in self.parameters.items()}

    def state_dict(self):
        return {"kind": self.kind, "learning_rate": self.lr, "weight_decay": self.weight_decay,
                "step": self.step_count, "m": self.m, "v": self.v}


def optimizer(parameters=None, kind="adamw", learning_rate=3e-4, weight_decay=0.0, **kwargs):
    return Optimizer(parameters or {}, kind, learning_rate, weight_decay, **kwargs)


def sgd(parameters=None, learning_rate=1e-2, **kwargs): return optimizer(parameters, "sgd", learning_rate, **kwargs)
def adam(parameters=None, learning_rate=3e-4, **kwargs): return optimizer(parameters, "adam", learning_rate, **kwargs)
def adamw(parameters=None, learning_rate=3e-4, **kwargs): return optimizer(parameters, "adamw", learning_rate, **kwargs)
def rmsprop(parameters=None, learning_rate=1e-3, **kwargs): return optimizer(parameters, "rmsprop", learning_rate, **kwargs)
def adagrad(parameters=None, learning_rate=1e-2, **kwargs): return optimizer(parameters, "adagrad", learning_rate, **kwargs)

def update_weights(weights, gradients, learning_rate=3e-4, weight_decay=0.0):
    return np.asarray(weights) - learning_rate * (np.asarray(gradients) + weight_decay * np.asarray(weights))

def step(weights, gradients, learning_rate=3e-4, weight_decay=0.0):
    return update_weights(weights, gradients, learning_rate, weight_decay)

def zero_grad(gradients):
    return {k: np.zeros_like(v) for k, v in gradients.items()} if isinstance(gradients, dict) else np.zeros_like(gradients)

def learning_rate(optimizer_state):
    return float(optimizer_state.lr if hasattr(optimizer_state, "lr") else optimizer_state.get("learning_rate", 0.0))
