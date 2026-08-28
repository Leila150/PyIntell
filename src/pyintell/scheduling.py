"""Learning-rate and training scheduling helpers."""

import math


def learning_rate(initial, step=0): return float(initial)
def lr_scheduler(initial, kind="constant", total_steps=1, step=0):
    if kind == "constant": return float(initial)
    if kind == "linear": return linear_decay(initial, total_steps, step)
    if kind == "cosine": return cosine_decay(initial, total_steps, step)
    raise ValueError(f"unknown scheduler: {kind}")

def constant_lr(initial, step=0): return float(initial)
def linear_decay(initial, total_steps, step): return max(0.0, initial * (1.0 - min(step, total_steps) / max(total_steps, 1)))
def cosine_decay(initial, total_steps, step): return initial * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / max(total_steps, 1)))
def warmup(initial, warmup_steps, step): return initial * min(1.0, (step + 1) / max(warmup_steps, 1))
