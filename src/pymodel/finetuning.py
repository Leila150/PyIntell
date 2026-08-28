"""Fine-tuning helpers and configuration primitives."""

import numpy as np


def freeze(model): setattr(model, "frozen", True); return model
def unfreeze(model): setattr(model, "frozen", False); return model
def freeze_layers(model, layers=None): setattr(model, "frozen_layers", layers); return model
def unfreeze_layers(model, layers=None): setattr(model, "frozen_layers", []); return model
def trainable(model): return not getattr(model, "frozen", False)
def finetune(model, dataset, **settings): setattr(model, "finetune_settings", dict(settings)); return model

def lora(model, rank=8, alpha=16): setattr(model, "lora", {"rank": rank, "alpha": alpha}); return model
def qlora(model, rank=8, alpha=16): setattr(model, "qlora", {"rank": rank, "alpha": alpha}); return model
def adapter(model, size=64): setattr(model, "adapter_size", size); return model
def prompt_tuning(model, tokens=20): setattr(model, "prompt_tuning_tokens", tokens); return model
def prefix_tuning(model, tokens=20): setattr(model, "prefix_tuning_tokens", tokens); return model

def parameter_efficient_finetuning(model, method="lora", **kwargs):
    methods = {"lora": lora, "qlora": qlora, "adapter": adapter, "prompt_tuning": prompt_tuning, "prefix_tuning": prefix_tuning}
    if method not in methods: raise ValueError(f"unknown fine-tuning method: {method}")
    return methods[method](model, **kwargs)

def gradient_clipping(gradients, max_norm=1.0):
    norm = np.sqrt(sum(np.sum(np.asarray(g) ** 2) for g in gradients.values())) if isinstance(gradients, dict) else np.linalg.norm(gradients)
    scale = min(1.0, max_norm / (norm + 1e-12)); return {k: v*scale for k,v in gradients.items()} if isinstance(gradients, dict) else gradients*scale

def weight_decay(value, rate=0.01): return value * (1-rate)
def ema(previous, current, decay=0.99): return decay * previous + (1-decay) * current
