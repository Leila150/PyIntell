"""pymodel: a modular Python framework for building AI models."""

from .tokenization import tokenizer, vocab, reverse_vocab, encode, decode
from .embeddings import embedding, positional_embedding
from .attention import attention, self_attention, multihead_attention, causal_attention
from .layers import linear, feedforward, layer_norm, dropout
from .transformer import transformer_block, transformer
from .loss import loss, cross_entropy, mse
from .optim import optimizer, sgd, adam, adamw
from .training import train, evaluate
from .generation import generate, sample
from .system import system_info, ram, storage_info, memory_info
from .builder import build

__all__ = [
    "tokenizer", "vocab", "reverse_vocab", "encode", "decode",
    "embedding", "positional_embedding",
    "attention", "self_attention", "multihead_attention", "causal_attention",
    "linear", "feedforward", "layer_norm", "dropout",
    "transformer_block", "transformer",
    "loss", "cross_entropy", "mse",
    "optimizer", "sgd", "adam", "adamw",
    "train", "evaluate", "generate", "sample",
    "system_info", "ram", "storage_info", "memory_info", "build",
]

__version__ = "0.1.0"
