"""Simple NumPy quantization and compression helpers."""

import numpy as np


def quantize(x, bits=8):
    x = np.asarray(x, dtype=np.float32)
    if bits == 8:
        scale = max(float(np.max(np.abs(x))) / 127.0, 1e-12); return np.round(x / scale).astype(np.int8), scale
    if bits == 4:
        scale = max(float(np.max(np.abs(x))) / 7.0, 1e-12); return np.clip(np.round(x / scale), -8, 7).astype(np.int8), scale
    raise ValueError("bits must be 8 or 4")

def dequantize(values, scale): return np.asarray(values, dtype=np.float32) * scale
def int8(x): return quantize(x, 8)
def int4(x): return quantize(x, 4)
def float16(x): return np.asarray(x, dtype=np.float16)
def bfloat16(x): return np.asarray(x, dtype=np.float32)
def prune(x, threshold=0.0): return np.where(np.abs(x) < threshold, 0, x)
def sparsify(x, threshold=0.0): return prune(x, threshold)
def compress(x): return np.asarray(x).tobytes()
def decompress(data, dtype=np.float32): return np.frombuffer(data, dtype=dtype)
def low_rank(x, rank):
    u, s, vh = np.linalg.svd(np.asarray(x), full_matrices=False); return u[:, :rank], s[:rank], vh[:rank, :]
def factorize(x, rank): return low_rank(x, rank)
