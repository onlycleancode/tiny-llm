"""Week 0 Day 7: inference systems math, quantization, and tiling."""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ModelConfig",
    "QWEN3_0_6B",
    "param_count",
    "kv_cache_bytes",
    "roofline_time",
    "decode_tokens_per_second",
    "quantize_groupwise",
    "dequantize_groupwise",
    "pack_int4",
    "unpack_int4",
    "tiled_matmul",
]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    tie_word_embeddings: bool = True
    qk_norm: bool = True


# Values from Qwen/Qwen3-0.6B config.json.
QWEN3_0_6B = ModelConfig(
    vocab_size=151936,
    hidden_size=1024,
    num_layers=28,
    num_heads=16,
    num_kv_heads=8,
    head_dim=128,
    intermediate_size=3072,
    tie_word_embeddings=True,
    qk_norm=True,
)


# Task 1: parameter count
def param_count(cfg: ModelConfig, include_embeddings: bool = True) -> int:
    h, d = cfg.hidden_size, cfg.head_dim
    q = h * cfg.num_heads * d
    kv = 2 * h * cfg.num_kv_heads * d
    o = cfg.num_heads * d * h
    mlp = 3 * h * cfg.intermediate_size
    norms = 2 * h + (2 * d if cfg.qk_norm else 0)
    total = cfg.num_layers * (q + kv + o + mlp + norms) + h  # + final norm
    if include_embeddings:
        total += cfg.vocab_size * h
        if not cfg.tie_word_embeddings:
            total += cfg.vocab_size * h
    return total


# Task 2: KV cache size
def kv_cache_bytes(
    cfg: ModelConfig, seq_len: int, batch_size: int, bytes_per_element: int
) -> int:
    return (
        2
        * cfg.num_layers
        * cfg.num_kv_heads
        * cfg.head_dim
        * seq_len
        * batch_size
        * bytes_per_element
    )


# Task 3: roofline and decode speed
def roofline_time(
    flops: float, bytes_moved: float, peak_flops: float, bandwidth: float
) -> tuple[float, bool]:
    compute = flops / peak_flops
    memory = bytes_moved / bandwidth
    return max(compute, memory), memory >= compute


def decode_tokens_per_second(
    weight_bytes: float, kv_bytes: float, bandwidth: float
) -> float:
    return bandwidth / (weight_bytes + kv_bytes)


# Task 4: group-wise quantization
def quantize_groupwise(
    w: np.ndarray, group_size: int, bits: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = w.shape
    if cols % group_size != 0:
        raise ValueError(f"{cols} columns do not split into groups of {group_size}")
    groups = w.reshape(rows, cols // group_size, group_size).astype(np.float32)
    lo = groups.min(axis=-1)
    hi = groups.max(axis=-1)
    levels = (1 << bits) - 1
    scales = (hi - lo) / levels
    scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
    q = np.round((groups - lo[..., None]) / scales[..., None])
    q = np.clip(q, 0, levels).astype(np.uint8).reshape(rows, cols)
    return q, scales, lo.astype(np.float32)


def dequantize_groupwise(
    q: np.ndarray, scales: np.ndarray, biases: np.ndarray, group_size: int
) -> np.ndarray:
    rows, cols = q.shape
    groups = q.reshape(rows, cols // group_size, group_size).astype(np.float32)
    w = groups * scales[..., None] + biases[..., None]
    return w.reshape(rows, cols)


def pack_int4(q: np.ndarray) -> np.ndarray:
    rows, cols = q.shape
    if cols % 8 != 0:
        raise ValueError("int4 packing needs a multiple of 8 columns")
    nibbles = q.reshape(rows, cols // 8, 8).astype(np.uint32) & 0xF
    shifts = np.arange(8, dtype=np.uint32) * 4
    return np.bitwise_or.reduce(nibbles << shifts, axis=-1).astype(np.uint32)


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    rows, words = packed.shape
    shifts = np.arange(8, dtype=np.uint32) * 4
    nibbles = (packed[..., None] >> shifts) & 0xF
    return nibbles.reshape(rows, words * 8).astype(np.uint8)


# Task 5: tiling
def tiled_matmul(a: np.ndarray, b: np.ndarray, tile: int) -> np.ndarray:
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"cannot multiply {a.shape} by {b.shape}")
    out = np.zeros((m, n), dtype=np.result_type(a, b, np.float32))
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            acc = np.zeros((min(tile, m - i), min(tile, n - j)), dtype=out.dtype)
            for p in range(0, k, tile):
                acc += a[i : i + tile, p : p + tile] @ b[p : p + tile, j : j + tile]
            out[i : i + tile, j : j + tile] = acc
    return out
