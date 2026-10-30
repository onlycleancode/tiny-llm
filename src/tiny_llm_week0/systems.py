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
    """Total parameters in a Qwen3-style dense decoder."""
    pass


# Task 2: KV cache size
def kv_cache_bytes(
    cfg: ModelConfig, seq_len: int, batch_size: int, bytes_per_element: int
) -> int:
    """Bytes to hold K and V for every layer, KV head, token and request."""
    pass


# Task 3: roofline and decode speed
def roofline_time(
    flops: float, bytes_moved: float, peak_flops: float, bandwidth: float
) -> tuple[float, bool]:
    """Return (time, memory_bound) for a kernel under the roofline model."""
    pass


def decode_tokens_per_second(
    weight_bytes: float, kv_bytes: float, bandwidth: float
) -> float:
    """Upper bound on single-request decode speed when every token reads all weights and KV once."""
    pass


# Task 4: group-wise quantization
def quantize_groupwise(
    w: np.ndarray, group_size: int, bits: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric min/max quantization per group of columns. Return (q uint8, scales, biases)."""
    pass


def dequantize_groupwise(
    q: np.ndarray, scales: np.ndarray, biases: np.ndarray, group_size: int
) -> np.ndarray:
    """w = q * scale + bias, per group."""
    pass


def pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack 8 four-bit values per uint32; element 0 in the lowest 4 bits."""
    pass


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    """Inverse of pack_int4."""
    pass


# Task 5: tiling
def tiled_matmul(a: np.ndarray, b: np.ndarray, tile: int) -> np.ndarray:
    """a @ b computed tile by tile with an accumulator per output tile."""
    pass
