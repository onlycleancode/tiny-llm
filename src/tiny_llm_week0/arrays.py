"""Week 0 Day 2: array programming (shapes, broadcasting, strides, heads)."""

import numpy as np

__all__ = [
    "broadcast_shape",
    "contiguous_strides",
    "transpose_strides",
    "is_contiguous",
    "split_heads",
    "merge_heads",
    "batched_matmul",
    "mlx_softmax",
]


# Task 1: broadcasting rules
def broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Result shape of broadcasting the given shapes; ValueError if incompatible. Do not call np.broadcast_shapes."""
    pass


# Task 2: strides
def contiguous_strides(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    """Byte strides of a C-contiguous array with this shape."""
    pass


def transpose_strides(
    shape: tuple[int, ...], strides: tuple[int, ...], axes: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(shape, strides) after permuting axes, with no data movement. ValueError if axes is not a permutation."""
    pass


def is_contiguous(
    shape: tuple[int, ...], strides: tuple[int, ...], itemsize: int
) -> bool:
    """True if the strides describe a C-contiguous layout (size-1 axes may have any stride)."""
    pass


# Task 3: attention heads
def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """(..., L, H*D) -> (..., H, L, D). ValueError if the last dim does not split into num_heads."""
    pass


def merge_heads(x: np.ndarray) -> np.ndarray:
    """(..., H, L, D) -> (..., L, H*D); the inverse of split_heads."""
    pass


# Task 4: batched matmul from broadcasting
def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # (..., M, K, 1) * (..., 1, K, N) -> sum over K
    """np.matmul semantics using broadcasting and sum only (no @, matmul, dot or einsum)."""
    pass


# Task 5 (needs MLX): softmax in MLX
def mlx_softmax(x, axis: int = -1):
    """Stable softmax written with mlx.core ops (import mlx inside the function)."""
    pass
