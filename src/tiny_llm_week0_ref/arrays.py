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
    ndim = max((len(s) for s in shapes), default=0)
    out = []
    for axis in range(-ndim, 0):
        size = 1
        for shape in shapes:
            if -axis > len(shape):
                continue
            dim = shape[axis]
            if dim == 1:
                continue
            if size == 1:
                size = dim
            elif size != dim:
                raise ValueError(f"shapes {shapes} are not broadcastable")
        out.append(size)
    return tuple(out)


# Task 2: strides
def contiguous_strides(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    strides = []
    step = itemsize
    for dim in reversed(shape):
        strides.append(step)
        step *= dim
    return tuple(reversed(strides))


def transpose_strides(
    shape: tuple[int, ...], strides: tuple[int, ...], axes: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if sorted(axes) != list(range(len(shape))):
        raise ValueError(f"{axes} is not a permutation of {len(shape)} axes")
    return tuple(shape[a] for a in axes), tuple(strides[a] for a in axes)


def is_contiguous(
    shape: tuple[int, ...], strides: tuple[int, ...], itemsize: int
) -> bool:
    expected = contiguous_strides(shape, itemsize)
    return all(dim == 1 or s == e for dim, s, e in zip(shape, strides, expected))


# Task 3: attention heads
def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    *batch, length, hidden = x.shape
    if hidden % num_heads != 0:
        raise ValueError(f"hidden size {hidden} is not divisible by {num_heads}")
    head_dim = hidden // num_heads
    x = x.reshape(*batch, length, num_heads, head_dim)
    return np.swapaxes(x, -3, -2)


def merge_heads(x: np.ndarray) -> np.ndarray:
    *batch, heads, length, head_dim = x.shape
    x = np.swapaxes(x, -3, -2)
    return x.reshape(*batch, length, heads * head_dim)


# Task 4: batched matmul from broadcasting
def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # (..., M, K, 1) * (..., 1, K, N) -> sum over K
    return np.sum(a[..., :, :, None] * b[..., None, :, :], axis=-2)


# Task 5 (needs MLX): softmax in MLX
def mlx_softmax(x, axis: int = -1):
    import mlx.core as mx

    shifted = x - mx.max(x, axis=axis, keepdims=True)
    e = mx.exp(shifted)
    return e / mx.sum(e, axis=axis, keepdims=True)
