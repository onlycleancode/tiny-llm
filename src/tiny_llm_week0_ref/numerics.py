"""Week 0 Day 1: numerics and linear algebra for inference."""

import numpy as np

__all__ = [
    "matmul_naive",
    "matmul_flops",
    "matmul_bytes",
    "arithmetic_intensity",
    "logsumexp",
    "softmax",
    "round_to_bf16",
]


# Task 1: matrix multiplication from loops
def matmul_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"cannot multiply {a.shape} by {b.shape}")
    m, k = a.shape
    n = b.shape[1]
    out = np.zeros((m, n), dtype=np.result_type(a, b))
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += a[i, p] * b[p, j]
            out[i, j] = acc
    return out


# Task 2: counting work and traffic
def matmul_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def matmul_bytes(m: int, k: int, n: int, bytes_per_element: int) -> int:
    return (m * k + k * n + m * n) * bytes_per_element


def arithmetic_intensity(m: int, k: int, n: int, bytes_per_element: int) -> float:
    return matmul_flops(m, k, n) / matmul_bytes(m, k, n, bytes_per_element)


# Task 3: numerically stable softmax
def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


# Task 4: BF16 rounding
def round_to_bf16(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32).astype(np.uint64)
    # Round to nearest, ties to even, on the 16 bits that BF16 drops.
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) >> 16 << 16
    out = (rounded & 0xFFFFFFFF).astype(np.uint32).view(np.float32)
    # NaN stays NaN (rounding could otherwise carry into the exponent).
    return np.where(np.isnan(x), x, out)
