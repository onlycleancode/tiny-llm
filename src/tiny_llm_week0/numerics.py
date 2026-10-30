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
    """Multiply 2-D arrays with explicit loops (no @, np.dot or np.matmul). Raise ValueError on a shape mismatch."""
    pass


# Task 2: counting work and traffic
def matmul_flops(m: int, k: int, n: int) -> int:
    """FLOPs for an (m, k) @ (k, n) matmul: one multiply and one add per term."""
    pass


def matmul_bytes(m: int, k: int, n: int, bytes_per_element: int) -> int:
    """Bytes moved if A, B and C each cross memory exactly once."""
    pass


def arithmetic_intensity(m: int, k: int, n: int, bytes_per_element: int) -> float:
    """FLOPs per byte for the matmul."""
    pass


# Task 3: numerically stable softmax
def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(sum(exp(x))) along axis, stable for large inputs; the axis is removed."""
    pass


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along axis."""
    pass


# Task 4: BF16 rounding
def round_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest BF16 value (ties to even); return float32."""
    pass
