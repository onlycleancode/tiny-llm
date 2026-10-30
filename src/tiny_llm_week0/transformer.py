"""Week 0 Day 6: the transformer block in NumPy."""

import numpy as np

# Reuse your Day 1 and Day 3 work here.
from .nn import silu  # noqa: F401
from .numerics import softmax  # noqa: F401

__all__ = [
    "layer_norm",
    "rms_norm",
    "causal_mask",
    "attention",
    "sinusoidal_positions",
    "rotate_pairs",
    "rope_angles",
    "swiglu_mlp",
    "decoder_block",
]


# Task 1: normalization
def layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Normalize the last axis to zero mean and unit variance, then scale and shift."""
    pass


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """x / sqrt(mean(x**2) + eps) * weight over the last axis."""
    pass


# Task 2: causal attention
def causal_mask(length: int) -> np.ndarray:
    """(L, L) additive mask: 0 on and below the diagonal, -inf above."""
    pass


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """softmax(q @ k^T / sqrt(D) + mask) @ v over the last two axes."""
    pass


# Task 3: positions
def sinusoidal_positions(length: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """(L, D) table: even columns sin(pos * f_i), odd columns cos(pos * f_i), f_i = base ** (-2i / D)."""
    pass


def rotate_pairs(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rotate each pair (x[2i], x[2i+1]) by theta[..., i]."""
    pass


def rope_angles(positions: np.ndarray, dim: int, base: float = 10000.0) -> np.ndarray:
    """theta[..., i] = position * base ** (-2i / dim)."""
    pass


# Task 4: a pre-norm decoder block
def swiglu_mlp(
    x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray
) -> np.ndarray:
    """(silu(x @ w_gate.T) * (x @ w_up.T)) @ w_down.T"""
    pass


def decoder_block(x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    """Pre-norm block: RMSNorm -> single-head causal attention with RoPE on q and k -> residual -> RMSNorm -> SwiGLU -> residual."""
    pass
