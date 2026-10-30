"""Week 0 Day 6: the transformer block in NumPy."""

import numpy as np

from .nn import silu
from .numerics import softmax

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
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
    return x / rms * weight


# Task 2: causal attention
def causal_mask(length: int) -> np.ndarray:
    mask = np.zeros((length, length))
    mask[np.triu_indices(length, k=1)] = -np.inf
    return mask


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores + mask
    return softmax(scores, axis=-1) @ v


# Task 3: positions
def sinusoidal_positions(length: int, dim: int, base: float = 10000.0) -> np.ndarray:
    pos = np.arange(length)[:, None]
    freqs = base ** (-np.arange(0, dim, 2) / dim)
    out = np.zeros((length, dim))
    out[:, 0::2] = np.sin(pos * freqs)
    out[:, 1::2] = np.cos(pos * freqs)
    return out


def rotate_pairs(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    cos, sin = np.cos(theta), np.sin(theta)
    out = np.empty_like(x, dtype=np.result_type(x, theta))
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def rope_angles(positions: np.ndarray, dim: int, base: float = 10000.0) -> np.ndarray:
    freqs = base ** (-np.arange(0, dim, 2) / dim)
    return np.asarray(positions)[..., None] * freqs


# Task 4: a pre-norm decoder block
def swiglu_mlp(
    x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray
) -> np.ndarray:
    return (silu(x @ w_gate.T) * (x @ w_up.T)) @ w_down.T


def decoder_block(x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    length, dim = x.shape
    h = rms_norm(x, p["attn_norm"])
    q = h @ p["wq"].T
    k = h @ p["wk"].T
    v = h @ p["wv"].T
    theta = rope_angles(np.arange(length), dim)
    q = rotate_pairs(q, theta)
    k = rotate_pairs(k, theta)
    x = x + attention(q, k, v, causal_mask(length)) @ p["wo"].T
    h = rms_norm(x, p["mlp_norm"])
    return x + swiglu_mlp(h, p["w_gate"], p["w_up"], p["w_down"])
