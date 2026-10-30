"""Week 0 Day 3: neural-network basics and backpropagation in NumPy."""

import numpy as np

__all__ = [
    "relu",
    "silu",
    "silu_grad",
    "gelu_tanh",
    "numerical_gradient",
    "linear_forward",
    "linear_backward",
    "softmax_cross_entropy",
    "init_mlp",
    "mlp_forward",
    "mlp_train_step",
    "train_mlp",
]


# Task 1: activations
def relu(x: np.ndarray) -> np.ndarray:
    """max(x, 0)"""
    pass


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def silu(x: np.ndarray) -> np.ndarray:
    """x * sigmoid(x)"""
    pass


def silu_grad(x: np.ndarray) -> np.ndarray:
    """d/dx silu(x)"""
    pass


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """The tanh approximation of GELU."""
    pass


# Task 2: linear layer and gradient checking
def numerical_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar f at x. Do not modify the caller's x."""
    pass


def linear_forward(
    x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None
) -> np.ndarray:
    """y = x @ w.T + b, with w of shape (out_features, in_features)."""
    pass


def linear_backward(
    dy: np.ndarray, x: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given dL/dy, return (dL/dx, dL/dw, dL/db). x may have extra leading dims."""
    pass


# Task 3: softmax cross-entropy
def softmax_cross_entropy(
    logits: np.ndarray, targets: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean cross-entropy over the batch and its gradient with respect to logits."""
    pass


# Task 4 (project): train a two-layer MLP
def init_mlp(
    in_dim: int, hidden: int, out_dim: int, seed: int = 0
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0, np.sqrt(2.0 / in_dim), (hidden, in_dim)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(0, np.sqrt(1.0 / hidden), (out_dim, hidden)),
        "b2": np.zeros(out_dim),
    }


def mlp_forward(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """logits = linear(relu(linear(x, w1, b1)), w2, b2)"""
    pass


def mlp_train_step(
    params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, lr: float
) -> float:
    """One full-batch SGD step: forward, backward by hand, update params in place. Return the loss."""
    pass


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int,
    num_classes: int,
    lr: float = 0.5,
    epochs: int = 500,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[float]]:
    """init_mlp, then `epochs` calls to mlp_train_step. Return (params, losses)."""
    pass
