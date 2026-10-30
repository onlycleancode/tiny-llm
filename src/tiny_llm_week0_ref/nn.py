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
    return np.maximum(x, 0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def silu_grad(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s * (1 + x * (1 - s))


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))


# Task 2: linear layer and gradient checking
def numerical_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.array(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        plus = f(x)
        x[idx] = old - eps
        minus = f(x)
        x[idx] = old
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def linear_forward(
    x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None
) -> np.ndarray:
    y = x @ w.T
    if b is not None:
        y = y + b
    return y


def linear_backward(
    dy: np.ndarray, x: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = dy @ w
    x2 = x.reshape(-1, x.shape[-1])
    dy2 = dy.reshape(-1, dy.shape[-1])
    dw = dy2.T @ x2
    db = dy2.sum(axis=0)
    return dx, dw, db


# Task 3: softmax cross-entropy
def softmax_cross_entropy(
    logits: np.ndarray, targets: np.ndarray
) -> tuple[float, np.ndarray]:
    n = logits.shape[0]
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    loss = -log_probs[np.arange(n), targets].mean()
    dlogits = np.exp(log_probs)
    dlogits[np.arange(n), targets] -= 1
    return float(loss), dlogits / n


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
    h = relu(linear_forward(x, params["w1"], params["b1"]))
    return linear_forward(h, params["w2"], params["b2"])


def mlp_train_step(
    params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, lr: float
) -> float:
    z1 = linear_forward(x, params["w1"], params["b1"])
    h = relu(z1)
    logits = linear_forward(h, params["w2"], params["b2"])
    loss, dlogits = softmax_cross_entropy(logits, y)
    dh, dw2, db2 = linear_backward(dlogits, h, params["w2"])
    dz1 = dh * (z1 > 0)
    _, dw1, db1 = linear_backward(dz1, x, params["w1"])
    for name, g in (("w1", dw1), ("b1", db1), ("w2", dw2), ("b2", db2)):
        params[name] -= lr * g
    return loss


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int,
    num_classes: int,
    lr: float = 0.5,
    epochs: int = 500,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[float]]:
    params = init_mlp(x.shape[1], hidden, num_classes, seed)
    losses = [mlp_train_step(params, x, y, lr) for _ in range(epochs)]
    return params, losses
