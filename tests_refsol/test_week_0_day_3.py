import numpy as np
import pytest

from .week0_base import *


def test_task_1_activations():
    x = np.linspace(-6, 6, 101)
    np.testing.assert_allclose(relu(x), np.where(x > 0, x, 0))
    np.testing.assert_allclose(silu(x), x / (1 + np.exp(-x)), rtol=1e-12)
    # tanh-approximate GELU, as used by GPT-2.
    expected = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    np.testing.assert_allclose(gelu_tanh(x), expected, rtol=1e-12, atol=1e-15)


def test_task_1_silu_grad_matches_finite_difference():
    x = np.linspace(-5, 5, 41)
    eps = 1e-6
    fd = (silu(x + eps) - silu(x - eps)) / (2 * eps)
    np.testing.assert_allclose(silu_grad(x), fd, rtol=1e-6, atol=1e-8)


def test_task_2_numerical_gradient():
    x = np.array([[1.0, -2.0], [0.5, 3.0]])
    grad = numerical_gradient(lambda t: float(np.sum(t**3)), x)
    np.testing.assert_allclose(grad, 3 * x**2, rtol=1e-6)
    # The helper must not change its input.
    np.testing.assert_array_equal(x, [[1.0, -2.0], [0.5, 3.0]])


def test_task_2_linear_forward():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 4))
    w = rng.normal(size=(5, 4))  # (out_features, in_features), like the course
    b = rng.normal(size=(5,))
    np.testing.assert_allclose(
        linear_forward(x, w, b), np.einsum("bli,oi->blo", x, w) + b
    )
    np.testing.assert_allclose(linear_forward(x, w), np.einsum("bli,oi->blo", x, w))


def test_task_2_linear_backward_matches_numerical_gradient():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 4))
    w = rng.normal(size=(2, 4))
    b = rng.normal(size=(2,))
    dy = rng.normal(size=(3, 2))  # upstream gradient of L = sum(dy * y)

    def loss(x_, w_, b_):
        return float(np.sum(dy * linear_forward(x_, w_, b_)))

    dx, dw, db = linear_backward(dy, x, w)
    np.testing.assert_allclose(
        dx, numerical_gradient(lambda t: loss(t, w, b), x), rtol=1e-5
    )
    np.testing.assert_allclose(
        dw, numerical_gradient(lambda t: loss(x, t, b), w), rtol=1e-5
    )
    np.testing.assert_allclose(
        db, numerical_gradient(lambda t: loss(x, w, t), b), rtol=1e-5
    )


def test_task_3_softmax_cross_entropy():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(6, 5))
    targets = rng.integers(0, 5, size=6)

    def ce(z):
        z = z - z.max(axis=-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        return float(-logp[np.arange(6), targets].mean())

    loss, dlogits = softmax_cross_entropy(logits, targets)
    assert loss == pytest.approx(ce(logits))
    np.testing.assert_allclose(
        dlogits, numerical_gradient(ce, logits), rtol=1e-5, atol=1e-8
    )
    # Stable for large logits.
    big_loss, big_grad = softmax_cross_entropy(logits * 1000, targets)
    assert np.isfinite(big_loss) and np.all(np.isfinite(big_grad))


def make_spirals(n_per_class: int = 100, classes: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.normal(
            0, 0.2, n_per_class
        )
        xs.append(np.stack([r * np.sin(t), r * np.cos(t)], axis=1))
        ys.append(np.full(n_per_class, c))
    return np.concatenate(xs), np.concatenate(ys)


def test_task_4_mlp_train_step_reduces_loss():
    x, y = make_spirals()
    params = init_mlp(2, 64, 3, seed=0)
    first = mlp_train_step(params, x, y, lr=0.5)
    for _ in range(50):
        last = mlp_train_step(params, x, y, lr=0.5)
    assert last < first


def test_task_4_train_mlp_on_spirals():
    x, y = make_spirals()
    params, losses = train_mlp(
        x, y, hidden=100, num_classes=3, lr=1.0, epochs=2000, seed=0
    )
    assert len(losses) == 2000
    accuracy = (mlp_forward(params, x).argmax(axis=-1) == y).mean()
    assert accuracy >= 0.9, f"train accuracy {accuracy:.2f}"
