import numpy as np
import pytest

from .week0_base import *


def numeric_grad(f, x, eps=1e-6):
    x = np.array(x, dtype=np.float64)
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        hi = f(x)
        x[idx] = old - eps
        lo = f(x)
        x[idx] = old
        g[idx] = (hi - lo) / (2 * eps)
    return g


@pytest.mark.parametrize(
    "grad_shape,shape",
    [((3,), (3,)), ((2, 3), (3,)), ((2, 3), (1, 3)), ((4, 2, 3), (2, 1)), ((2, 3), ())],
)
def test_task_1_unbroadcast(grad_shape, shape):
    grad = np.arange(np.prod(grad_shape), dtype=np.float64).reshape(grad_shape)
    out = unbroadcast(grad, shape)
    assert out.shape == shape
    # Broadcasting x to grad_shape and summing back must agree with the
    # gradient of sum(grad * broadcast(x)) with respect to x.
    expected = numeric_grad(
        lambda x: float(np.sum(grad * np.broadcast_to(x, grad_shape))),
        np.zeros(shape),
    )
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_task_2_forward_values():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[0.5], [-1.0]])
    np.testing.assert_allclose((a + b).data, a.data + b.data)
    np.testing.assert_allclose((a * b).data, a.data * b.data)
    np.testing.assert_allclose((a - b).data, a.data - b.data)
    np.testing.assert_allclose((a @ b).data, a.data @ b.data)
    np.testing.assert_allclose(a.sum(axis=0).data, a.data.sum(axis=0))
    np.testing.assert_allclose(a.exp().log().data, a.data)
    np.testing.assert_allclose((-a).relu().data, np.zeros((2, 2)))
    np.testing.assert_allclose((2.0 * a + 1.0).data, 2 * a.data + 1)
    np.testing.assert_allclose((1.0 - a).data, 1 - a.data)


def check_grads(build, *arrays):
    tensors = [Tensor(a) for a in arrays]
    build(*tensors).backward()
    for i, t in enumerate(tensors):

        def f(x, i=i):
            args = [Tensor(a) for a in arrays]
            args[i] = Tensor(x)
            return float(build(*args).data)

        np.testing.assert_allclose(
            t.grad, numeric_grad(f, arrays[i]), rtol=1e-5, atol=1e-7
        )


def test_task_2_and_3_elementwise_grads():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 2.0, size=(3, 4))
    b = rng.uniform(0.5, 2.0, size=(3, 4))
    check_grads(lambda x, y: (x * y + x).sum(), a, b)
    check_grads(lambda x, y: (x - y * y).exp().sum(), a, b)
    check_grads(lambda x, y: (x * y).log().sum(), a, b)
    check_grads(lambda x, y: (x - 1.2).relu().sum() + y.sum(), a, b)


def test_task_2_and_3_broadcast_and_matmul_grads():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 3))
    w = rng.normal(size=(3, 4))
    b = rng.normal(size=(4,))
    check_grads(lambda x_, w_, b_: ((x_ @ w_ + b_).relu() * 2.0).sum(), x, w, b)
    check_grads(
        lambda x_, w_, b_: (x_ @ w_).sum(axis=0).exp().sum() + b_.sum(), x, w, b
    )


def test_task_3_reused_node_accumulates():
    x = Tensor(3.0)
    y = x * x + x  # x is used three times
    y.backward()
    assert float(x.grad) == pytest.approx(7.0)


def test_task_3_topological_order():
    a = Tensor(1.0)
    b = a * 2.0
    c = a + b
    d = c * b
    order = d.topological_order()
    pos = {id(t): i for i, t in enumerate(order)}
    assert pos[id(a)] < pos[id(b)] < pos[id(c)] < pos[id(d)]
    assert len({id(t) for t in order}) == len(order)


def test_task_4_project_logistic_regression_learns():
    rng = np.random.default_rng(2)
    xs = rng.normal(size=(200, 2))
    ys = (xs[:, :1] + 2 * xs[:, 1:] > 0).astype(np.float64)
    w = rng.normal(size=(2, 1)) * 0.1
    b = np.zeros((1,))
    for _ in range(300):
        wt, bt = Tensor(w), Tensor(b)
        z = Tensor(xs) @ wt + bt
        p = 1.0 / (1.0 + np.exp(-z.data))  # forward only, for the loss below
        # Binary cross-entropy via softplus: log(1 + exp(z)) - y * z
        loss = ((z.exp() + 1.0).log() - Tensor(ys) * z).sum() * (1.0 / len(xs))
        loss.backward()
        w -= 1.0 * wt.grad
        b -= 1.0 * bt.grad
    accuracy = ((p > 0.5) == ys).mean()
    assert accuracy > 0.95
