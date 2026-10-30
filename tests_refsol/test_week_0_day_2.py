import numpy as np
import pytest

from .week0_base import *


@pytest.mark.parametrize(
    "shapes",
    [
        [(3,), (3,)],
        [(4, 1), (3,)],
        [(2, 1, 5), (7, 1)],
        [(1,), (2, 3, 4)],
        [(), (2, 2)],
        [(8, 1, 6, 1), (7, 1, 5)],
        [(1, 4), (3, 1), (2, 1, 1)],
    ],
)
def test_task_1_broadcast_shape(shapes):
    assert broadcast_shape(*shapes) == np.broadcast_shapes(*shapes)


@pytest.mark.parametrize("shapes", [[(3,), (4,)], [(2, 3), (3, 2)], [(5, 4), (1, 3)]])
def test_task_1_broadcast_shape_rejects(shapes):
    with pytest.raises(ValueError):
        broadcast_shape(*shapes)


@pytest.mark.parametrize("shape", [(5,), (2, 3), (4, 1, 6), (2, 3, 4, 5)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int64])
def test_task_2_contiguous_strides(shape, dtype):
    a = np.zeros(shape, dtype=dtype)
    assert contiguous_strides(shape, a.itemsize) == a.strides


def test_task_2_transpose_strides():
    a = np.zeros((2, 3, 4), dtype=np.float32)
    for axes in [(0, 1, 2), (2, 1, 0), (1, 0, 2), (0, 2, 1)]:
        t = a.transpose(axes)
        assert transpose_strides(a.shape, a.strides, axes) == (t.shape, t.strides)
    with pytest.raises(ValueError):
        transpose_strides(a.shape, a.strides, (0, 0, 1))


def test_task_2_is_contiguous():
    a = np.zeros((2, 3, 4), dtype=np.float32)
    assert is_contiguous(a.shape, a.strides, 4)
    t = a.transpose(0, 2, 1)
    assert not is_contiguous(t.shape, t.strides, 4)
    # Size-1 axes can have any stride.
    b = np.zeros((3, 1, 4), dtype=np.float32)
    assert is_contiguous(b.shape, (16, 999, 4), 4)


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_task_3_split_and_merge_heads(batch):
    rng = np.random.default_rng(0)
    L, H, D = 5, 4, 3
    x = rng.normal(size=(*batch, L, H * D))
    heads = split_heads(x, H)
    assert heads.shape == (*batch, H, L, D)
    # Head h at position l is the h-th slice of length D of x[..., l, :].
    np.testing.assert_array_equal(heads[..., 2, 1, :], x[..., 1, 2 * D : 3 * D])
    np.testing.assert_array_equal(merge_heads(heads), x)


def test_task_3_split_heads_rejects_bad_size():
    with pytest.raises(ValueError):
        split_heads(np.zeros((2, 10)), 3)


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [((3, 4), (4, 5)), ((2, 3, 4), (2, 4, 5)), ((2, 1, 3, 4), (5, 4, 6))],
)
def test_task_4_batched_matmul(a_shape, b_shape):
    rng = np.random.default_rng(1)
    a = rng.normal(size=a_shape)
    b = rng.normal(size=b_shape)
    np.testing.assert_allclose(batched_matmul(a, b), np.matmul(a, b), rtol=1e-10)


def test_task_5_mlx_softmax():
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    x = mx.array(np.random.default_rng(2).normal(size=(3, 8)).astype(np.float32))
    out = mlx_softmax(x * 100, axis=-1)
    np.testing.assert_allclose(
        np.array(out), np.array(mx.softmax(x * 100, axis=-1)), rtol=1e-5, atol=1e-6
    )
