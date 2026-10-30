import numpy as np
import pytest

from .week0_base import *


def test_task_1_matmul_naive_matches_numpy():
    rng = np.random.default_rng(0)
    for m, k, n in [(1, 1, 1), (2, 3, 4), (5, 7, 3), (8, 8, 8)]:
        a = rng.normal(size=(m, k))
        b = rng.normal(size=(k, n))
        np.testing.assert_allclose(matmul_naive(a, b), a @ b, rtol=1e-10, atol=1e-10)


def test_task_1_matmul_naive_rejects_bad_shapes():
    with pytest.raises(ValueError):
        matmul_naive(np.ones((2, 3)), np.ones((2, 3)))


def test_task_2_matmul_flops_and_bytes():
    assert matmul_flops(2, 3, 4) == 48
    assert matmul_bytes(2, 3, 4, 2) == (6 + 12 + 8) * 2
    # One decode token through a 4096 x 4096 BF16 projection: matrix-vector.
    decode = arithmetic_intensity(1, 4096, 4096, 2)
    # A 4096-token prefill through the same projection: matrix-matrix.
    prefill = arithmetic_intensity(4096, 4096, 4096, 2)
    assert decode < 1.01
    assert prefill > 1000
    assert decode == pytest.approx(2 * 4096 * 4096 / ((4096 + 4096 * 4096 + 4096) * 2))


def test_task_3_softmax_matches_definition():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 7))
    expected = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(softmax(x, axis=-1), expected, rtol=1e-12)
    np.testing.assert_allclose(softmax(x, axis=0).sum(axis=0), np.ones(7))


def test_task_3_softmax_is_stable_for_large_inputs():
    x = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
    out = softmax(x, axis=-1)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[0], softmax(np.array([0.0, 1.0, 2.0])), rtol=1e-12)


def test_task_3_logsumexp():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 5))
    np.testing.assert_allclose(
        logsumexp(x, axis=-1), np.log(np.exp(x).sum(axis=-1)), rtol=1e-12
    )
    big = np.array([1000.0, 1000.0])
    assert logsumexp(big, axis=-1) == pytest.approx(1000.0 + np.log(2.0))
    assert logsumexp(x, axis=0).shape == (5,)


def test_task_4_round_to_bf16_exact_values():
    one = 1.0
    ulp = 2.0**-7  # BF16 keeps 7 explicit mantissa bits
    x = np.array(
        [
            one,
            one + ulp,
            one + ulp / 2,  # tie: rounds to even (1.0)
            one + 3 * ulp / 2,  # tie: rounds to even (1 + 2 ulp)
            one + ulp / 4,  # below half: rounds down
            -2.5,
            0.0,
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [one, one + ulp, one, one + 2 * ulp, one, -2.5, 0.0], dtype=np.float32
    )
    out = round_to_bf16(x)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)


def test_task_4_round_to_bf16_relative_error_and_specials():
    rng = np.random.default_rng(3)
    x = (rng.normal(size=10_000) * 100).astype(np.float32)
    out = round_to_bf16(x)
    rel = np.abs(out - x) / np.abs(x)
    assert rel.max() <= 2.0**-8 + 1e-12
    # The low 16 bits of every result are zero: it is representable in BF16.
    assert np.all(out.view(np.uint32) & 0xFFFF == 0)
    specials = round_to_bf16(np.array([np.inf, -np.inf, np.nan], dtype=np.float32))
    assert specials[0] == np.inf and specials[1] == -np.inf and np.isnan(specials[2])
