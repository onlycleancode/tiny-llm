import numpy as np
import pytest

from .week0_base import *


def test_task_1_layer_norm():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, size=(4, 16))
    out = layer_norm(x, np.ones(16), np.zeros(16))
    np.testing.assert_allclose(out.mean(axis=-1), 0, atol=1e-10)
    np.testing.assert_allclose(out.var(axis=-1), 1, rtol=1e-4)
    gamma, beta = rng.normal(size=16), rng.normal(size=16)
    np.testing.assert_allclose(
        layer_norm(x, gamma, beta), out * gamma + beta, rtol=1e-6, atol=1e-12
    )


def test_task_1_rms_norm():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 8))
    w = rng.normal(size=8)
    expected = x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-5) * w
    np.testing.assert_allclose(rms_norm(x, w), expected, rtol=1e-12)
    # Unlike LayerNorm, RMSNorm does not remove the mean.
    shifted = rms_norm(x + 5, np.ones(8))
    assert np.abs(shifted.mean(axis=-1)).min() > 0.5


def test_task_2_causal_mask():
    m = causal_mask(4)
    assert m.shape == (4, 4)
    assert np.all(m[np.tril_indices(4)] == 0)
    assert np.all(np.isneginf(m[np.triu_indices(4, k=1)]))


def test_task_2_attention_matches_definition():
    rng = np.random.default_rng(2)
    q, k, v = (rng.normal(size=(2, 5, 8)) for _ in range(3))
    s = q @ k.transpose(0, 2, 1) / np.sqrt(8)
    w = np.exp(s - s.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    np.testing.assert_allclose(attention(q, k, v), w @ v, rtol=1e-10)


def test_task_2_causal_attention_ignores_the_future():
    rng = np.random.default_rng(3)
    q, k, v = (rng.normal(size=(6, 4)) for _ in range(3))
    out = attention(q, k, v, causal_mask(6))
    k2, v2 = k.copy(), v.copy()
    k2[4:] += 10
    v2[4:] -= 10
    out2 = attention(q, k2, v2, causal_mask(6))
    np.testing.assert_allclose(out[:4], out2[:4], rtol=1e-12)
    # The first position can only see itself.
    np.testing.assert_allclose(out[0], v[0], rtol=1e-12)


def test_task_3_sinusoidal_positions():
    pe = sinusoidal_positions(10, 8)
    assert pe.shape == (10, 8)
    np.testing.assert_allclose(pe[0, 0::2], 0)
    np.testing.assert_allclose(pe[0, 1::2], 1)
    assert pe[3, 2] == pytest.approx(np.sin(3 / 10000 ** (2 / 8)))
    assert pe[3, 3] == pytest.approx(np.cos(3 / 10000 ** (2 / 8)))


def test_task_3_rotate_pairs():
    x = np.array([1.0, 0.0, 0.0, 2.0])
    out = rotate_pairs(x, np.array([np.pi / 2, np.pi / 2]))
    np.testing.assert_allclose(out, [0.0, 1.0, -2.0, 0.0], atol=1e-12)
    rng = np.random.default_rng(4)
    y = rng.normal(size=(3, 8))
    theta = rng.normal(size=(3, 4))
    # Rotations preserve length.
    np.testing.assert_allclose(
        np.linalg.norm(rotate_pairs(y, theta), axis=-1), np.linalg.norm(y, axis=-1)
    )


def test_task_3_rope_scores_depend_only_on_relative_position():
    rng = np.random.default_rng(5)
    q, k = rng.normal(size=16), rng.normal(size=16)

    def score(m, n):
        qm = rotate_pairs(q, rope_angles(np.array(m), 16))
        kn = rotate_pairs(k, rope_angles(np.array(n), 16))
        return qm @ kn

    assert rope_angles(np.arange(3), 16).shape == (3, 8)
    assert score(7, 3) == pytest.approx(score(104, 100))
    assert score(2, 2) == pytest.approx(q @ k)


def make_block_params(dim, hidden, seed=6):
    rng = np.random.default_rng(seed)
    s = 1 / np.sqrt(dim)
    return {
        "attn_norm": rng.uniform(0.5, 1.5, dim),
        "wq": rng.normal(0, s, (dim, dim)),
        "wk": rng.normal(0, s, (dim, dim)),
        "wv": rng.normal(0, s, (dim, dim)),
        "wo": rng.normal(0, s, (dim, dim)),
        "mlp_norm": rng.uniform(0.5, 1.5, dim),
        "w_gate": rng.normal(0, s, (hidden, dim)),
        "w_up": rng.normal(0, s, (hidden, dim)),
        "w_down": rng.normal(0, 1 / np.sqrt(hidden), (dim, hidden)),
    }


def test_task_4_swiglu_mlp():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 4))
    g, u, d = rng.normal(size=(6, 4)), rng.normal(size=(6, 4)), rng.normal(size=(4, 6))
    a = x @ g.T
    expected = (a / (1 + np.exp(-a)) * (x @ u.T)) @ d.T
    np.testing.assert_allclose(swiglu_mlp(x, g, u, d), expected, rtol=1e-10)


def test_task_4_decoder_block_matches_reference_math():
    dim, hidden, length = 8, 16, 5
    p = make_block_params(dim, hidden)
    x = np.random.default_rng(8).normal(size=(length, dim))

    def rms(t, w):
        return t / np.sqrt(np.mean(t**2, -1, keepdims=True) + 1e-5) * w

    def rope(t):
        freqs = 10000 ** (-np.arange(0, dim, 2) / dim)
        ang = np.arange(length)[:, None] * freqs
        out = t.copy()
        out[:, 0::2] = t[:, 0::2] * np.cos(ang) - t[:, 1::2] * np.sin(ang)
        out[:, 1::2] = t[:, 0::2] * np.sin(ang) + t[:, 1::2] * np.cos(ang)
        return out

    h = rms(x, p["attn_norm"])
    q, k, v = rope(h @ p["wq"].T), rope(h @ p["wk"].T), h @ p["wv"].T
    s = q @ k.T / np.sqrt(dim) + np.triu(np.full((length, length), -np.inf), 1)
    w = np.exp(s - s.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    x1 = x + (w @ v) @ p["wo"].T
    h = rms(x1, p["mlp_norm"])
    a = h @ p["w_gate"].T
    expected = x1 + ((a / (1 + np.exp(-a))) * (h @ p["w_up"].T)) @ p["w_down"].T

    np.testing.assert_allclose(decoder_block(x, p), expected, rtol=1e-10, atol=1e-12)


def test_task_4_decoder_block_is_causal():
    dim, hidden, length = 8, 16, 6
    p = make_block_params(dim, hidden)
    x = np.random.default_rng(9).normal(size=(length, dim))
    y = x.copy()
    y[3:] += 1.0
    np.testing.assert_allclose(
        decoder_block(x, p)[:3], decoder_block(y, p)[:3], rtol=1e-12
    )
