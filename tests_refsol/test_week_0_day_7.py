import numpy as np
import pytest

from .week0_base import *


def test_task_1_param_count_qwen3_0_6b():
    # Qwen3-0.6B is published as 0.6B parameters, 0.44B without embeddings.
    assert param_count(QWEN3_0_6B) == 596_049_920
    assert param_count(QWEN3_0_6B, include_embeddings=False) == 440_467_456


def test_task_1_param_count_small_config():
    cfg = ModelConfig(
        vocab_size=10,
        hidden_size=4,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        intermediate_size=6,
        tie_word_embeddings=False,
        qk_norm=False,
    )
    per_layer = 4 * 4 + 2 * 4 * 2 + 4 * 4 + 3 * 4 * 6 + 2 * 4
    assert param_count(cfg, include_embeddings=False) == 2 * per_layer + 4
    assert param_count(cfg) == 2 * per_layer + 4 + 2 * 10 * 4


def test_task_2_kv_cache_bytes():
    # 2 (K and V) x 28 layers x 8 KV heads x 128 dims x 2 bytes = 112 KiB per token.
    assert kv_cache_bytes(QWEN3_0_6B, 1, 1, 2) == 114_688
    assert kv_cache_bytes(QWEN3_0_6B, 4096, 8, 2) == 114_688 * 4096 * 8
    gqa_free = ModelConfig(**{**QWEN3_0_6B.__dict__, "num_kv_heads": 16})
    assert kv_cache_bytes(gqa_free, 1, 1, 2) == 2 * kv_cache_bytes(QWEN3_0_6B, 1, 1, 2)


def test_task_3_roofline():
    # 100 GB/s and 10 TFLOP/s: the ridge point is 100 FLOPs per byte.
    t, memory_bound = roofline_time(1e9, 1e9, 10e12, 100e9)
    assert memory_bound and t == pytest.approx(0.01)
    t, memory_bound = roofline_time(1e12, 1e9, 10e12, 100e9)
    assert not memory_bound and t == pytest.approx(0.1)


def test_task_3_decode_tokens_per_second():
    # 0.6B parameters at 4 bits is roughly 0.3 GB of weights.
    tps = decode_tokens_per_second(0.3e9, 0.0, 100e9)
    assert tps == pytest.approx(100e9 / 0.3e9)
    assert decode_tokens_per_second(0.3e9, 0.1e9, 100e9) < tps


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_task_4_quantize_round_trip(bits):
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 128)).astype(np.float32)
    q, scales, biases = quantize_groupwise(w, group_size=32, bits=bits)
    assert q.shape == w.shape and q.dtype == np.uint8
    assert scales.shape == biases.shape == (4, 4)
    assert q.max() <= (1 << bits) - 1
    back = dequantize_groupwise(q, scales, biases, group_size=32)
    err = np.abs(back - w).reshape(4, 4, 32)
    assert np.all(err <= scales[..., None] / 2 + 1e-6)
    # Each group's minimum is exact.
    np.testing.assert_allclose(biases, w.reshape(4, 4, 32).min(-1))


def test_task_4_quantize_constant_group():
    w = np.full((1, 32), 0.25, dtype=np.float32)
    q, scales, biases = quantize_groupwise(w, 32, 4)
    np.testing.assert_allclose(dequantize_groupwise(q, scales, biases, 32), w)
    with pytest.raises(ValueError):
        quantize_groupwise(np.zeros((1, 30), np.float32), 32, 4)


def test_task_4_pack_int4():
    q = np.arange(16, dtype=np.uint8).reshape(1, 16)
    packed = pack_int4(q)
    assert packed.dtype == np.uint32 and packed.shape == (1, 2)
    # Element 0 sits in the lowest 4 bits.
    assert packed[0, 0] == 0x76543210
    assert packed[0, 1] == 0xFEDCBA98
    rng = np.random.default_rng(1)
    r = rng.integers(0, 16, size=(3, 64)).astype(np.uint8)
    np.testing.assert_array_equal(unpack_int4(pack_int4(r)), r)
    with pytest.raises(ValueError):
        pack_int4(np.zeros((1, 6), np.uint8))


@pytest.mark.parametrize("shape", [(8, 8, 8), (5, 7, 3), (16, 9, 12), (1, 33, 2)])
@pytest.mark.parametrize("tile", [1, 4, 8])
def test_task_5_tiled_matmul(shape, tile):
    m, k, n = shape
    rng = np.random.default_rng(2)
    a = rng.normal(size=(m, k))
    b = rng.normal(size=(k, n))
    np.testing.assert_allclose(tiled_matmul(a, b, tile), a @ b, rtol=1e-10, atol=1e-12)
