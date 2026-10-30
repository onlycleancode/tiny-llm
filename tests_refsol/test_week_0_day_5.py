import numpy as np
import pytest

from .week0_base import *

TEXT = (
    "the cat sat on the mat. the dog sat on the log. "
    "a cat and a dog met on the mat and sat together. "
) * 4


def test_task_1_pair_counts_and_merge():
    ids = list(b"aaabdaaabac")
    counts = pair_counts(ids)
    assert counts[(97, 97)] == 4
    assert counts[(97, 98)] == 2
    assert merge_pair(ids, (97, 97), 256) == [256, 97, 98, 100, 256, 97, 98, 97, 99]
    assert merge_pair([1, 1, 1, 1], (1, 1), 9) == [9, 9]


def test_task_1_bpe_train_known_example():
    merges = bpe_train("aaabdaaabac", 3)
    # aa -> 256, then (256, a) and (a, b) tie at 2; the smaller pair wins.
    assert merges[0] == (97, 97)
    assert merges[1] == (97, 98)
    assert merges[2] == (256, 257)
    assert bpe_encode("aaabdaaabac", merges) == [258, 100, 258, 97, 99]


def test_task_1_bpe_round_trip_and_compression():
    merges = bpe_train(TEXT, 40)
    assert len(merges) == 40
    ids = bpe_encode(TEXT, merges)
    assert bpe_decode(ids, merges) == TEXT
    assert len(ids) < len(TEXT.encode()) / 2
    for unseen in ["the bird sat", "héllo wörld ✓", "", "z"]:
        assert bpe_decode(bpe_encode(unseen, merges), merges) == unseen


def test_task_1_bpe_stops_when_nothing_to_merge():
    assert bpe_train("ab", 10) == [(97, 98)]
    assert bpe_train("", 10) == []


def test_task_2_bigram_counts_and_logprobs():
    ids = [0, 1, 0, 1, 2]
    counts = bigram_counts(ids, 3)
    np.testing.assert_array_equal(counts, [[0, 2, 0], [1, 0, 1], [0, 0, 0]])
    lp = bigram_logprobs(counts, alpha=1.0)
    np.testing.assert_allclose(np.exp(lp).sum(axis=-1), np.ones(3))
    assert lp[0, 1] == pytest.approx(np.log(3 / 5))
    assert lp[2, 0] == pytest.approx(np.log(1 / 3))


def test_task_2_perplexity():
    uniform = np.log(np.full((4, 4), 0.25))
    assert perplexity(uniform, [0, 1, 2, 3, 0]) == pytest.approx(4.0)
    assert sequence_logprob(uniform, [0, 1, 2]) == pytest.approx(2 * np.log(0.25))
    merges = bpe_train(TEXT, 20)
    ids = bpe_encode(TEXT, merges)
    lp = bigram_logprobs(bigram_counts(ids, 256 + 20), alpha=0.1)
    # A trained model beats uniform guessing on its training text.
    assert perplexity(lp, ids) < 256 + 20
    with pytest.raises(ValueError):
        perplexity(uniform, [0])


def test_task_3_sample_token():
    logprobs = np.log(np.array([0.1, 0.6, 0.3]))
    rng = np.random.default_rng(0)
    assert sample_token(logprobs, 0.0, rng) == 1
    draws = [sample_token(logprobs, 1.0, rng) for _ in range(4000)]
    freq = np.bincount(draws, minlength=3) / len(draws)
    np.testing.assert_allclose(freq, [0.1, 0.6, 0.3], atol=0.03)
    # Low temperature sharpens toward the argmax.
    cold = [sample_token(logprobs, 0.1, rng) for _ in range(500)]
    assert np.mean(np.array(cold) == 1) > 0.99


def test_task_3_generate():
    # Deterministic cycle 0 -> 1 -> 2 -> 0.
    lp = np.log(np.full((3, 3), 1e-9))
    lp[0, 1] = lp[1, 2] = lp[2, 0] = 0.0
    rng = np.random.default_rng(1)
    assert generate(lp, [0], 5, 0.0, rng) == [0, 1, 2, 0, 1, 2]
    assert generate(lp, [2, 0], 2, 0.0, rng) == [2, 0, 1, 2]
