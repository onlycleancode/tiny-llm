"""Week 0 Day 5: tokens and language modeling."""

import numpy as np

__all__ = [
    "pair_counts",
    "merge_pair",
    "bpe_train",
    "bpe_encode",
    "bpe_decode",
    "bigram_counts",
    "bigram_logprobs",
    "sequence_logprob",
    "perplexity",
    "sample_token",
    "generate",
]


# Task 1: byte-pair encoding
def pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def bpe_train(text: str, num_merges: int) -> list[tuple[int, int]]:
    ids = list(text.encode("utf-8"))
    merges: list[tuple[int, int]] = []
    for i in range(num_merges):
        counts = pair_counts(ids)
        if not counts:
            break
        # Most frequent pair; ties go to the smallest pair.
        pair = min(counts, key=lambda p: (-counts[p], p))
        ids = merge_pair(ids, pair, 256 + i)
        merges.append(pair)
    return merges


def bpe_encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    ranks = {pair: 256 + i for i, pair in enumerate(merges)}
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        candidates = [p for p in pair_counts(ids) if p in ranks]
        if not candidates:
            break
        pair = min(candidates, key=lambda p: ranks[p])
        ids = merge_pair(ids, pair, ranks[pair])
    return ids


def bpe_decode(ids: list[int], merges: list[tuple[int, int]]) -> str:
    vocab = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges):
        vocab[256 + i] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


# Task 2: a bigram language model
def bigram_counts(ids: list[int], vocab_size: int) -> np.ndarray:
    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    for a, b in zip(ids, ids[1:]):
        counts[a, b] += 1
    return counts


def bigram_logprobs(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    smoothed = counts + alpha
    return np.log(smoothed) - np.log(smoothed.sum(axis=-1, keepdims=True))


def sequence_logprob(logprobs: np.ndarray, ids: list[int]) -> float:
    return float(sum(logprobs[a, b] for a, b in zip(ids, ids[1:])))


def perplexity(logprobs: np.ndarray, ids: list[int]) -> float:
    n = len(ids) - 1
    if n <= 0:
        raise ValueError("need at least two tokens")
    return float(np.exp(-sequence_logprob(logprobs, ids) / n))


# Task 3: sampling and the autoregressive loop
def sample_token(
    logprobs: np.ndarray, temperature: float, rng: np.random.Generator
) -> int:
    if temperature == 0:
        return int(np.argmax(logprobs))
    scaled = logprobs / temperature
    scaled = scaled - scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def generate(
    logprobs: np.ndarray,
    prompt: list[int],
    max_new_tokens: int,
    temperature: float,
    rng: np.random.Generator,
) -> list[int]:
    ids = list(prompt)
    for _ in range(max_new_tokens):
        ids.append(sample_token(logprobs[ids[-1]], temperature, rng))
    return ids
