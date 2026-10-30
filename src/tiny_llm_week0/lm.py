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
    """Count adjacent pairs in ids."""
    pass


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace each non-overlapping occurrence of pair (left to right) with new_id."""
    pass


def bpe_train(text: str, num_merges: int) -> list[tuple[int, int]]:
    """Learn up to num_merges merges on the UTF-8 bytes of text. Merge i creates token 256 + i. Pick the most frequent pair; break ties with the smallest pair."""
    pass


def bpe_encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Encode text by repeatedly applying the earliest-learned merge present."""
    pass


def bpe_decode(ids: list[int], merges: list[tuple[int, int]]) -> str:
    """Turn ids back into a string."""
    pass


# Task 2: a bigram language model
def bigram_counts(ids: list[int], vocab_size: int) -> np.ndarray:
    """counts[a, b] = how often token b follows token a."""
    pass


def bigram_logprobs(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Add-alpha smoothed log P(next | prev); each row sums to 1 in probability space."""
    pass


def sequence_logprob(logprobs: np.ndarray, ids: list[int]) -> float:
    """Sum of log P(ids[t+1] | ids[t])."""
    pass


def perplexity(logprobs: np.ndarray, ids: list[int]) -> float:
    """exp(-mean log-prob per predicted token). ValueError for fewer than two tokens."""
    pass


# Task 3: sampling and the autoregressive loop
def sample_token(
    logprobs: np.ndarray, temperature: float, rng: np.random.Generator
) -> int:
    """Argmax at temperature 0; otherwise sample from softmax(logprobs / temperature)."""
    pass


def generate(
    logprobs: np.ndarray,
    prompt: list[int],
    max_new_tokens: int,
    temperature: float,
    rng: np.random.Generator,
) -> list[int]:
    """Autoregressively extend the prompt by max_new_tokens tokens."""
    pass
