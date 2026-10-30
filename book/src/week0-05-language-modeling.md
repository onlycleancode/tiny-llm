# Week 0 Day 5: Tokens and Language Models

A language model maps a sequence of token IDs to a probability distribution over the next token. Week 1 uses Qwen3's
tokenizer through `mlx_lm` and spends its time on the network in the middle. Today you will build both ends yourself at
toy scale: a byte-pair-encoding tokenizer, a bigram model with log probabilities and perplexity, and the
autoregressive sampling loop.

```
src/tiny_llm_week0/lm.py
```

```bash
pdm run test --week 0 --day 5
```

**📚 Readings**

- [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) and the
  [minbpe](https://github.com/karpathy/minbpe) repository
- [Andrej Karpathy: The spelled-out intro to language modeling (makemore)](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [Hugging Face LLM Course: Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5)

## Task 1: Byte-Pair Encoding

Start from the UTF-8 bytes of the text, so the first 256 token IDs are the byte values and any string can be encoded.

- `pair_counts(ids)`: count each adjacent pair.
- `merge_pair(ids, pair, new_id)`: replace each occurrence of `pair`, scanning left to right without overlaps, so
  `[1, 1, 1, 1]` becomes `[9, 9]`.
- `bpe_train(text, num_merges)`: repeatedly merge the most frequent pair into a new token. Merge `i` creates token
  `256 + i`. Break ties by choosing the smallest pair, so results are deterministic. Stop early if there is nothing left
  to merge.
- `bpe_encode(text, merges)`: start from bytes and repeatedly apply the earliest-learned merge that appears in the
  sequence, until none applies.
- `bpe_decode(ids, merges)`: build each token's bytes from its two parts and decode as UTF-8.

The known-answer test uses the classic `"aaabdaaabac"` example. The round-trip test includes text that was never seen
during training, including non-ASCII characters; byte-level BPE handles them because every byte is already a token.

## Task 2: A Bigram Model and Perplexity

- `bigram_counts(ids, vocab_size)`: `counts[a, b]` is how often `b` follows `a`.
- `bigram_logprobs(counts, alpha)`: add-`alpha` smoothing, then normalize each row and take logs. Each row is now a
  log-probability distribution over the next token, the same object that Week 1's model produces for every position.
- `sequence_logprob(logprobs, ids)`: the sum of `log P(ids[t+1] | ids[t])`.
- `perplexity(logprobs, ids)`: `exp` of the negative mean log probability per predicted token. A uniform model over `V`
  tokens has perplexity exactly `V`. Raise `ValueError` for sequences with fewer than two tokens.

## Task 3: Sampling and the Decode Loop

- `sample_token(logprobs, temperature, rng)`: return the argmax at temperature 0. Otherwise divide the log
  probabilities by the temperature, renormalize with a stable softmax, and draw with `rng.choice`.
- `generate(logprobs, prompt, max_new_tokens, temperature, rng)`: repeatedly sample the next token from the row of the
  last token and append it.

This loop is the skeleton of Week 1 Day 6. The difference there is that each step runs the whole transformer, and Week 2
then makes each step cheaper with a KV cache. Week 1 Day 7 adds top-k and top-p filtering on top of this temperature
sampler.

{{#include copyright.md}}
