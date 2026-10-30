# Week 0 Day 6: The Transformer Block

Today you will assemble one decoder block from NumPy pieces: normalization, causal attention, rotary position encoding,
and a gated MLP. Week 1 rebuilds each of these in MLX with multiple heads, grouped-query attention, and real Qwen3
weights; doing it once at small scale first makes those chapters about the model instead of about the math.

```
src/tiny_llm_week0/transformer.py
```

```bash
pdm run test --week 0 --day 6
```

The starter imports `softmax` from Day 1 and `silu` from Day 3, so finish those first.

**📚 Readings**

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Andrej Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [EleutherAI: Rotary Embeddings](https://blog.eleuther.ai/rotary-embeddings/)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (SwiGLU)

## Task 1: LayerNorm and RMSNorm

Implement `layer_norm(x, gamma, beta, eps)` and `rms_norm(x, weight, eps)` over the last axis.

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma + \beta, \qquad
\text{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{D}\sum_i x_i^2 + \epsilon}}\,w
$$

RMSNorm skips the mean subtraction and the bias. Qwen3 uses RMSNorm before attention, before the MLP, and on each query
and key head.

## Task 2: Causal Attention

- `causal_mask(length)`: an additive `(L, L)` mask with `0` on and below the diagonal and `-inf` above it.
- `attention(q, k, v, mask)`: `softmax(q @ k^T / sqrt(D) + mask) @ v` over the last two axes, with any leading batch
  axes.

With the mask, position `i` can only attend to positions `0..i`. The test changes future keys and values and checks that
earlier outputs do not move. That property is what makes a KV cache valid in Week 2: earlier positions never need to be
recomputed.

## Task 3: Positions and Rotations

- `sinusoidal_positions(length, dim, base)`: the original Transformer's table, with `sin(pos * f_i)` in even columns,
  `cos(pos * f_i)` in odd columns, and `f_i = base^(-2i/dim)`.
- `rotate_pairs(x, theta)`: rotate each pair `(x[2i], x[2i+1])` by the angle `theta[..., i]`.
- `rope_angles(positions, dim, base)`: `theta[..., i] = position * base^(-2i/dim)`.

Rotary position embedding (RoPE) applies `rotate_pairs` to queries and keys instead of adding a table to the input. The
test checks the property that makes it useful: the score between a query at position `m` and a key at position `n`
depends only on `m - n`.

This chapter pairs adjacent elements (the "traditional" layout). Qwen3 pairs element `i` with element `i + D/2` instead.
[Week 1 Day 2](./week1-02-positional-encodings.md) implements both layouts; the rotation is the same.

## Task 4: The Decoder Block

- `swiglu_mlp(x, w_gate, w_up, w_down)`: `(silu(x W_gate^T) * (x W_up^T)) W_down^T`.
- `decoder_block(x, p)`: for a single sequence `x` of shape `(L, D)` and one attention head of width `D`:
  1. `h = rms_norm(x, p["attn_norm"])`
  2. project `q`, `k`, `v` with `p["wq"]`, `p["wk"]`, `p["wv"]` (weights are `(out, in)`, as on Day 3)
  3. apply RoPE to `q` and `k` at positions `0..L-1`
  4. `x = x + attention(q, k, v, causal_mask(L)) @ p["wo"].T`
  5. `h = rms_norm(x, p["mlp_norm"])`
  6. return `x + swiglu_mlp(h, p["w_gate"], p["w_up"], p["w_down"])`

This is a Qwen3 decoder layer minus multiple heads, grouped-query attention, and the per-head Q/K norms. Stack 28 of these
between an embedding table and an output projection, and you have the 0.6B model from Week 1.

{{#include copyright.md}}
