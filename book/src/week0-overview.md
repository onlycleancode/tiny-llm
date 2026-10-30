# Week 0: Prerequisites

The [preface](./preface.md#prerequisites) asks for "the basics of deep learning" and some familiarity with PyTorch, and it
points to two university courses. In practice, Weeks 1–3 lean on a specific set of skills: reading array shapes and
broadcasting rules, reasoning about numerical precision, knowing what a linear layer and its gradient are, understanding
how text becomes token IDs and back, recognizing the pieces of a transformer block, and estimating whether a kernel is
limited by compute or by memory bandwidth.

Week 0 builds those skills one day at a time. Each day has a short chapter, a starter module in `src/tiny_llm_week0`,
a reference solution in `src/tiny_llm_week0_ref`, and supplied tests in `tests_refsol/test_week_0_day_N.py`, exactly
like the later weeks.

Unlike Weeks 1–3, Week 0 uses **NumPy only**. It does not need Apple silicon, the MLX extensions, Xcode, or a downloaded
model, so you can work through it on any machine with Python while you set up the rest of the toolchain. One optional
task on Day 2 uses MLX and is skipped when MLX is not installed.

## How to Run Week 0

Install the dependencies as described in [Setting Up the Environment](./setup.md), then run one day at a time:

```bash
pdm run test --week 0 --day 1
```

The command copies `tests_refsol/test_week_0_day_1.py` into `tests/` and runs it against your code in
`src/tiny_llm_week0`. In the untouched starter, every test fails. Narrow the run to one task while you work:

```bash
pdm run test --week 0 --day 1 -- -k task_3
```

To see what passing looks like, run the same tests against the reference solution:

```bash
pdm run test-refsol --week 0 --day 1
```

If you are on a machine where `pdm install` cannot resolve MLX (for example Linux or Windows), plain `pip` is enough
for Week 0:

```bash
pip install numpy pytest
python scripts/dev-tools.py test --week 0 --day 1
```

## The Week at a Glance

| Day | Topic | Starter module | Builds toward |
|---|---|---|---|
| [1](./week0-01-numerics.md) | Numerics and linear algebra | `numerics.py` | softmax, BF16 activations, matmul cost (Weeks 1–2) |
| [2](./week0-02-arrays.md) | Array programming | `arrays.py` | multi-head reshapes, broadcasting masks, strided views (Week 1) |
| [3](./week0-03-neural-networks.md) | Neural networks and backprop | `nn.py` | linear layers, activations, the MLP (Week 1 Day 4) |
| [4](./week0-04-autodiff.md) | A tiny autodiff engine | `autodiff.py` | how frameworks such as MLX and PyTorch compute gradients |
| [5](./week0-05-language-modeling.md) | Tokens and language models | `lm.py` | tokenization, log probabilities, sampling, the decode loop (Week 1 Days 6–7) |
| [6](./week0-06-transformer-block.md) | The transformer block | `transformer.py` | attention, RoPE, RMSNorm, SwiGLU (Week 1 Days 1–5) |
| [7](./week0-07-inference-math.md) | Inference systems math | `systems.py` | KV-cache sizing, quantization, tiling, rooflines (Weeks 2–3) |

Two days are projects rather than drills: Day 3 ends with training a small MLP by hand, and Day 4 is building a
reverse-mode autodiff engine and training a model with it.

The courses that the preface recommends have their own homework. The
[external coursework](./week0-external-coursework.md) page explains which of those assignments are bundled with this
repository and which ones Week 0 replaces.

## Self-Check: Can I Skip Week 0?

You can go straight to Week 1 if you can answer these without looking anything up:

- What shape does `(2, 1, 5) + (7, 1)` broadcast to, and why does `x.transpose(...)` usually not copy memory?
- Why do we subtract the maximum before exponentiating in softmax, and roughly how many decimal digits does BF16 keep?
- Given `y = x @ W.T + b`, what are the gradients with respect to `x`, `W`, and `b`?
- How does byte-pair encoding turn a string into token IDs, and what does perplexity measure?
- Write single-head causal attention from memory. What does a causal mask look like?
- How many bytes of KV cache does one token of Qwen3-0.6B need in BF16, and why is single-request decode memory bound?

If any of these are fuzzy, spend a day on the matching chapter. The tests tell you when you are done.

## Readings for the Whole Week

- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [3Blue1Brown: Neural Networks](https://www.3blue1brown.com/topics/neural-networks) (the visual intuition for Days 3–6)
- [Andrej Karpathy: Neural Networks, Zero to Hero](https://karpathy.ai/zero-to-hero.html) (micrograd, makemore, and the GPT
  tokenizer lectures pair with Days 4, 5, and 6)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [MLX documentation](https://ml-explore.github.io/mlx/build/html/index.html), especially "Quick Start" and "Lazy Evaluation"

{{#include copyright.md}}
