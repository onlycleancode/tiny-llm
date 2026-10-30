# Week 0 Day 1: Numerics and Linear Algebra

LLM inference is mostly matrix multiplication done in low precision. Today you will write a matrix multiply by hand,
count how much work and memory traffic it needs, implement the numerically stable softmax that attention depends on, and
see exactly what precision BF16 keeps.

```
src/tiny_llm_week0/numerics.py
```

Run the day's tests at any point:

```bash
pdm run test --week 0 --day 1
```

**📚 Readings**

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra), chapters 1–4 (vectors, linear maps, matrix multiplication)
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) (skim the first section)
- [bfloat16 floating-point format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)

## Task 1: Matrix Multiplication from Loops

Implement `matmul_naive(a, b)` for 2-D arrays with three nested loops. Do not use `@`, `np.dot`, or `np.matmul`. Raise
`ValueError` when the inner dimensions do not match.

$$
C_{ij} = \sum_{p=0}^{K-1} A_{ip} B_{pj}
$$

Keep the shapes in your head as `(M, K) @ (K, N) -> (M, N)`. Every projection in the model has this form: a batch of
`M` token vectors of width `K` multiplied by a weight matrix that maps width `K` to width `N`.

## Task 2: Counting FLOPs and Bytes

Implement `matmul_flops`, `matmul_bytes`, and `arithmetic_intensity`.

- Each of the `M * N` outputs needs `K` multiplies and `K` adds, so a matmul costs `2 * M * K * N` floating-point
  operations.
- If `A`, `B`, and `C` each cross the memory bus exactly once, the matmul moves `(M*K + K*N + M*N) * bytes_per_element`
  bytes.
- Arithmetic intensity is FLOPs per byte.

The tests compare a decode step (`M = 1`, one new token) with a prefill step (`M = 4096`, a whole prompt) through the same
`4096 x 4096` projection. Decode does about one FLOP per byte; prefill does over a thousand. Keep that contrast in mind:
it is the reason Week 2 treats decode and prefill with different kernels, and Day 7 turns it into a speed estimate.

## Task 3: Stable Softmax and Log-Sum-Exp

Implement `softmax(x, axis)` and `logsumexp(x, axis)`.

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
$$

$$
\log \sum_j e^{x_j} = m + \log \sum_j e^{x_j - m}
$$

Subtracting the maximum does not change the result but keeps every exponent at or below zero, so nothing overflows.
`np.exp(1000)` is `inf`; `np.exp(0)` is fine. `logsumexp` removes the reduced axis. Week 1 Day 1 offers a bonus exercise
to write this same softmax in MLX, and Week 2's attention kernels use the running-maximum version of this trick.

## Task 4: Rounding to BF16

The course model stores weights and activations in BF16. BF16 is the top 16 bits of an FP32 value: the same sign bit and
8-bit exponent (so the same range), but only 7 explicit mantissa bits instead of 23.

Implement `round_to_bf16(x)`, which takes a `float32` array and returns a `float32` array whose values are exactly
representable in BF16, using round-to-nearest with ties to even:

1. View the array as `uint32` with `x.view(np.uint32)`.
2. Add `0x7FFF` plus the lowest kept bit (`(bits >> 16) & 1`). Use a wider integer type so the sum cannot overflow.
3. Clear the low 16 bits and view the result as `float32` again.
4. Leave NaN values unchanged.

Then look at what you built: the relative error is at most `2^-8`, which is roughly two to three significant decimal
digits. This is why Week 1 promotes sensitive reductions such as softmax and RMSNorm to FP32 before casting back.

{{#include copyright.md}}
