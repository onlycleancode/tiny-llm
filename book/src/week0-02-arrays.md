# Week 0 Day 2: Array Programming

Most bugs in Week 1 are shape bugs: a transpose on the wrong axes, a mask that broadcasts the wrong way, or a reshape
that silently scrambles attention heads. Today you will implement the rules that NumPy and MLX apply for you, so that
you can predict them.

```
src/tiny_llm_week0/arrays.py
```

```bash
pdm run test --week 0 --day 2
```

**📚 Readings**

- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy internals: memory layout and strides](https://numpy.org/doc/stable/dev/internals.html)
- [MLX: Quick Start](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html) and
  [Lazy Evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)

## Task 1: Broadcasting

Implement `broadcast_shape(*shapes)` without calling `np.broadcast_shapes`. Align the shapes on their last axis. For
each axis, the sizes must be equal or one of them must be `1`; the result takes the larger size. Missing leading axes
count as `1`. Raise `ValueError` for incompatible shapes.

This is how a `(L, L)` causal mask adds to `(B, H, L, L)` attention scores and how a `(D,)` bias adds to `(B, L, D)`
activations.

## Task 2: Strides

An array is a flat buffer plus a shape and strides: the number of bytes to step for each axis. Implement:

- `contiguous_strides(shape, itemsize)`: the strides of a row-major (C-contiguous) array.
- `transpose_strides(shape, strides, axes)`: the new shape and strides after permuting axes. No data moves; raise
  `ValueError` if `axes` is not a permutation.
- `is_contiguous(shape, strides, itemsize)`: whether the strides describe a row-major layout. Axes of size 1 may have any
  stride.

The tests compare your answers with NumPy's own `.strides`. A transposed array is a view with permuted strides; a GPU
kernel that assumes contiguous rows will read it wrong. Week 2 kernels check contiguity for exactly this reason.

## Task 3: Splitting and Merging Attention Heads

Multi-head attention projects a `(..., L, H*D)` activation and then treats it as `H` independent heads of width `D`.

- `split_heads(x, num_heads)`: `(..., L, H*D) -> (..., H, L, D)`. Reshape to `(..., L, H, D)`, then swap the `L` and `H`
  axes. Raise `ValueError` if the last dimension does not divide evenly.
- `merge_heads(x)`: the inverse, `(..., H, L, D) -> (..., L, H*D)`.

Head `h` at position `l` must be the slice `x[..., l, h*D:(h+1)*D]`. Reshaping directly to `(..., H, L, D)` without the
transpose produces the right shape and the wrong data; the test catches it. You will use this exact pattern in Week 1
Day 1.

## Task 4: Batched Matmul from Broadcasting

Implement `batched_matmul(a, b)` with `np.matmul` semantics (leading batch axes broadcast) using only broadcasting and
`sum`. Insert axes so that `a` becomes `(..., M, K, 1)` and `b` becomes `(..., 1, K, N)`, multiply, and reduce over `K`.

This is the slow-but-clear form of every attention score computation: `Q @ K^T` over batch and head axes.

## Task 5 (Optional, Needs MLX): Softmax in MLX

Implement `mlx_softmax(x, axis)` using `mlx.core` operations (`mx.max`, `mx.exp`, `mx.sum`). Import MLX inside the
function so the rest of the module works without it. The test is skipped when MLX is not installed.

While you are in MLX, try this in a Python shell and notice that nothing is computed until you ask for it:

```python
import mlx.core as mx
a = mx.random.normal((4096, 4096))
b = a @ a        # returns immediately: this only records the graph
mx.eval(b)       # the matmul runs here
```

MLX's lazy evaluation is why Week 1's tests and benchmarks call `mx.eval` before measuring.

{{#include copyright.md}}
