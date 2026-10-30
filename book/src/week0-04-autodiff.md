# Week 0 Day 4: A Tiny Autodiff Engine

Yesterday you derived every gradient by hand. Frameworks such as PyTorch, MLX, and the Needle framework from CMU's Deep
Learning Systems course record the operations of the forward pass in a graph and walk it backward instead. Today you
will build a small version of that machinery on top of NumPy arrays. This day is a project: the tests end by training a
classifier with your engine.

```
src/tiny_llm_week0/autodiff.py
```

```bash
pdm run test --week 0 --day 4
```

**📚 Readings**

- [Andrej Karpathy: The spelled-out intro to neural networks and backpropagation (micrograd)](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Deep Learning Systems, Lecture 4: Automatic Differentiation](https://dlsyscourse.org/lectures/)
- [Deep Learning Systems homework 1 (Needle)](https://github.com/dlsyscourse/hw1), which you can also work through from
  the [external coursework](./week0-external-coursework.md) page

## How the Engine Works

A `Tensor` holds `data` (a float64 NumPy array), `grad`, its `parents`, and a `_backward` closure. Each operation creates
a new tensor whose `_backward(grad)` adds the right contribution into each parent's gradient through `_accumulate`.
`__init__`, `_wrap`, and `_accumulate` are supplied, as are `__neg__`, `__sub__`, `__rsub__`, `__radd__`, and `__rmul__`,
which are defined in terms of the operators you write.

## Task 1: Undo Broadcasting

Implement `unbroadcast(grad, shape)`. If `x` of shape `(3,)` was broadcast to `(2, 3)` in the forward pass, its gradient
is the `(2, 3)` upstream gradient summed over the broadcast axis. Sum away extra leading axes, then sum (keeping the
axis) over every axis where `shape` has size 1 but `grad` does not.

## Task 2: Operators

Implement the forward value and the backward closure for `__add__`, `__mul__`, `__matmul__`, `sum`, `relu`, `exp`, and
`log`. For example, for `out = a * b`, the closure adds `grad * b.data` to `a` and `grad * a.data` to `b`. For
`out = a @ b`, it adds `grad @ b^T` to `a` and `a^T @ grad` to `b`. For `sum(axis)`, it broadcasts `grad` back to the
input shape (re-inserting the reduced axis when `keepdims` is false).

## Task 3: Backpropagation

Implement `topological_order()`, returning every tensor that this tensor depends on, including itself, each listed after
all of its parents. A recursive depth-first search works; an explicit stack avoids Python's recursion limit on deep
graphs.

Then implement `backward()`: set this tensor's `grad` to ones and call each node's `_backward` in reverse topological
order. Because `_accumulate` adds rather than assigns, a tensor used several times (such as `x` in `x * x + x`) receives
the sum of its contributions.

## Task 4 (Project): Train with Your Engine

The final test fits a logistic-regression classifier using only your `Tensor` operations and plain gradient descent. If
it fails while the earlier tests pass, look for a missing `unbroadcast` on the bias gradient.

As a stretch goal, rebuild yesterday's spiral MLP with your engine and confirm that it reaches the same accuracy as the
hand-derived version.

{{#include copyright.md}}
