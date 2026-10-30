# Week 0 Day 3: Neural Networks and Backpropagation

The serving stack you build in Weeks 1–3 only runs the forward pass, but you should still know where the weights came
from. Today you will implement a linear layer, its activations, and its gradients by hand, verify them numerically, and
train a small classifier. This is the material that the preface's recommended course, CMU 10-601, covers in its neural
network unit.

```
src/tiny_llm_week0/nn.py
```

```bash
pdm run test --week 0 --day 3
```

**📚 Readings**

- [3Blue1Brown: Neural Networks](https://www.3blue1brown.com/topics/neural-networks), chapters 1–4
- [CS231n: Optimization and Backpropagation](https://cs231n.github.io/optimization-2/)
- [Matrix calculus for deep learning](https://explained.ai/matrix-calculus/) (reference, not cover to cover)

## Task 1: Activations

Implement `relu`, `silu`, `silu_grad`, and `gelu_tanh`.

$$
\text{SiLU}(x) = x \cdot \sigma(x), \qquad \frac{d}{dx}\text{SiLU}(x) = \sigma(x)\,(1 + x(1 - \sigma(x)))
$$

$$
\text{GELU}_{\tanh}(x) = \tfrac{1}{2}x\left(1 + \tanh\!\left(\sqrt{2/\pi}\,(x + 0.044715x^3)\right)\right)
$$

A stable `_sigmoid` helper is supplied. Qwen3 uses SiLU inside its SwiGLU MLP (Week 1 Day 4); GPT-2 uses GELU.

## Task 2: The Linear Layer and Gradient Checking

Implement `numerical_gradient(f, x)` first: perturb each element of a copy of `x` by `±eps` and take the central
difference. Do not modify the caller's array. You will use it to check every gradient you derive.

Then implement the linear layer in the course's weight convention, where `w` has shape `(out_features, in_features)`:

$$
y = x W^\top + b
$$

`linear_forward(x, w, b)` must accept extra leading axes on `x`. `linear_backward(dy, x, w)` returns
`(dx, dw, db)` given the upstream gradient `dy = dL/dy`:

$$
\frac{\partial L}{\partial x} = dy\,W, \qquad \frac{\partial L}{\partial W} = dy^\top x, \qquad \frac{\partial L}{\partial b} = \sum_{\text{rows}} dy
$$

When `x` has leading batch axes, flatten them before computing `dw` and `db`.

## Task 3: Softmax Cross-Entropy

Implement `softmax_cross_entropy(logits, targets)`, returning the mean loss and its gradient with respect to the logits.
Compute log probabilities with the stable trick from Day 1. The gradient has a famously simple form:

$$
\frac{\partial L}{\partial z} = \frac{1}{N}\left(\text{softmax}(z) - \text{onehot}(y)\right)
$$

This is the same loss that language models are trained with, where the classes are the vocabulary and the target is
the next token.

## Task 4 (Project): Train an MLP by Hand

`init_mlp` is supplied. Implement:

- `mlp_forward(params, x)`: `linear -> relu -> linear`, returning logits.
- `mlp_train_step(params, x, y, lr)`: one full-batch step of gradient descent. Run the forward pass while keeping the
  intermediate values, compute the loss and `dlogits`, backpropagate through the second linear layer, the ReLU (its
  gradient is `1` where the pre-activation was positive and `0` elsewhere), and the first linear layer. Update the
  parameters in place and return the loss.
- `train_mlp(...)`: initialize, then call `mlp_train_step` for `epochs` steps. Return the parameters and the list of losses.

The test trains on three interleaved spirals, a dataset that no linear classifier can separate, and expects at least 90%
training accuracy. If it fails, check each gradient with `numerical_gradient` before touching the learning rate.

{{#include copyright.md}}
