"""Week 0 Day 4: a minimal reverse-mode autodiff engine (Needle-lite)."""

import numpy as np

__all__ = ["unbroadcast", "Tensor"]


# Task 1: undo broadcasting in the backward pass
def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum grad down to `shape`, undoing NumPy broadcasting."""
    pass


class Tensor:
    """A NumPy array that remembers how it was computed."""

    def __init__(self, data, parents: tuple["Tensor", ...] = (), op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self.parents = parents
        self.op = op
        # Called during backward with this tensor's gradient; it must add
        # the gradient contribution of each parent into parent.grad.
        self._backward = lambda grad: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor({self.data!r}, op={self.op!r})"

    @staticmethod
    def _wrap(other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def _accumulate(self, grad: np.ndarray) -> None:
        grad = unbroadcast(grad, self.shape)
        self.grad = grad if self.grad is None else self.grad + grad

    # Task 2: operators
    def __add__(self, other) -> "Tensor":
        """Elementwise add; the backward closure sends grad to both parents."""
        pass

    def __mul__(self, other) -> "Tensor":
        """Elementwise multiply."""
        pass

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other) -> "Tensor":
        return self + (-self._wrap(other))

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other) -> "Tensor":
        return self._wrap(other) - self

    def __matmul__(self, other) -> "Tensor":
        """Matrix multiply."""
        pass

    def sum(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        """Sum over axis (or everything when axis is None)."""
        pass

    def relu(self) -> "Tensor":
        """max(x, 0)"""
        pass

    def exp(self) -> "Tensor":
        """Elementwise exp."""
        pass

    def log(self) -> "Tensor":
        """Elementwise natural log."""
        pass

    # Task 3: backpropagation
    def topological_order(self) -> list["Tensor"]:
        """Every tensor this one depends on (and itself), each listed after all of its parents."""
        pass

    def backward(self) -> None:
        """Seed this tensor's grad with ones and run every _backward in reverse topological order."""
        pass
