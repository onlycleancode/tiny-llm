"""Week 0 Day 4: a minimal reverse-mode autodiff engine (Needle-lite)."""

import numpy as np

__all__ = ["unbroadcast", "Tensor"]


# Task 1: undo broadcasting in the backward pass
def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


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
        other = self._wrap(other)
        out = Tensor(self.data + other.data, (self, other), "add")

        def backward(grad):
            self._accumulate(grad)
            other._accumulate(grad)

        out._backward = backward
        return out

    def __mul__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = Tensor(self.data * other.data, (self, other), "mul")

        def backward(grad):
            self._accumulate(grad * other.data)
            other._accumulate(grad * self.data)

        out._backward = backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other) -> "Tensor":
        return self + (-self._wrap(other))

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other) -> "Tensor":
        return self._wrap(other) - self

    def __matmul__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def backward(grad):
            self._accumulate(grad @ np.swapaxes(other.data, -1, -2))
            other._accumulate(np.swapaxes(self.data, -1, -2) @ grad)

        out._backward = backward
        return out

    def sum(self, axis: int | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def backward(grad):
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self._accumulate(np.broadcast_to(grad, self.shape))

        out._backward = backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0), (self,), "relu")

        def backward(grad):
            self._accumulate(grad * (self.data > 0))

        out._backward = backward
        return out

    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), (self,), "exp")

        def backward(grad):
            self._accumulate(grad * out.data)

        out._backward = backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), (self,), "log")

        def backward(grad):
            self._accumulate(grad / self.data)

        out._backward = backward
        return out

    # Task 3: backpropagation
    def topological_order(self) -> list["Tensor"]:
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for parent in node.parents:
                if id(parent) not in seen:
                    stack.append((parent, False))
        return order

    def backward(self) -> None:
        self.grad = np.ones_like(self.data)
        for node in reversed(self.topological_order()):
            if node.grad is not None:
                node._backward(node.grad)
