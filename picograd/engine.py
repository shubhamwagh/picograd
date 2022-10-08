"""
Autograd engine implementing reverse-mode auto-differentiation, aka backpropagation.

Heavily inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
"""

import math
from typing import Union, Tuple, List, Set, Callable

FloatInt = Union[float, int]


class Var:
    """ stores a single scalar value and its gradient """

    def __init__(self, data: FloatInt, children: Tuple["Var", ...] = (), op: str = "",
                 label: str = "") -> None:
        self.data: FloatInt = data
        self.grad: float = 0.0

        # Internal variables used for autograd graph construction
        self._backward: Callable = lambda: None
        self._prev: Set[Var] = set(children)
        self._op: str = op  # The operation that produced this node, for graphviz / debugging / etc
        self._label: str = label

    @property
    def children(self):
        return self._prev

    @property
    def op(self):
        return self._op

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label_name: str):
        self._label = label_name

    def __repr__(self):
        if len(self._label) != 0:
            return f"Var(data={self.data}, label={self.label})"
        else:
            return f"Var(data={self.data})"

    def __add__(self, other: Union["Var", FloatInt]) -> "Var":
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.data + other.data, children=(self, other), op="+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad  # local_grad * global_grad -> chain rule
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __neg__(self) -> "Var":  # -self
        return self * -1

    def __radd__(self, other: Union["Var", FloatInt]) -> "Var":  # other + self
        return self + other

    def __sub__(self, other: Union["Var", FloatInt]) -> "Var":  # self - other
        return self + (-other)

    def __rsub__(self, other: Union["Var", FloatInt]) -> "Var":  # other - self
        return -self + other

    def __mul__(self, other: Union["Var", FloatInt]) -> "Var":
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.data * other.data, children=(self, other), op='x')

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other: Union["Var", FloatInt]) -> "Var":  # other * self
        return self * other

    def __pow__(self, other: FloatInt) -> "Var":
        assert isinstance(other, (int, float)), "only supporting int/flot powers for now"
        out = Var(self.data ** other, children=(self,), op=f'pow{other}')

        def _backward() -> None:
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other: Union["Var", FloatInt]) -> "Var":  # self / other
        return self * other ** -1

    def __rtruediv__(self, other: Union["Var", FloatInt]) -> "Var":  # other / self
        return other * self ** -1

    def exp(self) -> "Var":
        """Compute exp()"""

        x = self.data
        out = Var(math.exp(x), children=(self,), op='exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def tanh(self) -> "Var":
        """Compute tanh()"""

        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Var(t, children=(self,), op='tanh')

        def _backward() -> None:
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    def relu(self) -> "Var":
        """Compute ReLU"""

        out = Var(0 if self.data < 0 else self.data, children=(self,), op='ReLU')

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self) -> "Var":
        """Compute sigmoid()"""

        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Var(s, children=(self,), op='sigmoid')

        def _backward() -> None:
            self.grad += (1 - s) * out.grad

        out._backward = _backward

        return out

    def backward(self) -> None:
        """Compute gradients through backpropagation"""

        # Topological order of all the children in the graph from left to right edges
        topo: List[Var] = []
        visited: Set[Var] = set()

        def build_topo(node: Var) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
