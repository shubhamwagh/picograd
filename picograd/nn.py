import random
from abc import ABC, abstractmethod
from picograd.engine import Var

from typing import List, Optional, Union


class Module(ABC):
    def __init(self) -> None:
        pass

    @abstractmethod
    def parameters(self) -> List[Var]:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    """A single neuron"""

    def __init__(self, in_features: int, activation: Optional[str] = None):
        self.w = [Var(random.uniform(-1, 1)) for _ in range(in_features)]
        self.b = Var(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        # w * x + b
        out = sum([w_i * x_i for w_i, x_i in zip(self.w, x)], self.b)
        if self.activation is None or self.activation is 'linear':
            return out
        elif self.activation == 'relu':
            return out.relu()
        elif self.activation == 'tanh':
            return out.tanh()
        elif self.activation == 'sigmoid':
            return out.sigmoid()
        raise NotImplementedError(
            f"Unexpected activation argument ('relu', 'tanh' and 'sigmoid' available). Got {self.activation}.")

    def parameters(self) -> List[Var]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"Neuron({len(self.w)}, {self.activation if self.activation is not None else 'linear'})"


class Layer(Module):
    """A layer of neurons"""

    def __init__(self, in_features: int, out_features: int, activation: Optional[str] = None):
        self.neurons = [Neuron(in_features, activation) for _ in range(out_features)]

    def __call__(self, x: List[Var]) -> List[Var]:
        outs = [n(x) for n in self.neurons]
        return outs  # outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Var]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """ A Multi-layer Perceptron """

    def __init__(self, in_features: int, layers: List[int], activations: List[str]):
        sizes = [in_features] + layers
        assert len(activations) != 0, "Please provide activation for layers. Available -> 'relu', 'tanh', 'sigmoid', 'linear'"
        assert len(activations) == len(layers), "length of activations does not match the length of layers"
        self.layers = [Layer(in_features=sizes[i], out_features=sizes[i + 1], activation=activations[i])
                       for i in range(len(layers))]

    def __call__(self, x: List[Var]) -> Union[List[Var], Var]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Var]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
