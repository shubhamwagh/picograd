from abc import ABC, abstractmethod
from picograd.engine import Var
from typing import List


class Optimizer(ABC):
    """Base class for optimizers"""

    def __init__(self, parameters: List[Var]) -> None:
        self.parameters: List[Var] = parameters

    def zero_grad(self) -> None:
        """Reset gradients for all parameters"""

        for p in self.parameters:
            p.grad = 0.0

    @abstractmethod
    def step(self) -> None:
        """Take a step of gradient descent"""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, parameters: List[Var], lr: float = 0.01, momentum: float = 0.0) -> None:
        super(SGD, self).__init__(parameters)
        assert momentum >= 0.0, "momentum cannot be negative"
        self.lr = lr
        self.momentum = momentum

        self._momentums = [0] * len(parameters)

    def step(self) -> None:
        """Update model parameters in the opposite direction of their gradient"""

        for ind, p in enumerate(self.parameters):
            # p.data -= self.lr * p.grad
            self._momentums[ind] = self._momentums[ind] * self.momentum + self.lr * p.grad
            p.data -= self._momentums[ind]


class Adam(Optimizer):
    def __init__(self, parameters: List[Var], lr: float = 1e-3, beta_1: float = 0.0, beta_2: float = 0.999,
                 eps: float = 1e-8) -> None:
        super(Adam, self).__init__(parameters)
        assert (0 <= beta_1) and (beta_1 < 1), "smoothing factor must be in [0,1)"
        assert (0 <= beta_2) and (beta_2 < 1), "smoothing factor must be in [0,1)"

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self._t = 0
        self._exp_avg = [0] * len(parameters)
        self._exp_avg_sq = [0] * len(parameters)

    def step(self) -> None:
        self._t += 1

        for ind, p in enumerate(self.parameters):
            self._exp_avg[ind] = self.beta_1 * self._exp_avg[ind] + (1. - self.beta_1) * p.grad
            self._exp_avg_sq[ind] = self.beta_2 * self._exp_avg_sq[ind] + (1. - self.beta_2) * (p.grad ** 2)

            bias_correction_1 = self._exp_avg[ind] / (1. - (self.beta_1 ** self._t))
            bias_correction_2 = self._exp_avg_sq[ind] / (1. - (self.beta_2 ** self._t))

            p.data -= self.lr * bias_correction_1 / (bias_correction_2 ** 0.5 + self.eps)
