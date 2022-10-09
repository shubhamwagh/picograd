import unittest

from picograd.engine import Var
from picograd.optim import SGD, Adam


class TestOptimizer(unittest.TestCase):

    def test_sgd(self):
        params = [Var(3), Var(-1), Var(0.5)]
        sgd = SGD(parameters=params, lr=0.1)

        # Simulate gradient update (normally done through backprop)
        for i, p in enumerate(params):
            p.grad = i

        sgd.step()
        self.assertEqual(params[0].data, 3)  # 3 - 0.1 * 0
        self.assertEqual(params[1].data, -1.1)  # -1 - 0.1 * 1
        self.assertEqual(params[2].data, 0.3)  # 0.5 - 0.1 * 2

        sgd.zero_grad()
        for _, p in enumerate(params):
            self.assertEqual(p.grad, 0)

    def test_sgd_with_momentum_and_nesterov(self):
        params = [Var(3), Var(-1), Var(0.5)]
        sgd = SGD(parameters=params, lr=0.1, momentum=0.9, nesterov=True)

        # Simulate gradient update (normally done through backprop)
        for i, p in enumerate(params):
            p.grad = i

        sgd.step()
        self.assertEqual(params[0].data, 3)  # 3 - (1 + 0.9) * 0.1 * 0
        self.assertEqual(params[1].data, -1.19)  # -1 - (1 + 0.9) * 0.1 * 1
        self.assertEqual(params[2].data, 0.12)  # 0.5 - (1 + 0.9) * 0.1 * 2

        sgd.zero_grad()
        for _, p in enumerate(params):
            self.assertEqual(p.grad, 0)

    def test_adam(self):
        params = [Var(3), Var(-1), Var(0.5)]
        sgd = Adam(parameters=params, lr=1e-3)

        # Simulate gradient update (normally done through backprop)
        for i, p in enumerate(params):
            p.grad = i

        sgd.step()
        self.assertAlmostEqual(params[0].data, 3, 2)
        self.assertAlmostEqual(params[1].data, -1, 2)
        self.assertAlmostEqual(params[2].data, 0.5, 2)

        sgd.zero_grad()
        for _, p in enumerate(params):
            self.assertEqual(p.grad, 0)


if __name__ == "__main__":
    unittest.main()
