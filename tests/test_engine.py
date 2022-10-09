import unittest
import torch

from picograd.engine import Var


class TestEngine(unittest.TestCase):
    def test_scalar(self):
        x = Var(1.0)
        y = (x * 2 + 1).relu()
        self.assertEqual(y.data, 3)
        y.backward()
        self.assertEqual(x.grad, 2)  # dy/dx

    def test_scalar_sub(self):
        x = Var(2.5)
        y = Var(3.5)

        z = x - y
        self.assertEqual(z.data, -1)
        z.backward()
        self.assertEqual(x.grad, 1)  # dz/dx
        self.assertEqual(y.grad, -1)  # dz/dy

        # Reset gradients
        x.grad = 0
        y.grad = 0

        z = y - x
        self.assertEqual(z.data, 1)
        z.backward()
        self.assertEqual(x.grad, -1)  # dz/dx
        self.assertEqual(y.grad, 1)  # dz/dy

        # Reset gradient
        x.grad = 0

        z = x - 3
        self.assertEqual(z.data, -0.5)
        z.backward()
        self.assertEqual(x.grad, 1)  # dz/dx

        # Reset gradient
        x.grad = 0

        z = 3 - x
        self.assertEqual(z.data, 0.5)
        z.backward()
        self.assertEqual(x.grad, -1)  # dz/dx

    def test_scalar_div(self):
        x = Var(1.0)
        y = Var(4.0)

        z = x / y
        self.assertEqual(z.data, 0.25)
        z.backward()
        self.assertEqual(x.grad, 0.25)  # dz/dx
        self.assertEqual(y.grad, -0.0625)  # dz/dy

        # Reset gradients
        x.grad = 0
        y.grad = 0

        z = y / x
        self.assertEqual(z.data, 4)
        z.backward()
        self.assertEqual(x.grad, -4)  # dz/dx
        self.assertEqual(y.grad, 1)  # dz/dy

        # Reset gradient
        x.grad = 0

        z = x / 3
        self.assertEqual(z.data, 1 / 3)
        z.backward()
        self.assertEqual(x.grad, 1 / 3)  # dz/dx

        # Reset gradient
        x.grad = 0

        z = 3 / x
        self.assertEqual(z.data, 3)
        z.backward()
        self.assertEqual(x.grad, -3)  # dz/dx


class TestEngineVersusPyTorch(unittest.TestCase):
    def test_sanity_check(self):
        x = Var(-4.0)
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        xmg, ymg = x, y

        x = torch.Tensor([-4.0]).double()
        x.requires_grad = True
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        xpt, ypt = x, y

        # forward pass went well
        self.assertEqual(ymg.data, ypt.data.item())
        # backward pass went well
        self.assertEqual(xmg.grad, xpt.grad.item())

    def test_more_ops(self):
        a = Var(-4.0)
        b = Var(2.0)
        c = a + b
        d = a * b + b ** 3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e ** 2
        g = f / 2.0
        g += 10.0 / f
        g.backward()
        amg, bmg, gmg = a, b, g

        a = torch.Tensor([-4.0]).double()
        b = torch.Tensor([2.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = a + b
        d = a * b + b ** 3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e ** 2
        g = f / 2.0
        g = g + 10.0 / f
        g.backward()
        apt, bpt, gpt = a, b, g

        tol = 1e-6
        # forward pass went well
        self.assertLess(abs(gmg.data - gpt.data.item()), tol)
        # backward pass went well
        self.assertLess(abs(amg.grad - apt.grad.item()), tol)
        self.assertLess(abs(bmg.grad - bpt.grad.item()), tol)


if __name__ == "__main__":
    unittest.main()
