import unittest

from picograd.engine import Var
from picograd.metrics import mean_squared_error, binary_accuracy


class TestMetrics(unittest.TestCase):
    def test_mse(self):
        y_true = [Var(3), Var(-0.5), Var(2), Var(7)]
        y_pred = [Var(2.5), Var(0.0), Var(2), Var(8)]
        error = mean_squared_error(y_true, y_pred)
        self.assertAlmostEqual(error.data, 0.375, 3)

    def test_binary_accuracy(self):
        y_true = [Var(0), Var(1), Var(2), Var(3)]
        y_pred = [Var(0), Var(2), Var(1), Var(3)]

        acc = binary_accuracy(y_true, y_pred)
        self.assertEqual(acc, 0.5)

        acc = binary_accuracy(y_true, y_true)
        self.assertEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
