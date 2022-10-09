import unittest

from picograd.engine import Var
from picograd.nn import MLP
from picograd.optim import SGD
from picograd.metrics import mean_squared_error, binary_accuracy
from picograd.data import BatchIterator
from picograd.trainer import Trainer, History


class TestTrainer(unittest.TestCase):
    def test_trainer(self):
        # dataset for AND logical function
        x_train = [
            list(map(Var, x)) for x in [[0, 0], [0, 1], [1, 0], [1, 1]]
        ]
        y_train = [Var(0), Var(0), Var(0), Var(1)]

        model = MLP(in_features=2, layers=[1], activations=['linear'])  # Logistic regression
        optimizer = SGD(model.parameters(), lr=0.1)
        data_iterator = BatchIterator(x_train, y_train)

        num_epochs = 50
        trainer = Trainer(model, optimizer, loss=mean_squared_error, acc_metric=binary_accuracy)
        history: History = trainer.fit(data_iterator, num_epochs=num_epochs, verbose=True)

        # Training metrics are recorded for each epoch
        self.assertEqual(len(history["loss"]), len(history["acc"]))
        self.assertEqual(len(history["loss"]), num_epochs)
        self.assertEqual(len(history["acc"]), num_epochs)

        # Access final values for metrics
        loss = history["loss"][-1]
        acc = history["acc"][-1]

        self.assertLess(loss, 0.1)
        self.assertEqual(acc, 1.0)  # 100%


if __name__ == "__main__":
    unittest.main()
