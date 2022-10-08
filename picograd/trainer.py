from picograd.nn import Module
from picograd.optim import Optimizer
from picograd.data import BatchIterator

from typing import Callable, Dict, List

# Used to record training history for metrics
History = Dict[str, List[float]]


# noinspection PyTypeChecker
class Trainer:
    """Encapsulates the model training loop"""

    def __init__(self, model: Module, optimizer: Optimizer, loss: Callable, acc_metric: Callable) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.acc_metric = acc_metric

    def fit(self, data_iterator: BatchIterator, num_epochs: int = 500, verbose: bool = False) -> History:
        """Fits the model to the data"""

        history: History = {"loss": [], "acc": []}
        epoch_loss = 0
        epoch_acc = 0
        epoch_y_true = []
        epoch_y_pred = []
        for epoch in range(num_epochs):
            # Reset the gradients of model parameters
            self.optimizer.zero_grad()
            # Reset epoch data
            epoch_loss = 0
            epoch_y_true = []
            epoch_y_pred = []

            for batch in data_iterator():
                # Forward pass
                # outputs = [self.model(mini_batch_input) for mini_batch_input in batch.inputs]
                outputs = list(map(self.model, batch.inputs))

                # Loss computation
                batch_y_pred = [item for sublist in outputs for item in sublist]
                batch_loss = self.loss(batch.targets, batch_y_pred)
                epoch_loss += batch_loss.data

                # Store batch predictions and ground truth for computing epoch metrics
                epoch_y_pred.extend(batch_y_pred)
                epoch_y_true.extend(batch.targets)

                # Backprop and gradient descent
                batch_loss.backward()
                self.optimizer.step()

            # Accuracy computation for epoch
            epoch_acc = self.acc_metric(epoch_y_true, epoch_y_pred)

            # Record training history
            history["loss"].append(epoch_loss)
            history["acc"].append(epoch_acc)

            if verbose:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"loss: {epoch_loss:.6f}, "
                    f"accuracy: {epoch_acc * 100:.2f}%"
                )

        return history
