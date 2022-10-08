import random
from typing import NamedTuple, Iterator, List
from picograd.engine import Var

Batch = NamedTuple("Batch", [("inputs", List[List[Var]]), ("targets", List[Var])])


class BatchIterator:
    """Iterates on data by batches"""

    def __init__(self, inputs: List[List[Var]], targets: List[Var], batch_size: int = 32, shuffle: bool = True) -> None:
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self) -> Iterator[Batch]:
        starts = list(range(0, len(self.inputs), self.batch_size))
        if self.shuffle:
            random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = self.inputs[start:end]
            batch_targets = self.targets[start:end]
            yield Batch(inputs=batch_inputs, targets=batch_targets)
