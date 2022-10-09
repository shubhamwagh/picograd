import unittest
from collections import Counter

from picograd.engine import Var
from picograd.data import BatchIterator


class TestDataUtils(unittest.TestCase):

    def test_batch_iterator(self):
        # Generate lists of 8 scalars with same values
        DATASET_SIZE = 8
        inputs = [[Var(x)] for x in range(DATASET_SIZE)]
        targets = [Var(x) for x in range(DATASET_SIZE)]

        batch_size = 3
        data_iterator = BatchIterator(inputs, targets, batch_size=batch_size)

        batch_sizes = []
        for batch in data_iterator():
            # Check lengths of batches
            self.assertEqual(len(batch.inputs), len(batch.targets))
            self.assertLessEqual(len(batch.inputs), batch_size)
            self.assertLessEqual(len(batch.targets), batch_size)

            # Store batch size (same for inputs and targets)
            batch_sizes.append(len(batch.inputs))

        # Check expected sizes of batches (exact order can vary)
        self.assertEqual(Counter(batch_sizes), Counter([3, 3, 2]))


if __name__ == "__main__":
    unittest.main()
