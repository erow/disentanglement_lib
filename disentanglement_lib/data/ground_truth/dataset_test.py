import unittest
from . import dsprites
import numpy as np


class DatasetTestCase(unittest.TestCase):
    def test_dsprites(self):
        ds = dsprites.DSprites([2, 3, 4, 5])

        random_state = np.random.RandomState(0)
        observation, factor = ds[32 * 32 * 40 + 2 * 32 * 32 + 3 * 32 + 4]
        self.assertTupleEqual(factor.shape, (4,))
        self.assertTupleEqual(observation.shape, (1, 64, 64))
        self.assertListEqual(factor.tolist(), [1, 2, 3, 4])

if __name__ == '__main__':
    unittest.main()
