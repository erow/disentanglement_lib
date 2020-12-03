import unittest
from . import dsprites
import numpy as np


class DatasetTestCase(unittest.TestCase):
    def test_dsprites(self):
        ds = dsprites.DSprites()

        random_state = np.random.RandomState(0)
        factors, observations = ds.sample(10, random_state)
        self.assertTupleEqual(factors.shape, (10, 6))
        self.assertTupleEqual(observations.shape, (10, 64, 64, 1))


if __name__ == '__main__':
    unittest.main()
