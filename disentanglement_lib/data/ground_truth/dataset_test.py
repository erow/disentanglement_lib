import unittest
from . import dsprites, translation
import numpy as np


class DatasetTestCase(unittest.TestCase):
    def test_dsprites(self):
        ds = dsprites.DSprites([2, 3, 4, 5])

        random_state = np.random.RandomState(0)
        observation, factor = ds[32 * 32 * 40 + 2 * 32 * 32 + 3 * 32 + 4]
        self.assertTupleEqual(factor.shape, (4,))
        self.assertTupleEqual(observation.shape, (1, 64, 64))
        self.assertListEqual(factor.tolist(), [1, 2, 3, 4])

    def test_translation(self):
        stride = 4
        ds = translation.Translation(stride, (10, 4, 1))
        random_state = np.random.RandomState(0)
        factors = ds.sample_factors(100, random_state)
        obs = ds.sample_observations_from_factors(factors, random_state)

        for i in range(len(factors)):
            x, y = factors[i]
            self.assertEqual(obs[i, x * stride:x * stride + 4, y * stride:y * stride + 4].sum(), 16)
            # self.assertEqual(obs[i].sum(), 40)



if __name__ == '__main__':
    unittest.main()
