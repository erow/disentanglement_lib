import unittest

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from . import dsprites, translation
import numpy as np


class DatasetTestCase(unittest.TestCase):
    def test_dsprites(self):
        ds = dsprites.DSprites([3])
        dl = DataLoader(ds, batch_size=64)
        imgs, labels = next(iter(dl))
        img = imgs[::4].mean(dim=[0, 2])
        plt.imshow(img[:, :])
        plt.show()
        print(labels)

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
