import unittest

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from . import dsprites, translation
import numpy as np


def mse(img, target):
    return np.sum(np.square(img - target))


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
        stride = 1
        ds = translation.Translation(stride, (2, 8, 1))

        rand_seed = np.random.RandomState(0)
        factors = np.zeros((16, 2), np.int)
        factors[:, 0] = np.arange(16)
        obs = ds.sample_observations_from_factors(factors, rand_seed)
        mean = np.mean(obs, 0, keepdims=True)
        print(0, mse(obs, mean))

        factors = np.zeros((16, 2), np.int)
        factors[:, 1] = np.arange(16)
        obs = ds.sample_observations_from_factors(factors, rand_seed)
        mean = np.mean(obs, 0, keepdims=True)
        print(1, mse(obs, mean))


if __name__ == '__main__':
    unittest.main()
