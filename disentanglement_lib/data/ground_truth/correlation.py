from disentanglement_lib.data.ground_truth import ground_truth_data
import cv2
import numpy as np


class Correlation(ground_truth_data.GroundTruthData):
    @property
    def num_factors(self):
        return 3

    @property
    def factors_num_values(self):
        return [32, 32, 32]

    @property
    def observation_shape(self):
        return [64, 64, 1]

    def __len__(self):
        return 32 * 32

    def sample_factors(self, num, random_state):
        pos = np.random.randint([32, 32], size=(num, 2))
        p = pos - np.array([[16, 16]])
        theta = np.arctan2(p[:, 1], p[:, 0]) / np.pi * 180 + 180
        theta_r = np.random.randint(60, size=(num,)) - 30
        theta = (theta + theta_r + 360) % 360
        t = np.digitize(theta, self.angles).reshape(-1, 1)
        return np.concatenate([t, pos], 1)

    def latent_factor(self, index):
        """Get a latent factor from index."""

        x = index // 32
        y = index % 32
        theta = np.arctan2(y - 16, x - 16) / np.pi * 180 + 180
        t = np.digitize(theta, self.angles)
        return np.array([(t, x, y)], dtype=np.int64)

    def sample_observations_from_factors(self, factors, random_state):
        pos = factors[:, 1:]
        theta = self.angles[factors[:, 0] - 1] - 90
        imgs = np.zeros((len(factors), 64, 64, 1))
        for i in range(len(factors)):
            matRotate = cv2.getRotationMatrix2D((5, 5), theta[i], 1)
            dst = cv2.warpAffine(self.img.copy(), matRotate, (11, 11))
            imgs[i,
            10 + pos[i, 0]:21 + pos[i, 0],
            10 + pos[i, 1]:21 + pos[i, 1],
            0] = dst
        return imgs

    def __init__(self, random_seed=None):
        self.random_state = np.random.RandomState(random_seed)
        img = np.zeros((11, 11))
        self.img = cv2.fillPoly(img,
                                np.array([[[0, 2], [10, 5], [0, 8]]], dtype=np.int),
                                1)
        self.angles = np.linspace(0, 360, 33)
