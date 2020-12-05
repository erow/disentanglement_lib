import types

from disentanglement_lib.data.ground_truth.ground_truth_data import GroundTruthData
import numpy as np


class SpecialTuple(tuple):
    def __len__(self):
        return len(next(iter(self)))

    def __getitem__(self, key):
        return [a[key] for a in self]


def concat_representation(original_data: GroundTruthData, variables):
    original_data.mean = variables['mean']
    original_data.std = variables['std']
    original_data.factor = variables['factor']

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        distribution = SpecialTuple((
            self.mean[indices].astype(np.float32),
            self.std[indices].astype(np.float32)
        ))
        return distribution

    original_data.sample_observations_from_factors = types.MethodType(sample_observations_from_factors, original_data)
    return original_data
