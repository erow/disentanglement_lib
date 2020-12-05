import gin
import numpy as np


@gin.configurable('mean_representation')
def mean_representation(distribute):
    """Computes representation vector for input images."""
    mean, std = distribute
    return mean


@gin.configurable('sample_representation')
def sample_representation(distribute):
    """Computes representation vector for input images."""
    mean, std = distribute
    return np.random.normal(0, 1, size=mean.shape) * std + mean
