# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""translation dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
import gin
from six.moves import range


@gin.configurable("translation")
class Translation(ground_truth_data.GroundTruthData):
    """Translation dataset.
    The ground-truth factors of variation are (in the default setting):
    0 - position x (32 different values)
    1 - position y (32 different values)
    """

    def __init__(self, stride=1, img_size=(4, 4, 1)):
        # By default, all factors (including shape) are considered ground truth
        # factors.
        self.data_shape = [64, 64, 1]
        self.factor_sizes = [16, 16]

        self.latent_factor_indices = np.zeros(self.factor_sizes, dtype=np.int)
        for i in range(self.factor_sizes[0]):
            self.latent_factor_indices[i] = self.factor_sizes[0] * i + np.arange(self.factor_sizes[1])

        self.data = np.zeros([len(self)] + self.data_shape, dtype=np.float32)
        img = np.ones(img_size, dtype=np.float32)
        for i in range(self.factor_sizes[0]):
            for j in range(self.factor_sizes[1]):
                original = self.data[i * self.factor_sizes[1] + j]
                set_img(original, i * stride, j * stride, img)

    @property
    def num_factors(self):
        return len(self.factor_sizes)

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return self.data_shape

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        factors = [random_state.randint(i, size=num) for i in self.factors_num_values]
        return np.stack(factors, axis=1)

    def sample_observations_from_factors(self, factors, random_state):
        indices = self.latent_factor_indices[factors[:, 0], factors[:, 1]]
        return self.data[indices]

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)


def set_img(original, i, j, img):
    h, w, _ = img.shape
    if i + h > 64:
        original[i:i + h, j:j + w] = img[:min(-1, 64 - i - h)]
    else:
        original[i:i + h, j:j + w] = img
    return original
