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

import cv2

from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
import gin
from six.moves import range


def to_honeycomb(x):
    x1 = np.zeros(x.shape)
    x1[:, 0] = x[:, 0] + (x[:, 1] % 2) * 0.5
    x1[:, 1] = x[:, 1] / 2 * np.sqrt(3)
    return x1


@gin.configurable("translation")
class Translation(ground_truth_data.GroundTruthData):
    """Translation dataset.
    """

    def __init__(self, pos_type: int, radius=10):
        # By default, all factors (including shape) are considered ground truth
        # factors.
        factors = np.zeros((22 * 22, 2))
        factors[:, 0] = np.arange(22 * 22) // 22
        factors[:, 1] = np.arange(22 * 22) % 22
        factors = factors
        if pos_type == 0:
            pos = factors * 2  # cartesian
        elif pos_type == 1:
            pos = to_honeycomb(factors) * 2
        elif pos_type == 2:
            r = 1 + factors[:, 0] / 22 * 20
            theta = factors[:, 1] / 22 * 2 * np.pi
            pos = np.zeros((22 * 22, 2))
            pos[:, 1] = r * np.cos(theta) + 32
            pos[:, 0] = r * np.sin(theta) + 32
        else:
            raise NotImplementedError()
        self.data_shape = [64, 64, 1]
        self.factor_sizes = [22, 22]
        self.pos = pos

        self.latent_factor_indices = np.zeros(self.factor_sizes, dtype=np.int)
        for i in range(self.factor_sizes[0]):
            self.latent_factor_indices[i] = self.factor_sizes[0] * i + np.arange(self.factor_sizes[1])

        self.data = np.zeros([len(self)] + self.data_shape, dtype=np.float32)

        index = 0

        for i in range(self.factor_sizes[0]):
            for j in range(self.factor_sizes[1]):
                img = np.zeros(self.data_shape)
                cv2.circle(img, (10, 10), radius, (1.0,), -1)
                M = np.float32([[1, 0, pos[index, 0]], [0, 1, pos[index, 1]]])
                self.data[index, :, :, 0] = cv2.warpAffine(img, M, (64, 64))
                index += 1

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
