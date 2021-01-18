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

"""Shapes3D data set."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
from six.moves import range
import h5py

SHAPES3D_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "3dshapes",
    "3dshapes.h5"
)


class Shapes3D(ground_truth_data.GroundTruthData):
    """Shapes3D dataset.

    The data set was originally introduced in "Disentangling by Factorising".

    The ground-truth factors of variation are:
    0 - floor color (10 different values)
    1 - wall color (10 different values)
    2 - object color (10 different values)
    3 - object size (8 different values)
    4 - object type (4 different values)
    5 - azimuth (15 different values)
    """
    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                         'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                              'scale': 8, 'shape': 4, 'orientation': 15}
    def __init__(self):
        data = h5py.File(SHAPES3D_PATH, 'r')

        images = np.array(data["images"])
        labels = np.array(data["labels"])

        self.factor_sizes = [10, 10, 10, 8, 4, 15]
        n_samples = np.prod(self.factor_sizes)
        self.images = images.astype(np.float32) / 255.
        features = labels.reshape([n_samples, 6])
        self.latent_factor_indices = list(range(6))
        self.num_total_factors = features.shape[1]
        self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                        self.latent_factor_indices)
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
            self.factor_sizes)

    def get_from_index(self, indices):
        ims = []
        for ind in indices:
            im = self.images[ind]
            im = np.asarray(im)
            ims.append(im)
        ims = np.stack(ims, axis=0)
        return ims.astype(np.float32) / 255.

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors


    @property
    def factors_num_values(self):
        return self.factor_sizes


    @property
    def observation_shape(self):
        return [64, 64, 3]


    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)


    def sample_observations_from_factors(self, factors, random_state):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return self.images[indices]
