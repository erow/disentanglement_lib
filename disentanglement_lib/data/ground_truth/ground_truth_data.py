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

"""Abstract class for data sets that are two-step generative models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


class GroundTruthData(object):
    """Abstract class for data sets that are two-step generative models."""

    @property
    def num_factors(self):
        raise NotImplementedError()

    @property
    def factors_num_values(self):
        raise NotImplementedError()

    @property
    def observation_shape(self):
        raise NotImplementedError()

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        raise NotImplementedError()

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError()

    def sample(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)

    def sample_observations(self, num, random_state):
        """Sample a batch of observations X."""
        return self.sample(num, random_state)[1]

    #  Compatible for torch DataSet
    def __len__(self):
        return np.prod(self.factors_num_values)

    def latent_factor(self, index):
        """Get a latent factor from index."""
        factors = np.zeros(
            shape=(1, self.num_factors), dtype=np.int64)
        factors_num_values = self.factors_num_values
        for pos, i in enumerate(factors_num_values):
            num = np.prod(factors_num_values[pos + 1:])
            factors[0, pos] = index // num
            index %= num
        return factors

    def __getitem__(self, item):
        factors = self.latent_factor(item)
        observations = self.sample_observations_from_factors(factors, np.random.RandomState(0))
        return observations.transpose((0, 3, 1, 2))[0], factors[0]

def sample_factor(ds:GroundTruthData):
    factor = np.array([np.random.randint(i) for i in ds.factors_num_values])
    return factor

def action(ds:GroundTruthData,factor,dim):
    rand_seed = np.random.RandomState()
    action_len = ds.factors_num_values[dim]
    factors = np.repeat(factor.reshape(1,-1),action_len,0)
    factors[:, dim] = np.arange(action_len)
    return ds.sample_observations_from_factors(factors, rand_seed)

class RandomAction(object):
    def __init__(self, ground_truth_data: GroundTruthData,
                 factor_index):
        super().__init__()
        self.data = ground_truth_data
        self.action_index = factor_index
        self.factor = sample_factor(self.data)

    def __len__(self):
        return self.data.factors_num_values[self.action_index]

    def __getitem__(self, item):
        factor = self.factor.copy()
        factor[self.action_index]=item

        obs = self.data.sample_observations_from_factors(factor.reshape(*([1]+list(factor.shape))),
                            np.random.RandomState(0))
        obs = obs.transpose(0,3,1,2)
        return obs.squeeze(0),factor

