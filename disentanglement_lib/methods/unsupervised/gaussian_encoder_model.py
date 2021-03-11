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

"""Defines a common interface for Gaussian encoder based models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.methods.shared import architectures  # pylint: disable=unused-import

import os

import torch
import gin
from torch import nn
import torch.nn.functional as F


class GaussianModel:
    """Abstract base class of a Gaussian encoder model."""

    def reconstruct(self, images):
        mu, logvar = self.encode(images)
        return self.decode(mu)

    # def encode(self, images):
    #     raise NotImplementedError
    #
    # def decode(self, latent):
    #     raise NotImplementedError

    def sample_from_latent_distribution(self, z_mean, z_logvar):
        """Samples from the Gaussian distribution defined by z_mean and z_logvar."""
        e = torch.randn_like(z_mean)
        return torch.add(
            z_mean,
            torch.exp(z_logvar / 2) * e,
        )

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z
