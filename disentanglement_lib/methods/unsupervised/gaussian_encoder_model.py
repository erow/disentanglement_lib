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

import os

import torch
import gin
from torch import nn
import torch.nn.functional as F


class Savable(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.arags = args
        self.kwargs = kwargs

    def save(self, model_dir, filename='ckp.pth'):
        ckp_path = os.path.join(model_dir, filename)
        ckp_dict = dict()
        ckp_dict['name'] = self.__class__.__name__
        ckp_dict['args'] = self.arags
        ckp_dict['kwargs'] = self.kwargs
        ckp_dict['model'] = self.state_dict()
        torch.save(ckp_dict, ckp_path)


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def load(cls, model_dir, filename='ckp.pth'):
    ckp_path = os.path.join(model_dir, filename)
    ckp_dict = torch.load(ckp_path)
    for sc in all_subclasses(cls):
        if sc.__name__ == ckp_dict['name']:
            model = sc(*ckp_dict['args'], **ckp_dict['kwargs'])
            model.load_state_dict(ckp_dict['model'])
            return model
    raise LookupError('Unexpected model:%s' % ckp_dict['name'])



class GaussianModel(Savable):
    """Abstract base class of a Gaussian encoder model."""
    encode: callable(torch.Tensor)
    decode: callable(torch.Tensor)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.arags = args
        self.kwargs = kwargs

    def forward(self, images):
        mu, logvar = self.encode(images)
        return self.decode(mu)

    def model_fn(self, features, labels):
        """Compatible model function used for training/evaluation."""
        raise NotImplementedError()

    def sample_from_latent_distribution(self, z_mean, z_logvar):
        """Samples from the Gaussian distribution defined by z_mean and z_logvar."""
        e = torch.randn_like(z_mean)
        return torch.add(
            z_mean,
            torch.exp(z_logvar / 2) * e,
        )
