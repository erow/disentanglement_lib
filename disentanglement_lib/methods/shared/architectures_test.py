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

"""Tests for the architectures.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
from absl.testing import absltest
from disentanglement_lib.methods.shared import architectures
import numpy as np
import torch


class ArchitecturesTest(parameterized.TestCase):

    @parameterized.named_parameters(
        ('fc_encoder', architectures.fc_encoder),
        ('conv_encoder', architectures.conv_encoder),
    )
    def test_encoder(self, encoder_f):
        minibatch = np.ones(shape=(10, 1, 64, 64), dtype=np.float32)
        input_tensor = torch.Tensor(minibatch)
        encoder = encoder_f([1, 64, 64], 10)
        latent_mean, latent_logvar = encoder(input_tensor)

    @parameterized.named_parameters(
        ('fc_decoder', architectures.fc_decoder),
        ('deconv_decoder', architectures.deconv_decoder),
    )
    def test_decoder(self, decoder_f):
        latent_variable = np.ones(shape=(10, 15), dtype=np.float32)
        input_tensor = torch.Tensor(latent_variable)
        decoder = decoder_f(15, [1, 64, 64])
        images = decoder(input_tensor)

    @parameterized.named_parameters(
        ('fc_discriminator', architectures.fc_discriminator),
    )
    def test_discriminator(self, discriminator_f):
        images = np.ones(shape=(32, 10), dtype=np.float32)
        input_tensor = torch.Tensor(images)
        discriminator = discriminator_f(10)
        logits, probs = discriminator(input_tensor)


if __name__ == '__main__':
    absltest.main()
