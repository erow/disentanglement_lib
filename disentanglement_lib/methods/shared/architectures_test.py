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
    
    @parameterized.parameters(0,0.1,1)
    def test_deft(self,gamma):
        encoder1 = architectures.FracEncoder([1,64,64],10,gamma)
        encoder2 = architectures.FracEncoder([1,64,64],10,1)
        encoder1.stage=1
        encoder1.load_state_dict(encoder2.state_dict())

        x1 = torch.rand([1,1,64,64])
        x2 = x1.clone()
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        e = torch.randn(1,10)

        # forward
        mu, logvar = encoder1(x1)
        z = e*logvar.exp() + mu
        kl = (mu**2 + logvar.exp()-logvar).mean()
        ((z*e).sum()+kl).backward()

        mu, logvar = encoder2(x2)
        z = e*logvar.exp() + mu
        kl = (mu**2 + logvar.exp()-logvar).mean()
        ((z*e).sum()+kl).backward()

        # check
        c1 = encoder1.encoders[0].net
        c2 = encoder2.encoders[0].net
        error = 0
        for p1,p2 in zip(c1.parameters(),c2.parameters()):
            error += (p1.grad-p2.grad*gamma).abs().sum()
        print(error)
        assert error<1e-5


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
