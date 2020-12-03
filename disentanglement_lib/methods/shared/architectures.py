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

"""Library of commonly used architectures and reconstruction losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from torch import nn
import gin


@gin.configurable("encoder", whitelist=["num_latent", "encoder_fn"])
def make_gaussian_encoder(input_shape,
                          num_latent=gin.REQUIRED,
                          encoder_fn=gin.REQUIRED):
    """Gin wrapper to create and apply a Gaussian encoder configurable with gin.

    This is a separate function so that several different models (such as
    BetaVAE and FactorVAE) can call this function while the gin binding always
    stays 'encoder.(...)'. This makes it easier to configure models and parse
    the results files.

    Args:
      num_latent: Integer with dimensionality of latent space.
      encoder_fn: Function that that takes the arguments (input_tensor,
        num_latent, is_training) and returns the tuple (means, log_vars) with the
        encoder means and log variances.

    Returns:
      Tuple (means, log_vars) with the encoder means and log variances.
    """
    return encoder_fn(input_shape=input_shape, num_latent=num_latent)


@gin.configurable("decoder", whitelist=["decoder_fn"])
def make_decoder(num_latent, output_shape,
                 decoder_fn=gin.REQUIRED):
    """Gin wrapper to create and apply a decoder configurable with gin.

    This is a separate function so that several different models (such as
    BetaVAE and FactorVAE) can call this function while the gin binding always
    stays 'decoder.(...)'. This makes it easier to configure models and parse
    the results files.

    Args:
      output_shape: Tuple with the output shape of the observations to be
        generated.
      decoder_fn: Function that that takes the arguments (input_tensor,
        output_shape, is_training) and returns the decoded observations.

    Returns:
      Tensor of decoded observations.
    """
    return decoder_fn(num_latent=num_latent, output_shape=output_shape)


@gin.configurable("discriminator", whitelist=["discriminator_fn"])
def make_discriminator(num_latent,
                       discriminator_fn=gin.REQUIRED):
    """Gin wrapper to create and apply a discriminator configurable with gin.

    This is a separate function so that several different models (such as
    FactorVAE) can potentially call this function while the gin binding always
    stays 'discriminator.(...)'. This makes it easier to configure models and
    parse the results files.

    Args:
        num_latent: Number of the latent variables.
      discriminator_fn: Function that that takes the arguments
      (input_tensor, is_training) and returns tuple of (logits, clipped_probs).

    Returns:
      Tuple of (logits, clipped_probs) tensors.
    """
    # logits, probs = discriminator_fn(num_latent)
    # clipped = torch.clamp(probs, 1e-6, 1 - 1e-6)
    return discriminator_fn(num_latent)


@gin.configurable("fc_encoder", whitelist=[])
class fc_encoder(nn.Module):
    """Fully connected encoder used in beta-VAE paper for the dSprites data.
    Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl).

    Args:
      num_latent: Number of latent variables to output.
    Returns:
      means: Output tensor of shape (batch_size, num_latent) with latent variable
        means.
      log_var: Output tensor of shape (batch_size, num_latent) with latent
        variable log variances.
    """

    def __init__(self, input_shape, num_latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape).item(), 1200), nn.ReLU(),
            nn.Linear(1200, 1200), nn.ReLU(),
            nn.Linear(1200, num_latent * 2)
        )
        self.num_latent = num_latent
        self.input_shape = input_shape

    def forward(self, input_tensor):
        x = self.net(input_tensor)
        means, log_var = torch.split(x, [self.num_latent] * 2, 1)
        return means, log_var


@gin.configurable("conv_encoder", whitelist=[])
class conv_encoder(nn.Module):
    """Convolutional encoder used in beta-VAE paper for the chairs data.

    Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Args:
      input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
        build encoder on.
      num_latent: Number of latent variables to output.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      means: Output tensor of shape (batch_size, num_latent) with latent variable
        means.
      log_var: Output tensor of shape (batch_size, num_latent) with latent
        variable log variances.
    """

    def __init__(self, input_shape, num_latent):
        super().__init__()
        self.num_latent = num_latent
        self.input_shape = input_shape

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (4, 4), stride=2, padding=1), nn.ReLU(),  # 32x32
            nn.Conv2d(32, 32, (4, 4), stride=2, padding=1), nn.ReLU(),  # 16
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1), nn.ReLU(),  # 8
            nn.Conv2d(64, 64, (4, 4), stride=2, padding=1), nn.ReLU(),  # 4
            nn.Flatten(),
            nn.Linear(4 * 4 * 64, 256), nn.ReLU(),
            nn.Linear(256, num_latent * 2)
        )

    def forward(self, input_tensor):
        x = self.net(input_tensor)
        means, log_var = torch.split(x, [self.num_latent] * 2, 1)
        return means, log_var


@gin.configurable("fc_decoder", whitelist=[])
class fc_decoder(nn.Module):
    def __init__(self, num_latent, output_shape):
        super().__init__()
        self.num_latent = num_latent
        self.output_shape = output_shape
        self.net = nn.Sequential(
            nn.Linear(num_latent, 1200), nn.ReLU(),
            nn.Linear(1200, 1200), nn.ReLU(),
            nn.Linear(1200, 1200), nn.ReLU(),
            nn.Linear(1200, np.prod(output_shape).item())
        )

    def forward(self, latent_tensor):
        x = self.net(latent_tensor)
        return torch.reshape(x, shape=[-1] + self.output_shape)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer('shape', torch.LongTensor(shape))

    def forward(self, x):
        return x.view(*self.shape)


@gin.configurable("deconv_decoder", whitelist=[])
class deconv_decoder(nn.Module):
    """Convolutional decoder used in beta-VAE paper for the chairs data.

    Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Args:
      latent_tensor: Input tensor of shape (batch_size,) to connect decoder to.
      output_shape: Shape of the data.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      Output tensor of shape (batch_size, 64, 64, num_channels) with the [0,1]
        pixel intensities.
    """

    def __init__(self, num_latent, output_shape):
        super().__init__()
        self.num_latent = num_latent
        self.output_shape = output_shape
        self.net = nn.Sequential(
            nn.Linear(num_latent, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            View([-1, 64, 4, 4]),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(),  # 8
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),  # 16
            nn.ConvTranspose2d(32, 4, 4, stride=2, padding=1), nn.ReLU(),  # 32
            nn.ConvTranspose2d(4, output_shape[0], 4, stride=2, padding=1)  # 64
        )

    def forward(self, latent_tensor):
        x = self.net(latent_tensor)
        return torch.reshape(x, shape=[-1] + self.output_shape)


@gin.configurable("fc_discriminator", whitelist=[])
class fc_discriminator(nn.Module):
    """Fully connected discriminator used in FactorVAE paper for all datasets.

    Based on Appendix A page 11 "Disentangling by Factorizing"
    (https://arxiv.org/pdf/1802.05983.pdf)

    Args:
      input_tensor: Input tensor of shape (None, num_latents) to build
        discriminator on.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      logits: Output tensor of shape (batch_size, 2) with logits from
        discriminator.
      probs: Output tensor of shape (batch_size, 2) with probabilities from
        discriminator.
    """

    def __init__(self, num_latent):
        super().__init__()
        self.num_latent = num_latent
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_latent, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 2)
        )

    def forward(self, input_tensor):
        logits = self.net(input_tensor)
        probs = torch.softmax(logits, 1)
        clipped = torch.clamp(probs, 1e-6, 1 - 1e-6)
        return logits, clipped


@gin.configurable("test_encoder", whitelist=[])
class test_encoder(nn.Module):
    """Fully connected encoder used in beta-VAE paper for the dSprites data.
    Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl).

    Args:
      num_latent: Number of latent variables to output.
    Returns:
      means: Output tensor of shape (batch_size, num_latent) with latent variable
        means.
      log_var: Output tensor of shape (batch_size, num_latent) with latent
        variable log variances.
    """

    def __init__(self, input_shape, num_latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape).item(), 1200), nn.ReLU(),
            nn.Linear(1200, num_latent * 2)
        )
        self.num_latent = num_latent

    def forward(self, input_tensor):
        x = self.net(input_tensor)
        means, log_var = torch.split(x, [self.num_latent] * 2, 1)
        return means, log_var


@gin.configurable("test_decoder", whitelist=[])
class test_decoder(nn.Module):
    """Fully connected encoder used in beta-VAE paper for the dSprites data.

    Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Args:
      latent_tensor: Input tensor to connect decoder to.
      output_shape: Shape of the data.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      Output tensor of shape (None, 64, 64, num_channels) with the [0,1] pixel
      intensities.
    """

    def __init__(self, num_latent, output_shape):
        super().__init__()
        self.num_latent = num_latent
        self.output_shape = output_shape
        self.net = nn.Sequential(
            nn.Linear(num_latent, 1200), nn.ReLU(),
            nn.Linear(1200, np.prod(output_shape).item())
        )

    def forward(self, latent_tensor):
        x = self.net(latent_tensor)
        return torch.reshape(x, shape=[-1] + self.output_shape)
