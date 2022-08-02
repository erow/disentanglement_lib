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

import os

import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock
import gin



@gin.configurable("fc_encoder", allowlist=[])
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
        self.K = num_latent
        self.num_latent = num_latent
        self.input_shape = input_shape

    def forward(self, input_tensor):
        x = self.net(input_tensor)
        means, log_var = torch.split(x, [self.num_latent] * 2, 1)
        return means, log_var


@gin.configurable("conv_encoder", allowlist=[])
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

    def __init__(self, input_shape, num_latent, base_channel=32):
        super().__init__()
        self.num_latent = num_latent
        self.input_shape = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], base_channel, (4, 4), stride=2, padding=1), nn.ReLU(True),  # 32x32
            nn.Conv2d(base_channel, base_channel, (4, 4), stride=2, padding=1), nn.ReLU(True),    # 16
            nn.Conv2d(base_channel, base_channel * 2, (4, 4), stride=2, padding=1), nn.ReLU(True),# 8
            nn.Conv2d(base_channel * 2, base_channel * 2, (4, 4), stride=2, padding=1), nn.ReLU(True),  # 4
            nn.Conv2d(base_channel * 2, base_channel * 8, (4, 4)), nn.ReLU(True),           # 1x1
            nn.Flatten(),
            nn.Linear(1 * 1 * base_channel * 8, 128), nn.ReLU(),
            nn.Linear(128, num_latent * 2)
        )
        self.K = num_latent

    def forward(self, input_tensor):
        x = self.net(input_tensor)
        means, log_var = torch.split(x, [self.num_latent] * 2, 1)
        return means, log_var


@gin.configurable("fc_decoder", allowlist=[])
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


@gin.configurable("deconv_decoder", allowlist=[])
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
            nn.Linear(num_latent, 256), nn.ReLU(True),
            nn.Linear(256, 1024), nn.ReLU(True),
            View([-1, 64, 4, 4]),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(True),  # 8
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(True),  # 16
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(True),  # 32
            nn.ConvTranspose2d(32, output_shape[0], 4, stride=2, padding=1)  # 64
        )

    def forward(self, latent_tensor):
        x = self.net(latent_tensor)
        return torch.reshape(x, shape=[-1] + self.output_shape)


@gin.configurable("fc_discriminator", allowlist=[])
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
            nn.Linear(num_latent, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 2)
        )

    def forward(self, input_tensor):
        logits = self.net(input_tensor)
        probs = torch.softmax(logits, 1)
        clipped = torch.clamp(probs, 1e-6, 1 - 1e-6)
        return logits, clipped


@gin.configurable("test_encoder", allowlist=[])
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


@gin.configurable("test_decoder", allowlist=[])
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


class lite_decoder(nn.Module):
    def __init__(self, num_latent, output_shape):
        super().__init__()
        self.num_latent = num_latent
        self.output_shape = output_shape
        self.net = nn.Sequential(
            nn.Linear(num_latent, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            View([-1, 16, 4, 4]),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1), nn.ReLU(),  # 8
            nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),  # 16
            nn.ConvTranspose2d(32, 4, 4, stride=2, padding=1), nn.ReLU(),  # 32
            nn.ConvTranspose2d(4, output_shape[0], 4, stride=2, padding=1)  # 64
        )

    def forward(self, latent_tensor):
        x = self.net(latent_tensor)
        return torch.reshape(x, shape=[-1] + self.output_shape)
@gin.configurable("deep_decoder", allowlist=[])
class DeepConvDecoder(nn.Module):
    def __init__(self, num_latent, output_shape, width=256):
        super().__init__()
        self.output_shape = output_shape
        self.num_latent = num_latent
        self.width = width

        def block(in_feat, out_feat, size):
            layers = [BasicBlock(in_feat, in_feat),
                      BasicBlock(in_feat, in_feat),
                      nn.UpsamplingBilinear2d(size),
                      nn.Conv2d(in_feat, out_feat, kernel_size=(1, 1))]
            return layers

        self.convert_2d = nn.Sequential(
            nn.Linear(num_latent, width * 2), nn.ReLU(0.02),
            nn.Linear(width * 2, 4 * 4 * width),
        )

        c, h, w = output_shape
        conv_blocks = []

        ch, cw = 4, 4
        c_in = width
        while min(h / ch, w / cw) > 2:
            conv_blocks += block(c_in, c_in // 2, [ch * 2, cw * 2])
            c_in = c_in // 2
            ch, cw = ch * 2, cw * 2
        conv_blocks += (block(c_in, c_in // 2, [h, w]))  # same size
        c_in = c_in // 2

        conv_blocks = conv_blocks + [
            nn.ReLU(0.02),
            nn.Conv2d(c_in, c, (5, 5), padding=2)
        ]
        # same channel
        self.conv = nn.Sequential(*conv_blocks)

    def forward(self, z):
        img = self.convert_2d(z).view(z.size(0), self.width, 4, 4)
        img = self.conv(img)
        return img


@gin.configurable("deep_encoder", allowlist=[])
class DeepConvEncoder(nn.Module):
    def __init__(self, input_shape, num_latent, width=256):
        super().__init__()
        self.input_shape = input_shape
        self.num_latent = num_latent
        self.width = width

        def block(in_feat, out_feat, size):
            layers = [
                nn.Conv2d(in_feat, out_feat, kernel_size=1),
                BasicBlock(out_feat, out_feat),
                nn.AvgPool2d(2, 2)
            ]
            return layers

        self.convert_1d = nn.Sequential(
            nn.Linear(4 * 4 * width, width * 2), nn.ReLU(0.02),
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, num_latent * 2)
        )

        c, h, w = input_shape
        conv_blocks = [
            nn.Conv2d(c, 64, 5, 2, padding=2), nn.ReLU(0.02)
        ]

        ch, cw = h // 2, w // 2
        c_in = 64
        while min(ch // 4, cw // 4) > 2:
            t_in = min(width, c_in * 2)
            conv_blocks += block(c_in, t_in, [ch // 2, cw // 2])
            c_in = t_in
            ch, cw = ch // 2, cw // 2

        conv_blocks += [
            nn.Conv2d(c_in, width, kernel_size=1),
            BasicBlock(width, width),
            nn.AdaptiveAvgPool2d([4, 4]),
        ]

        self.conv = nn.Sequential(*conv_blocks)

    def forward(self, x):
        img = self.conv(x)
        feature = self.convert_1d(img.reshape(x.size(0), 4 * 4 * self.width))
        mu, logvar = torch.split(feature, [self.num_latent] * 2, 1)
        return mu, logvar


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.ReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
    
@gin.configurable("frac_encoder", allowlist=[])
class FracEncoder(nn.Module):
    def __init__(self,
                input_shape,
                num_latent, G=5):
        super().__init__()
        self.G=G
        self.num_latent = num_latent
        assert num_latent % G==0
        self.K=num_latent//G
        self.sub_encoders = nn.Sequential(
            *[conv_encoder(input_shape, self.K,8) for _ in range(G)])
        self.projs = nn.Sequential(
            *[Projection(self.K) for _ in range(G)])
        self.set_stage(0)
        
    def set_stage(self, i):
        if i>= self.G:
            i = self.G
        self.stage = i
        for i in range(min(self.stage, self.G)):
            self.sub_encoders[i].requires_grad_(False)
            
    def forward(self, x):
        mus, logvars = [], []
        for i in range(self.G):
            f = self.sub_encoders[i]
            if i<self.stage:
                mu, logvar = f(x)
                mu, logvar = self.projs[i](mu.data,logvar.data)
            elif i ==self.stage:
                mu, logvar = f(x)
            else:
                mu = torch.zeros_like(mu)
                logvar = torch.zeros_like(mu)
            mus.append(mu)
            logvars.append(logvar)
        
        mu = torch.cat(mus,1)
        logvar = torch.cat(logvars,1)
        return mu, logvar

@gin.configurable("discriminator", allowlist=["discriminator_fn"])
def make_discriminator(num_latent,
                       discriminator_fn=fc_discriminator):
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

class Projection(nn.Module):
    def __init__(self,num_latent) -> None:
        super().__init__()
        self.num_latent = num_latent  
        self.W1 = nn.Parameter(torch.zeros(1,num_latent))             
        self.W2 = nn.Parameter(torch.zeros(1,num_latent))

    def forward(self, mu, logvar):
        z_mean1 = mu * self.W1.exp()
        z_logvar1 = logvar * self.W2.exp()
        return z_mean1, z_logvar1
    
    def extra_repr(self):
        return "W1: " + str((self.W1.exp()).data.cpu().numpy().round(2))+",\n"+\
        "W2: " + str(self.W2.exp().data.cpu().numpy().round(2))