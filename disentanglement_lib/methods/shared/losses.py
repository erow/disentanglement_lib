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

"""Library of commonly used losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
import gin


@gin.configurable("bernoulli_loss", whitelist=["subtract_true_image_entropy"])
def bernoulli_loss(true_images,
                   reconstructed_images,
                   activation,
                   subtract_true_image_entropy=False):
    """Computes the Bernoulli loss. A vector on the batch."""
    img_size = np.prod(true_images.shape[1:])
    reconstructed_images = reconstructed_images.reshape(-1, img_size)
    true_images = true_images.reshape(-1, img_size)

    # Because true images are not binary, the lower bound in the xent is not zero:
    # the lower bound in the xent is the entropy of the true images.？
    if subtract_true_image_entropy:
        dist = torch.distributions.Bernoulli(
            probs=torch.clamp(true_images, 1e-6, 1 - 1e-6))

        loss_lower_bound = dist.entropy().sum(1)
    else:
        loss_lower_bound = 0

    if activation == "logits":
        loss = F.binary_cross_entropy_with_logits(reconstructed_images,
                                                  true_images,
                                                  reduction="none").sum(1)
    elif activation == "tanh":
        reconstructed_images = torch.clamp(
            F.tanh(reconstructed_images) / 2 + 0.5, 1e-6, 1 - 1e-6)
        loss = -torch.sum(
            true_images * torch.log(reconstructed_images) +
            (1 - true_images) * torch.log(1 - reconstructed_images),
            dim=1)
    else:
        raise NotImplementedError("Activation not supported.")

    return loss - loss_lower_bound


@gin.configurable("l2_loss", whitelist=[])
def l2_loss(true_images, reconstructed_images, activation):
    """Computes the l2 loss."""
    if activation == "logits":
        return torch.sum(
            torch.square(true_images - torch.sigmoid(reconstructed_images)), [1, 2, 3])
    elif activation == "tanh":
        reconstructed_images = torch.tanh(reconstructed_images) / 2 + 0.5
        return torch.sum(
            torch.square(true_images - reconstructed_images), [1, 2, 3])
    else:
        raise NotImplementedError("Activation not supported.")


@gin.configurable(
    "reconstruction_loss", deneylist=["true_images", "reconstructed_images"])
def make_reconstruction_loss(true_images,
                             reconstructed_images,
                             loss_fn=gin.REQUIRED,
                             activation="logits"):
    """Wrapper that creates reconstruction loss."""
    per_sample_loss = loss_fn(true_images, reconstructed_images, activation)
    return per_sample_loss


def kl_normal_loss(mean, logvar, mean_dim=None):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)
    """
    if mean_dim is None:
        mean_dim = [0]
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=mean_dim)
    return latent_kl
