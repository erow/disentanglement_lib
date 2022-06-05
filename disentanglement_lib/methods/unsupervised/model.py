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

"""Library of losses for disentanglement learning.

Implementation of VAE based models for unsupervised learning of disentangled
representations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import math
import os
from argparse import ArgumentParser

from disentanglement_lib.methods.shared import architectures  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import losses  # pylint: disable=unused-import
from six.moves import range
from six.moves import zip
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import gin


def BernoulliH(p):
    p = torch.clamp(p, 1e-6, 1 - 1e-6)
    h = p * (p).log() + (1 - p) * (1 - p).log()
    return -h

def sample_from_latent_distribution(z_mean, z_logvar):
        """Samples from the Gaussian distribution defined by z_mean and z_logvar."""
        e = torch.randn_like(z_mean)
        return torch.add(
            z_mean,
            torch.exp(z_logvar / 2) * e,
        )



class Regularizer(nn.Module):
    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        raise NotImplementedError


def shuffle_codes(z):
    """Shuffles latent variables across the batch.

    Args:
      z: [batch_size, num_latent] representation.

    Returns:
      shuffled: [batch_size, num_latent] shuffled representation across the batch.
    """
    z_shuffle = []
    for i in range(z.shape[1]):
        idx = torch.randperm(z.shape[0], device=z.device)
        z_shuffle.append(z[idx, i])
    shuffled = torch.stack(z_shuffle, 1, )
    return shuffled


def compute_gaussian_kl(z_mean, z_logvar):
    """Compute KL divergence between input Gaussian and Standard Normal."""
    return 0.5 * torch.mean(
        torch.square(z_mean) + torch.exp(z_logvar) - z_logvar - 1, [0])


def make_metric_fn(*names):
    """Utility function to report torch.metrics in model functions."""

    def metric_fn(*args):
        return {name: torch.mean(vec) for name, vec in zip(names, args)}

    return metric_fn


@gin.configurable("vae")
class BetaVAE(Regularizer):
    """BetaVAE model."""

    def __init__(self, beta=1, **kwargs):
        """Creates a beta-VAE model.

        Implementing Eq. 4 of "beta-VAE: Learning Basic Visual Concepts with a
        Constrained Variational Framework"
        (https://openreview.net/forum?id=Sy2fzU9gl).

        Args:
          beta: Hyperparameter for the regularizer.

        Returns:
          model_fn: Model function for TPUEstimator.
        """
        super().__init__()
        self.beta = beta

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        kl_loss = kl.sum()
        return self.beta * kl_loss
    
    def extra_repr(self) -> str:
        return f"beta={self.beta}"


def anneal(c_max, step, iteration_threshold):
    """Anneal function for anneal_vae (https://arxiv.org/abs/1804.03599).

    Args:
      c_max: Maximum capacity.
      step: Current step.
      iteration_threshold: How many iterations to reach c_max.

    Returns:
      Capacity annealed linearly until c_max.
    """
    return min(c_max * 1.,
               c_max * 1. * (step) / iteration_threshold)


@gin.configurable("annealed_vae")
class AnnealedVAE(Regularizer):
    """AnnealedVAE model."""

    def __init__(self, gamma=gin.REQUIRED, c_max=gin.REQUIRED, iteration_threshold=gin.REQUIRED, **kwargs):
        """Creates an AnnealedVAE model.

        Implementing Eq. 8 of "Understanding disentangling in beta-VAE"
        (https://arxiv.org/abs/1804.03599).

        Args:
          gamma: Hyperparameter for the regularizer.
          c_max: Maximum capacity of the bottleneck.
          iteration_threshold: How many iterations to reach c_max.
        """
        super().__init__()
        self.gamma = gamma
        self.c_max = c_max
        self.iteration_threshold = iteration_threshold

        self.c = 0
        self.delta = c_max / iteration_threshold

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        c = min(self.c_max, self.c + self.delta)
        self.c += self.delta
        kl_loss = kl.sum()
        return self.gamma * torch.abs(kl_loss - c)


@gin.configurable("factor_vae")
class FactorVAE(Regularizer):
    """FactorVAE model."""

    def __init__(self, num_latent=gin.REQUIRED,gamma=gin.REQUIRED):
        """Creates a FactorVAE model.

        Implementing Eq. 2 of "Disentangling by Factorizing"
        (https://arxiv.org/pdf/1802.05983).

        Args:
          gamma: Hyperparameter for the regularizer.
        """
        super().__init__()
        self.gamma = gamma
        self.discriminator = architectures.make_discriminator(num_latent)
        self.opt = torch.optim.Adam(self.discriminator.parameters())

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        features, _ = data_batch
        data_shape = features.shape[1:]

        z_shuffle = shuffle_codes(z_sampled)

        logits_z, probs_z = self.discriminator(z_sampled.data)
        _, probs_z_shuffle = self.discriminator(z_shuffle.data)

        discr_loss = -torch.add(
            0.5 * torch.mean(torch.log(probs_z[:, 0])),
            0.5 * torch.mean(torch.log(probs_z_shuffle[:, 1])))
        self.opt.zero_grad()
        discr_loss.backward()
        self.opt.step()

        logits_z, probs_z = self.discriminator(z_sampled)
        _, probs_z_shuffle = self.discriminator(z_shuffle)

        # tc = E[log(p_real)-log(p_fake)] = E[logit_real - logit_fake]
        tc_loss_per_sample = logits_z[:, 0] - logits_z[:, 1]
        tc_loss = torch.mean(tc_loss_per_sample, dim=0)

        return self.gamma * tc_loss


def compute_covariance_z_mean(z_mean):
    """Computes the covariance of z_mean.

    Uses cov(z_mean) = E[z_mean*z_mean^T] - E[z_mean]E[z_mean]^T.

    Args:
      z_mean: Encoder mean, tensor of size [batch_size, num_latent].

    Returns:
      cov_z_mean: Covariance of encoder mean, tensor of size [num_latent,
        num_latent].
    """
    expectation_z_mean_z_mean_t = torch.mean(
        torch.unsqueeze(z_mean, 2) * torch.unsqueeze(z_mean, 1), dim=0)
    expectation_z_mean = torch.mean(z_mean, dim=0)
    cov_z_mean = torch.sub(
        expectation_z_mean_z_mean_t,
        torch.unsqueeze(expectation_z_mean, 1) * torch.unsqueeze(
            expectation_z_mean, 0))
    return cov_z_mean


def regularize_diag_off_diag_dip(covariance_matrix, lambda_od, lambda_d):
    """Compute on and off diagonal regularizers for DIP-VAE models.

    Penalize deviations of covariance_matrix from the identity matrix. Uses
    different weights for the deviations of the diagonal and off diagonal entries.

    Args:
      covariance_matrix: Tensor of size [num_latent, num_latent] to regularize.
      lambda_od: Weight of penalty for off diagonal elements.
      lambda_d: Weight of penalty for diagonal elements.

    Returns:
      dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
    """
    covariance_matrix_diagonal = torch.diagonal(covariance_matrix)
    covariance_matrix_off_diagonal = covariance_matrix - torch.diag(
        covariance_matrix_diagonal)
    dip_regularizer = torch.add(
        lambda_od * torch.sum(covariance_matrix_off_diagonal ** 2),
        lambda_d * torch.sum((covariance_matrix_diagonal - 1) ** 2))
    return dip_regularizer


@gin.configurable("dip_vae")
class DIPVAE(Regularizer):
    """DIPVAE model."""

    def __init__(self,
                 lambda_od=gin.REQUIRED,
                 lambda_d_factor=gin.REQUIRED,
                 dip_type="i", **kwargs):
        """Creates a DIP-VAE model.

        Based on Equation 6 and 7 of "Variational Inference of Disentangled Latent
        Concepts from Unlabeled Observations"
        (https://openreview.net/pdf?id=H1kG7GZAW).

        Args:
          lambda_od: Hyperparameter for off diagonal values of covariance matrix.
          lambda_d_factor: Hyperparameter for diagonal values of covariance matrix
            lambda_d = lambda_d_factor*lambda_od.
          dip_type: "i" or "ii".
        """
        super().__init__()
        self.lambda_od = lambda_od
        self.lambda_d_factor = lambda_d_factor
        self.dip_type = dip_type

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        kl_loss = kl.sum()
        cov_z_mean = compute_covariance_z_mean(z_mean)
        lambda_d = self.lambda_d_factor * self.lambda_od
        if self.dip_type == "i":  # Eq 6 page 4
            # mu = z_mean is [batch_size, num_latent]
            # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
            cov_dip_regularizer = regularize_diag_off_diag_dip(
                cov_z_mean, self.lambda_od, lambda_d)
        elif self.dip_type == "ii":
            cov_enc = torch.diag(torch.exp(z_logvar))
            expectation_cov_enc = torch.mean(cov_enc, dim=0)
            cov_z = expectation_cov_enc + cov_z_mean
            cov_dip_regularizer = regularize_diag_off_diag_dip(
                cov_z, self.lambda_od, lambda_d)
        else:
            raise NotImplementedError("DIP variant not supported.")
        return kl_loss + cov_dip_regularizer


def gaussian_log_density(samples, mean, log_var):
    pi = torch.tensor(math.pi, device=samples.device)
    normalization = torch.log(2. * pi)
    inv_sigma = torch.exp(-log_var)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)


def decompose(z, z_mean, z_logvar):
    log_qz_prob = gaussian_log_density(
        torch.unsqueeze(z, 1), torch.unsqueeze(z_mean, 0),
        torch.unsqueeze(z_logvar, 0))
    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = torch.sum(
        torch.logsumexp(log_qz_prob, dim=1, keepdim=False),
        dim=1,
        keepdim=False)
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = torch.logsumexp(
        torch.sum(log_qz_prob, dim=2, keepdim=False),
        dim=1,
        keepdim=False)
    return log_qz_prob, log_qz, log_qz_product


def total_correlation(z, z_mean, z_logvar):
    """Estimate of total correlation on a batch.

    We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
    log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
    for the minimization. The constant should be equal to (num_latents - 1) *
    log(batch_size * dataset_size)

    Args:
      z: [batch_size, num_latents]-tensor with sampled representation.
      z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
      z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.

    Returns:
      Total correlation estimated on a batch.
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    # q(z,n); [ sample_z: batch_size, sample_x: batch_size, latent: num_latent]
    log_qz_prob = gaussian_log_density(
        torch.unsqueeze(z, 1), torch.unsqueeze(z_mean, 0),
        torch.unsqueeze(z_logvar, 0))
    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = torch.sum(
        torch.logsumexp(log_qz_prob, dim=1, keepdim=False),
        dim=1,
        keepdim=False)
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = torch.logsumexp(
        torch.sum(log_qz_prob, dim=2, keepdim=False),
        dim=1,
        keepdim=False)
    return torch.mean(log_qz - log_qz_product)


@gin.configurable("beta_tc_vae")
class BetaTCVAE(Regularizer):
    """BetaTCVAE model."""

    def __init__(self, beta=gin.REQUIRED, **kwargs):
        """Creates a beta-TC-VAE model.

        Based on Equation 4 with alpha = gamma = 1 of "Isolating Sources of
        Disentanglement in Variational Autoencoders"
        (https://arxiv.org/pdf/1802.04942).
        If alpha = gamma = 1, Eq. 4 can be written as ELBO + (1 - beta) * TC.

        Args:
          beta: Hyperparameter total correlation.
        """
        super().__init__()
        self.beta = beta

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        kl_loss = kl.sum()
        tc = (self.beta - 1.) * total_correlation(z_sampled, z_mean, z_logvar)
        return tc + kl_loss


@gin.configurable("cascade_vae_c")
class CascadeVAEC(Regularizer):
    """CascadeVAEC model."""

    def __init__(self,
                 stage_steps=gin.REQUIRED,
                 beta_min=1,
                 beta_max=10,
                 td=1e5):
        """Creates a CascadeVAE-C model.

        Refers https://github.com/snu-mllab/DisentanglementICML19

        Args:
          beta_min: Hyperparameter the lower pressure.
          beta_max: Hyperparameter the higher pressure.
        """
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.td = td
        self.stage_steps = stage_steps

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        global_step = model.global_step
        self.stage = min(global_step // self.stage_steps, model.num_latent-1)
        weight = torch.ones_like(kl) * self.beta_min
        weight[self.stage + 1:] = self.beta_max
        kl_loss = (weight * kl).sum()
        return kl_loss


@gin.configurable("annealing")
class Annealing(Regularizer):
    def __init__(self, beta_h=80):
        super().__init__()
        self.beta = beta_h
        self.total_steps = gin.query_parameter('train.training_steps')
        self.delta = beta_h / self.total_steps

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        beta = max(self.beta - self.delta, 1)
        model.summary['beta'] = beta
        self.beta = beta
        return beta * (kl.sum())


@gin.configurable('deft')
class DEFT(Regularizer):
    def __init__(self,
                 stage_steps=gin.REQUIRED,
                 betas=gin.REQUIRED,):
        super().__init__()
        self.betas = betas
        self.stage_steps = stage_steps

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        """Training compatible model function."""
        global_step = model.global_step
        self.stage = min(global_step // self.stage_steps, len(self.betas) - 1)
        if global_step % self.stage_steps == 0:
            model.encode.set_stage(self.stage)
            print('stage', self.stage)

        beta = self.betas[self.stage]
        model.summary['beta'] = beta
        return beta * (kl.sum())


class DecoderReg(Regularizer):
    def __init__(self, input_shape, shared=True, alpha=0, lam=0, ):
        super().__init__()
        self.alpha = alpha
        self.lam = lam
        self.shared = shared

        if not shared:
            self.decodes = \
                torch.nn.Sequential(*[architectures.lite_decoder(self.num_latent, input_shape)
                                      for _ in range(self.num_latent - 1)])

    def get_decoder(self, i):
        if self.shared:
            return self.decode

        if i == self.num_latent - 1:
            return self.decode
        return self.decodes[i]

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        features, _ = data_batch
        reg_loss = torch.tensor(0)

        num_samples = 100
        device = z_sampled.device

        H_xCz = []
        z1 = torch.randn(num_samples, self.num_latent, device=device)
        for i in range(self.num_latent + 1):
            r = torch.randn_like(z_sampled)
            r[:, :i] = z_sampled[:, :i]

            reconstructions = self.get_decoder(i)(r).detach()
            per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
            reg_recon = torch.mean(per_sample_loss)
            model.summary[f'reg_recon/{i}'] = reg_recon
            # reg_loss1 = reg_recon + kl[:i + 1].sum()
            # print(loss,reg_loss,self.alpha)
            # reg_loss = reg_loss + reg_loss1 * pow(self.alpha, i)

            # decoder reg
            z2 = torch.randn(num_samples, self.num_latent - i, device=device)
            z = torch.cat([z1[:, :i], z2], 1)
            recons = torch.sigmoid(self.decode(z))
            H_xCz.append(BernoulliH(recons.mean(0)).flatten())

        H_xCz = torch.stack(H_xCz)
        H_gap = H_xCz[1:] - H_xCz[:-1]
        gap_indices = torch.argmax(H_gap, 0)

        for i in range(1, H_xCz.size(0)):
            reg_loss = reg_loss + H_xCz[i, i > gap_indices].sum()

        model.summary["reg_loss"] = reg_loss

        for i, max_gap in enumerate(H_gap.max(1)[1]):
            model.summary[f"H_gap/{i}"] = max_gap.item()
        if self.alpha == 1:
            s = self.num_latent
        else:
            s = (1 - pow(self.alpha, self.num_latent)) / (1 - self.alpha)
        return reg_loss / s
