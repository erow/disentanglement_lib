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
import math
from abc import ABC

from disentanglement_lib.methods.shared import architectures  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import losses  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from six.moves import range
from six.moves import zip
import numpy as np
import torch
import gin

from disentanglement_lib.methods.unsupervised.gaussian_encoder_model import GaussianModel, load


@gin.configurable("model", allowlist=["num_latent", "encoder_fn", "decoder_fn"])
class BaseVAE(gaussian_encoder_model.GaussianModel):
    """Abstract base class of a basic Gaussian encoder model."""

    def __init__(self, input_shape,
                 num_latent=gin.REQUIRED,
                 encoder_fn=gin.REQUIRED,
                 decoder_fn=gin.REQUIRED,
                 **kwargs):
        if hasattr(encoder_fn, '__wrapped__'):
            super().__init__(input_shape, num_latent=num_latent,
                             encoder_fn=encoder_fn.__wrapped__,
                             decoder_fn=decoder_fn.__wrapped__,
                             **kwargs)
        else:
            super().__init__(input_shape, num_latent=num_latent,
                             encoder_fn=encoder_fn,
                             decoder_fn=decoder_fn,
                             **kwargs)

        self.encode = encoder_fn(input_shape=input_shape, num_latent=num_latent)
        self.decode = decoder_fn(num_latent=num_latent, output_shape=input_shape)
        self.num_latent = self.encode.num_latent
        self.input_shape = input_shape

    def model_fn(self, features, labels):
        """Training compatible model function."""
        del labels
        self.summary = {}
        z_mean, z_logvar = self.encode(features)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
        reconstructions = self.decode(z_sampled)
        per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
        reconstruction_loss = torch.mean(per_sample_loss)
        kl_loss = compute_gaussian_kl(z_mean, z_logvar)
        regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
        loss = torch.add(reconstruction_loss, regularizer)
        elbo = torch.add(reconstruction_loss, kl_loss)

        self.summary['reconstruction_loss'] = reconstruction_loss
        self.summary['elbo'] = -elbo
        self.summary['kl_loss'] = kl_loss
        self.summary['loss'] = loss

        kl = 0.5 * torch.mean(
            torch.square(z_mean.data) + torch.exp(z_logvar.data) - z_logvar.data - 1, [0])

        for i in range(kl.shape[0]):
            self.summary[f"kl/{i}"] = kl[i]

        return self.summary

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
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
    return torch.mean(
        0.5 * torch.sum(
            torch.square(z_mean) + torch.exp(z_logvar) - z_logvar - 1, [1]),
    )


def make_metric_fn(*names):
    """Utility function to report torch.metrics in model functions."""

    def metric_fn(*args):
        return {name: torch.mean(vec) for name, vec in zip(names, args)}

    return metric_fn


@gin.configurable("vae")
class BetaVAE(BaseVAE):
    """BetaVAE model."""

    def __init__(self, input_shape, beta=gin.REQUIRED, **kwargs):
        """Creates a beta-VAE model.

        Implementing Eq. 4 of "beta-VAE: Learning Basic Visual Concepts with a
        Constrained Variational Framework"
        (https://openreview.net/forum?id=Sy2fzU9gl).

        Args:
          beta: Hyperparameter for the regularizer.

        Returns:
          model_fn: Model function for TPUEstimator.
        """
        super().__init__(input_shape, beta=beta, **kwargs)
        self.beta = beta

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        return self.beta * kl_loss


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
class AnnealedVAE(BaseVAE):
    """AnnealedVAE model."""

    def __init__(self, input_shape, gamma=gin.REQUIRED, c_max=gin.REQUIRED, iteration_threshold=gin.REQUIRED, **kwargs):
        """Creates an AnnealedVAE model.

        Implementing Eq. 8 of "Understanding disentangling in beta-VAE"
        (https://arxiv.org/abs/1804.03599).

        Args:
          gamma: Hyperparameter for the regularizer.
          c_max: Maximum capacity of the bottleneck.
          iteration_threshold: How many iterations to reach c_max.
        """
        super().__init__(input_shape, gamma=gamma, c_max=c_max, iteration_threshold=iteration_threshold, **kwargs)
        self.gamma = gamma
        self.c_max = c_max
        self.iteration_threshold = iteration_threshold

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        del z_mean, z_logvar, z_sampled
        c = anneal(self.c_max, self.global_step, self.iteration_threshold)
        return self.gamma * torch.abs(kl_loss - c)


@gin.configurable("factor_vae")
class FactorVAE(BaseVAE):
    """FactorVAE model."""

    def __init__(self, input_shape, gamma=gin.REQUIRED, **kwargs):
        """Creates a FactorVAE model.

        Implementing Eq. 2 of "Disentangling by Factorizing"
        (https://arxiv.org/pdf/1802.05983).

        Args:
          gamma: Hyperparameter for the regularizer.
        """
        super().__init__(input_shape, gamma=gamma, **kwargs)
        self.gamma = gamma
        self.discriminator = architectures.make_discriminator(self.num_latent)
        self.opt = torch.optim.Adam(self.discriminator.parameters())

    def model_fn(self, features, labels):
        """TPUEstimator compatible model function."""

        data_shape = features.shape[1:]
        z_mean, z_logvar = self.encode(features)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
        z_shuffle = shuffle_codes(z_sampled)

        logits_z, probs_z = self.discriminator(z_sampled.data)
        _, probs_z_shuffle = self.discriminator(z_shuffle.data)

        self.opt.zero_grad()
        discr_loss = -torch.add(
            0.5 * torch.mean(torch.log(probs_z[:, 0])),
            0.5 * torch.mean(torch.log(probs_z_shuffle[:, 1])))
        discr_loss.backward()
        self.opt.step()

        logits_z, probs_z = self.discriminator(z_sampled)
        _, probs_z_shuffle = self.discriminator(z_shuffle)

        reconstructions = self.decode(z_sampled)
        per_sample_loss = losses.make_reconstruction_loss(
            features, reconstructions)
        reconstruction_loss = torch.mean(per_sample_loss)
        kl_loss = compute_gaussian_kl(z_mean, z_logvar)
        elbo = torch.add(reconstruction_loss, kl_loss)
        # tc = E[log(p_real)-log(p_fake)] = E[logit_real - logit_fake]
        tc_loss_per_sample = logits_z[:, 0] - logits_z[:, 1]
        tc_loss = torch.mean(tc_loss_per_sample, dim=0)
        regularizer = kl_loss + self.gamma * tc_loss
        factor_vae_loss = reconstruction_loss + regularizer

        summary = {'reconstruction_loss': reconstruction_loss,
                   'discr_loss': discr_loss,
                   'elbo': -elbo,
                   'loss': factor_vae_loss}

        kl = 0.5 * torch.mean(
            torch.square(z_mean.data) + torch.exp(z_logvar.data) - z_logvar.data - 1, [0])

        for i in range(kl.shape[0]):
            summary[f"kl/{i}"] = kl[i]

        return summary


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
class DIPVAE(BaseVAE):
    """DIPVAE model."""

    def __init__(self, input_shape,
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
        super().__init__(input_shape, lambda_od=lambda_od, lambda_d_factor=lambda_d_factor, dip_type=dip_type, **kwargs)
        self.lambda_od = lambda_od
        self.lambda_d_factor = lambda_d_factor
        self.dip_type = dip_type

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
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
    pi = torch.tensor(math.pi)
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
class BetaTCVAE(BaseVAE):
    """BetaTCVAE model."""

    def __init__(self, input_shape, beta=gin.REQUIRED, **kwargs):
        """Creates a beta-TC-VAE model.

        Based on Equation 4 with alpha = gamma = 1 of "Isolating Sources of
        Disentanglement in Variational Autoencoders"
        (https://arxiv.org/pdf/1802.04942).
        If alpha = gamma = 1, Eq. 4 can be written as ELBO + (1 - beta) * TC.

        Args:
          beta: Hyperparameter total correlation.
        """
        super().__init__(input_shape, beta=beta, **kwargs)
        self.beta = beta

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        tc = (self.beta - 1.) * total_correlation(z_sampled, z_mean, z_logvar)
        return tc + kl_loss


@gin.configurable("AnnealedTCVAE")  # This will allow us to reference the model.
class AnnealedTCVAE(BaseVAE):
    """AnnealedTCVAE model."""

    def __init__(self, input_shape,
                 beta=gin.REQUIRED,
                 gamma=gin.REQUIRED,
                 **kwargs):
        """
        Args:
          gamma: Hyperparameter for the regularizer.
          c_max: Maximum capacity of the bottleneck.
          iteration_threshold: How many iterations to reach c_max.
        """
        super().__init__(input_shape,
                         beta=beta,
                         gamma=gamma,
                         **kwargs)
        self.beta = beta
        self.gamma = gamma

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        log_qzCx = gaussian_log_density(z_sampled, z_mean, z_logvar).sum(1)
        log_pz = gaussian_log_density(z_sampled,
                                      torch.zeros_like(z_mean),
                                      torch.zeros_like(z_mean)).sum(1)
        _, log_qz, log_qz_product = decompose(z_sampled, z_mean, z_logvar)

        mi = torch.mean(log_qzCx - log_qz)
        tc = torch.mean(log_qz - log_qz_product)
        dw_kl_loss = torch.mean(log_qz_product - log_pz)
        self.summary['mi'] = mi
        self.summary['tc'] = tc
        self.summary['dw'] = dw_kl_loss
        return self.gamma * mi + self.beta * tc + dw_kl_loss


@gin.configurable("AnnealedTCVAE1")  # This will allow us to reference the model.
class AnnealedTCVAE1(BaseVAE):
    """AnnealedTCVAE model."""

    def __init__(self, input_shape,
                 beta=gin.REQUIRED,
                 gamma=gin.REQUIRED,
                 **kwargs):
        """
        Args:
          gamma: Hyperparameter for the regularizer.
          c_max: Maximum capacity of the bottleneck.
          iteration_threshold: How many iterations to reach c_max.
        """
        super().__init__(input_shape,
                         beta=beta,
                         gamma=gamma,
                         **kwargs)
        self.beta = beta
        self.gamma = gamma

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        c = anneal(self.gamma, self.global_step, 100000)

        log_qzCx = gaussian_log_density(z_sampled, z_mean, z_logvar).sum(1)
        log_pz = gaussian_log_density(z_sampled,
                                      torch.zeros_like(z_mean),
                                      torch.zeros_like(z_mean)).sum(1)
        _, log_qz, log_qz_product = decompose(z_sampled, z_mean, z_logvar)

        mi = torch.mean(log_qzCx - log_qz)
        tc = torch.mean(log_qz - log_qz_product)
        dw_kl_loss = torch.mean(log_qz_product - log_pz)
        self.summary['mi'] = mi
        self.summary['tc'] = tc
        self.summary['dw'] = dw_kl_loss
        self.summary['c'] = c
        return 500 * (-self.gamma + c - mi).abs() + self.beta * tc + dw_kl_loss


@gin.configurable("AnnealedTCVAE2")  # This will allow us to reference the model.
class AnnealedTCVAE2(BaseVAE):
    """AnnealedTCVAE model."""

    def __init__(self, input_shape,
                 beta=gin.REQUIRED,
                 gamma=gin.REQUIRED,
                 **kwargs):
        """
        Args:
          gamma: Hyperparameter for the regularizer.
          c_max: Maximum capacity of the bottleneck.
          iteration_threshold: How many iterations to reach c_max.
        """
        super().__init__(input_shape,
                         beta=beta,
                         gamma=gamma,
                         **kwargs)
        self.beta = beta
        self.gamma = gamma
        self.N = 3 * 6 * 40 * 32 * 32

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        c = 1 - anneal(1, self.global_step, 100000)

        log_qzCx = gaussian_log_density(z_sampled, z_mean, z_logvar).sum(1)
        log_pz = gaussian_log_density(z_sampled,
                                      torch.zeros_like(z_mean),
                                      torch.zeros_like(z_mean)).sum(1)
        _, log_qz, log_qz_product = decompose(z_sampled, z_mean, z_logvar)

        # 常数矫正，但是常数不影响结果
        batch_size = z_mean.size(0)
        log_qz = log_qz - np.log(batch_size * self.N)
        log_qz_product = log_qz_product - np.log(batch_size * self.N) * z_mean.size(1)

        mi = torch.mean(log_qzCx - log_qz)
        tc = torch.mean(log_qz - log_qz_product)
        dw_kl_loss = torch.mean(log_qz_product - log_pz)
        self.summary['mi'] = mi
        self.summary['tc'] = tc
        self.summary['dw'] = dw_kl_loss
        self.summary['c'] = c
        return self.gamma * mi * c + self.beta * tc + dw_kl_loss


def load_model(model_dir, filename='ckp.pth'):
    return load(GaussianModel, model_dir, filename)
