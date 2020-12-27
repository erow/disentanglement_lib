#!/usr/bin/env python
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

"""Pipeline to reproduce fixed models and evaluation protocols.

This is the main pipeline for the reasoning step in the paper:
Are Disentangled Representations Helpful for Abstract Visual Reasoning?
Sjoerd van Steenkiste, Francesco Locatello, Juergen Schmidhuber, Olivier Bachem.
NeurIPS, 2019.
https://arxiv.org/abs/1905.12506
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gin.torch
import torch
import wandb
from absl import app
from absl import flags
from absl import logging

from disentanglement_lib.methods.unsupervised.vae import anneal
from examples.TC import get_log_pz_qz_prodzi_qzCx

from disentanglement_lib.config import reproduce
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.methods.unsupervised import train, vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.visualize import visualize_model
import numpy as np

experiment = __file__.split('/')[-1][:-3]
base_directory = 'experiment_results'


def run_model(output_directory, config, overwrite=True):
    # study templates
    study = reproduce.STUDIES['unsupervised_study_v1']
    # Model training (if model directory is not provided).

    logging.info("Training model...")
    model_dir = os.path.join(output_directory, "model")
    model_bindings, model_config_file = study.get_model_config(0)
    gin_bindings = [
        'dataset.name = "dsprites_noshape"',
        f"train.model = @{config.method}",
        f"train.random_seed={config.random_seed}",
        f"{config.method}.beta={config.beta}",
        f"{config.method}.gamma={config.gamma}"
    ]

    # The main training protocol of disentanglement_lib is defined in the
    # disentanglement_lib.methods.unsupervised.train module. To configure
    # training we need to provide a gin config. For a standard VAE, you may have a
    # look at model.gin on how to do this.

    train.train_with_gin(model_dir, overwrite, [model_config_file],
                         gin_bindings)

    # We fix the random seed for the postprocessing and evaluation steps (each
    # config gets a different but reproducible seed derived from a master seed of
    # 0). The model seed was set via the gin bindings and configs of the study.
    random_state = np.random.RandomState(0)

    # We extract the different representations and save them to disk.
    postprocess_config_files = sorted(study.get_postprocess_config_files())
    for config in postprocess_config_files:
        post_name = os.path.basename(config).replace(".gin", "")
        logging.info("Extracting representation %s...", post_name)
        post_dir = os.path.join(output_directory, "postprocessed", post_name)
        postprocess_bindings = [
            "postprocess.random_seed = {}".format(random_state.randint(2 ** 32)),
            "postprocess.name = '{}'".format(post_name)
        ]
        postprocess.postprocess_with_gin(model_dir, post_dir, overwrite,
                                         [config], postprocess_bindings)

    # Iterate through the disentanglement metrics.
    eval_configs = sorted(study.get_eval_config_files())
    eval_configs = eval_configs[1:]  # TODO remove DCI for taking too much time.

    for post_name in ['mean', 'sample']:
        post_dir = os.path.join(output_directory, "postprocessed",
                                "representation")
        # Now, we compute all the specified scores.
        for gin_eval_config in eval_configs:
            metric_name = os.path.basename(gin_eval_config).replace(".gin", "")
            logging.info("Computing metric '%s' on '%s'...", metric_name, post_name)
            metric_dir = os.path.join(output_directory, "metrics", post_name,
                                      metric_name)
            eval_bindings = [
                "evaluation.random_seed = {}".format(random_state.randint(2 ** 32)),
                "evaluation.name = '{}'".format(metric_name),
                f"evaluation.representation_fn =@{post_name}_representation"

            ]
            evaluate.evaluate_with_gin(post_dir, metric_dir, overwrite,
                                       [gin_eval_config], eval_bindings)

    # Visualization

    visualize_model.visualize(model_dir, os.path.join(output_directory, 'visualization'))


@gin.configurable("AnnealedTCVAE1")  # This will allow us to reference the model.
class AnnealedTCVAE1(vae.BaseVAE):
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

        log_qzCx = vae.gaussian_log_density(z_sampled, z_mean, z_logvar).sum(1)
        log_pz = vae.gaussian_log_density(z_sampled,
                                          torch.zeros_like(z_mean),
                                          torch.zeros_like(z_mean)).sum(1)
        _, log_qz, log_qz_product = vae.decompose(z_sampled, z_mean, z_logvar)

        mi = torch.mean(log_qzCx - log_qz)
        tc = torch.mean(log_qz - log_qz_product)
        dw_kl_loss = torch.mean(log_qz_product - log_pz)
        self.summary['mi'] = mi
        self.summary['tc'] = tc
        self.summary['dw'] = dw_kl_loss
        self.summary['c'] = c
        return 500 * (-self.gamma + c - mi).abs() + self.beta * tc + dw_kl_loss


@gin.configurable("AnnealedTCVAE2")  # This will allow us to reference the model.
class AnnealedTCVAE2(vae.BaseVAE):
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

        log_qzCx = vae.gaussian_log_density(z_sampled, z_mean, z_logvar).sum(1)
        log_pz = vae.gaussian_log_density(z_sampled,
                                          torch.zeros_like(z_mean),
                                          torch.zeros_like(z_mean)).sum(1)
        _, log_qz, log_qz_product = vae.decompose(z_sampled, z_mean, z_logvar)

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


if __name__ == "__main__":
    method = "AnnealedTCVAE1"
    for i, gamma in enumerate([2.5, 5., 10., 25.]):
        for random_seed in range(3):
            wandb.init(project='experiments', tags=[experiment], reinit=True,
                       config={
                           'beta': 6.,
                           'gamma': gamma,
                           'method': method,
                           'random_seed': random_seed
                       })
            output_directory = os.path.join(base_directory, experiment, method, str(gamma), str(random_seed))
            run_model(output_directory, wandb.config)

    method = "AnnealedTCVAE2"
    for i, gamma in enumerate([5., 10., 25., 50., 75., 100.]):
        for random_seed in range(3):
            wandb.init(project='experiments', tags=[experiment], reinit=True,
                       config={
                           'beta': 6.,
                           'gamma': gamma,
                           'method': method,
                           'random_seed': random_seed
                       })
            output_directory = os.path.join(base_directory, experiment, method, str(gamma), str(random_seed))
            run_model(output_directory, wandb.config)
