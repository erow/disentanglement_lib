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
import wandb
from absl import app
from absl import flags
from absl import logging

from disentanglement_lib.methods.unsupervised.model import anneal
from examples.TC import get_log_pz_qz_prodzi_qzCx

from disentanglement_lib.config import reproduce
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.methods.unsupervised import train, model
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.visualize import visualize_model
import numpy as np

experiment = __file__.split('/')[-1][:-3]
base_directory = 'experiment_results'


def run_model(output_directory, strength, run_id, overwrite=True):
    # study templates
    study = reproduce.STUDIES['unsupervised_study_v1']
    # Model training (if model directory is not provided).

    logging.info("Training model...")
    model_dir = os.path.join(output_directory, "model")
    model_bindings, model_config_file = study.get_model_config(0)
    gin_bindings = [
        "train.model = @AnnealedTCVAE",
        f"train.random_seed={run_id}",
        # "train.training_steps=100",
        f"AnnealedTCVAE.beta={strength}",
        "AnnealedTCVAE.gamma=1000",
        "AnnealedTCVAE.c_max=50.",
        "AnnealedTCVAE.iteration_threshold=100000"
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


@gin.configurable("AnnealedTCVAE")  # This will allow us to reference the model.
class AnnealedTCVAE(model.BaseVAE):
    """AnnealedTCVAE model."""

    def __init__(self, input_shape,
                 beta=gin.REQUIRED,
                 gamma=gin.REQUIRED,
                 c_max=gin.REQUIRED,
                 iteration_threshold=gin.REQUIRED, **kwargs):
        """
        Args:
          gamma: Hyperparameter for the regularizer.
          c_max: Maximum capacity of the bottleneck.
          iteration_threshold: How many iterations to reach c_max.
        """
        super().__init__(input_shape, beta=beta, gamma=gamma,
                         c_max=c_max,
                         iteration_threshold=iteration_threshold, **kwargs)
        self.beta = beta
        self.gamma = gamma
        self.c_max = c_max
        self.iteration_threshold = iteration_threshold

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        c = 1 - anneal(1, self.global_step, self.iteration_threshold)
        log_pz, log_qz, log_prod_qzi, log_q_zCx = get_log_pz_qz_prodzi_qzCx(z_sampled,
                                                                            (z_mean, z_logvar),
                                                                            32 * 32 * 40 * 6 * 3,
                                                                            is_mss=True)

        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        wandb.log({
            'mi': mi_loss,
            'tc': tc_loss,
            'dw': dw_kl_loss,
            'c': c
        })
        return mi_loss * c * 100 + self.beta * tc_loss + dw_kl_loss


if __name__ == "__main__":
    for i, beta in enumerate([1., 2., 4., 6., 8., 16.]):
        for random_seed in range(10):
            wandb.init(project='experiments', tags=[experiment], reinit=True,
                       config={
                           'beta': beta,
                           'random_seed': random_seed
                       })
            output_directory = os.path.join(base_directory, experiment, str(i * 10 + random_seed))
            run_model(output_directory, beta, random_seed)
