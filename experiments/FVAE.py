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
import shutil
from shutil import copyfile

import gin.torch
import torch
import wandb
from absl import logging

from disentanglement_lib.methods.unsupervised.vae import anneal, load_model

from disentanglement_lib.config import reproduce
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.methods.unsupervised import train, vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.visualize import visualize_model
import numpy as np

experiment = __file__.split('/')[-1][:-3]
base_directory = 'experiment_results'


def run_model(output_directory, gin_bindings, train_model, overwrite=True):
    # study templates
    study = reproduce.STUDIES['unsupervised_study_v1']
    # Model training (if model directory is not provided).

    logging.info("Training model...")
    model_dir = os.path.join(output_directory, "model")
    model_bindings, model_config_file = study.get_model_config(0)

    # The main training protocol of disentanglement_lib is defined in the
    # disentanglement_lib.methods.unsupervised.train module. To configure
    # training we need to provide a gin config. For a standard VAE, you may have a
    # look at model.gin on how to do this.

    gin.parse_config_files_and_bindings([model_config_file], gin_bindings)

    def init_model(input_shape):
        global phase
        if phase == 0:
            model = train_model(input_shape)
        else:
            print("load ", phase)
            model = load_model(model_dir, f"{phase - 1}.pth")
        wandb.watch(model)
        return model

    train.train(model_dir, False, init_model)
    copyfile(os.path.join(model_dir, "ckp.pth"), os.path.join(model_dir, f"{phase}.pth"))
    gin.clear_config()

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

    # return
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

    visualize_model.visualize(model_dir, os.path.join(output_directory, 'visualization'), overwrite)


if __name__ == "__main__":
    epochs = [1, 2, 4, 40]
    for random_seed in range(1):
        for method in ["vae", "AnnealedTCVAE", ]:
            for phase, beta in enumerate([167, 90, 60, 6]):
                steps = epochs[phase] * 11520
                output_directory = os.path.join(base_directory, experiment, method, str(random_seed))
                model_file = os.path.join(output_directory, 'model', f"{phase}.pth")
                if os.path.exists(model_file):
                    # print("skip", random_seed, phase, method)
                    # continue
                    shutil.rmtree(output_directory)
                    # pass
                wandb.init(project='experiments', tags=[experiment], reinit=True,
                           config={
                               'beta': beta,
                               'phase': phase,
                               'method': method,
                               'random_seed': random_seed
                           })
                model = vae.BetaVAE if method == 'vae' else vae.AnnealedTCVAE
                gin_bindings = [
                    'dataset.name = "dsprites_noshape"',
                    f"train.model = @{method}",
                    f"train.random_seed={random_seed}",
                    f"vae.beta={beta}",
                    f"AnnealedTCVAE.beta=6.",
                    f"AnnealedTCVAE.gamma={beta}",
                    f"fractional_conv_encoder.active={phase}",
                    "model.encoder_fn = @fractional_conv_encoder",
                    "model.num_latent = 12",
                    f"train.training_steps={steps}"
                ]

                run_model(output_directory, gin_bindings, model)

                wandb.save(os.path.join(output_directory, "model/results/json/*.json"))
                wandb.save(os.path.join(output_directory, "visualization/animated_traversals/fixed_interval_cycle*"))

                representation = np.load(
                    os.path.join(output_directory, 'postprocessed', 'representation', 'representation.npy'),
                    allow_pickle=True)
                representation = representation[()]
                z = torch.Tensor(representation['mean'])
                fig = visualize_model.plot_latent_vs_ground(z, latnt_sizes=[6, 40, 32, 32])
                wandb.log({'projection': wandb.Image(fig)})
                wandb.join()