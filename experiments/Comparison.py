#!/usr/bin/env python
# coding=utf-8

"""
比较AnnealedVAE与ThresholdVAE的区别。

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

from disentanglement_lib.methods.unsupervised.model import anneal, load_model

from disentanglement_lib.config import reproduce
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.methods.unsupervised import train, model
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.visualize import visualize_model
import numpy as np

experiment = __file__.split('/')[-1][:-3]
base_directory = 'experiment_results'

gin_config = \
    """
    
    """


def run_model(output_directory, gin_bindings, train_model, overwrite=True):
    logging.info("Training model...")
    model_dir = os.path.join(output_directory, "model")
    gin.parse_config_files_and_bindings(["shared.gin"], gin_bindings)

    def init_model(input_shape):
        global phase
        if phase == 0:
            model = train_model(input_shape)
        else:
            print("load ", phase)
            model = load_model(model_dir, f"{phase - 1}.pth")
        wandb.watch(model.encode, log='all')
        return model

    train.train(model_dir, False, init_model)
    copyfile(os.path.join(model_dir, "ckp.pth"), os.path.join(model_dir, f"{phase}.pth"))

    postprocess.postprocess(
        model_dir, os.path.join(output_directory, 'representation'), True, 0
    )
    gin.clear_config()


if __name__ == "__main__":

    method = "annealed_vae"
    phase = 0
    for random_seed in range(3):
        for c in [3, 4, 5, 6]:
            for gamma in [10, 20, 40]:
                steps = 50000
                output_directory = os.path.join(base_directory, experiment, method, str(random_seed))
                wandb.init(project='experiments', tags=[experiment], reinit=True,
                           config={
                               'c': c,
                               'gamma': gamma,
                               'method': method,
                               'random_seed': random_seed
                           })
                model = model.AnnealedVAE
                gin_bindings = [
                    'dataset.name = "translation"',
                    f"translation.img_size=(2,8,1)",
                    f"translation.stride=1",
                    f"train.model = @{method}",
                    f"train.lr = 0.001",
                    f"train.random_seed=0",
                    f"train.training_steps={steps + 10000}",
                    "model.num_latent = 6",
                    f"annealed_vae.iteration_threshold={steps}",
                    f"annealed_vae.c_max = {c}",
                    f"annealed_vae.gamma = {gamma}"
                ]

                run_model(output_directory, gin_bindings, model)

                representation = np.load(os.path.join(output_directory, 'representation', 'representation.npy'),
                                         allow_pickle=True)
                representation = representation[()]
                z = torch.Tensor(representation['mean'])
                fig = visualize_model.plot_latent_vs_ground(z, latnt_sizes=[16, 16])
                wandb.log({'projection': wandb.Image(fig)})
                wandb.join()
    # exit()
    method = "vae"
    for random_seed in range(3):
        for phase, beta in enumerate([13, 3]):
            # if True:
            #     phase, beta = 1, 6
            steps = 15000 * phase + 8000
            output_directory = os.path.join(base_directory, experiment, method, str(random_seed))
            model_file = os.path.join(output_directory, 'model', f"{phase}.pth")
            if os.path.exists(model_file):
                # print("skip", random_seed, phase, method)
                # continue
                # shutil.rmtree(output_directory)
                pass
            wandb.init(project='experiments', tags=[experiment], reinit=True,
                       config={
                           'beta': beta,
                           'phase': phase,
                           'method': method,
                           'random_seed': random_seed
                       })
            model = model.BetaVAE
            gin_bindings = [
                'dataset.name = "translation"',
                f"translation.img_size=(2,8,1)",
                f"translation.stride=1",
                f"train.model = @{method}",
                f"train.lr = 0.001",
                f"train.random_seed=0",
                f"train.training_steps={steps}",
                f"vae.beta={beta}",
                f"AnnealedTCVAE.beta=6.",
                f"AnnealedTCVAE.gamma={beta}",
                f"fractional_conv_encoder.active={phase}",
                "fractional_conv_encoder.groups=2",
                "model.encoder_fn = @fractional_conv_encoder",
                "model.num_latent = 6",
            ]

            run_model(output_directory, gin_bindings, model)

            representation = np.load(os.path.join(output_directory, 'representation', 'representation.npy'),
                                     allow_pickle=True)
            representation = representation[()]
            z = torch.Tensor(representation['mean'])
            fig = visualize_model.plot_latent_vs_ground(z, latnt_sizes=[16, 16])
            wandb.log({'projection': wandb.Image(fig)})
            wandb.join()
