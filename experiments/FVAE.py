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

"""Main training protocol used for unsupervised disentanglement models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.data.ground_truth import util
from disentanglement_lib.data.ground_truth.ground_truth_data import *
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from disentanglement_lib.methods.unsupervised import model  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised.gaussian_encoder_model import GaussianModel
from disentanglement_lib.utils import results
from disentanglement_lib.evaluation.metrics import mig
from disentanglement_lib.methods.shared import losses

import numpy as np
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import gin
import pathlib, shutil
import wandb

from disentanglement_lib.utils.hub import convert_model
from disentanglement_lib.visualize.visualize_util import plt_sample_traversal

gin.enter_interactive_mode()


@gin.configurable("train", blacklist=[])
class Train(pl.LightningModule):
    def __init__(self,
                 data,
                 model=gin.REQUIRED,
                 training_steps=gin.REQUIRED,
                 random_seed=gin.REQUIRED,
                 batch_size=gin.REQUIRED,
                 opt_name=torch.optim.Adam,
                 lr=1e-4,
                 name="",
                 model_num=None):
        super().__init__()
        self.training_steps = training_steps
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.lr = lr
        self.name = name
        self.model_num = model_num
        self.save_hyperparameters()
        self.opt_name = opt_name

        self.data = data
        img_shape = np.array(self.data.observation_shape)[[2, 0, 1]].tolist()
        self.ae = model(img_shape)

    def training_step(self, batch, batch_idx):
        if (self.global_step + 1) % (self.training_steps // 10) == 0:
            self.evaluate()
        x, y = batch
        self.ae.alpha = self.global_step / self.training_steps
        loss, summary = self.ae.model_fn(x, y)
        self.log_dict(summary)
        return loss

    def evaluate(self) -> None:
        model = self.ae
        model.cpu()
        model.eval()
        self.visualize_model(model)
        self.compute_mig(model)
        self.gradient_test(model)
        model.cuda()
        model.train()

    def compute_mig(self, model):
        _encoder, _decoder = convert_model(model)
        result = mig.compute_mig(self.data, lambda x: _encoder(x)[0], np.random.RandomState(), )
        self.log_dict(result)

    def visualize_model(self, model) -> None:
        _encoder, _decoder = convert_model(model)
        num_latent = self.ae.num_latent
        mu = torch.zeros(1, num_latent)
        fig = plt_sample_traversal(mu, _decoder, 8, range(num_latent), 2)
        wandb.log({'traversal': wandb.Image(fig)})

    def gradient_test(self, model):
        dataset = self.data
        for i in range(dataset.num_factors):
            subset = []
            for _ in range(256 // dataset.factors_num_values[i]):
                factor = sample_factor(dataset)
                obs = action(dataset, factor, 0)
                subset.append(obs)
            subset = np.concatenate(subset).transpose([0, 3, 1, 2])
            subset = torch.FloatTensor(subset)
            mu, logvar = model.encode(subset)

            z = mu.detach().clone().requires_grad_(True)
            recons = model.decode(z)
            per_sample_loss = losses.make_reconstruction_loss(subset, recons)
            reconstruction_loss = torch.mean(per_sample_loss)
            reconstruction_loss.backward()
            grad = z.grad
            log = {}
            for j in range(grad.size(1)):
                log[f'grad_{i}/{j}'] = wandb.Histogram(grad[:, j].cpu())  # np_histogram=hist)
            wandb.log(log)

    def train_dataloader(self) -> DataLoader:
        dl = DataLoader(self.data,
                        batch_size=self.batch_size,
                        num_workers=4,
                        shuffle=True,
                        pin_memory=True)
        return dl

    def configure_optimizers(self):
        optimizer = self.opt_name(self.parameters(), lr=self.lr)
        return optimizer


import gin

gin_bindings = ['']
gin.clear_config()
gin.parse_config_files_and_bindings(['../examples/model.gin'], gin_bindings, False, skip_unknown=True)

if __name__ == '__main__':
    from disentanglement_lib.config.unsupervised_study_v1.sweep import UnsupervisedStudyV1

    study = UnsupervisedStudyV1()
    from disentanglement_lib.data.ground_truth import dsprites

    dataset = dsprites.DSprites([1, 2, 3, 4])
    from pytorch_lightning.loggers import WandbLogger

    logger = WandbLogger(project='dlib', tags=['v5'])
    print(logger.experiment.url)
    pl_model = Train(dataset, model.FVAE, )
    trainer = pl.Trainer(logger,
                         max_steps=pl_model.training_steps,
                         checkpoint_callback=False,
                         gpus=1)

    trainer.fit(pl_model)
