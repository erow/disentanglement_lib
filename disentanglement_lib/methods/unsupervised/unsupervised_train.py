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
from disentanglement_lib.methods.shared import losses
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from disentanglement_lib.methods.unsupervised import model  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised.gaussian_encoder_model import GaussianModel
from disentanglement_lib.methods.unsupervised.model import gaussian_log_density
from disentanglement_lib.utils import results
from disentanglement_lib.evaluation.metrics import mig

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
from disentanglement_lib.utils.mi_estimators import estimate_entropies
from disentanglement_lib.visualize.visualize_util import plt_sample_traversal


@gin.configurable("train", blacklist=[])
class Train(pl.LightningModule):
    """Trains the estimator and exports the snapshot and the gin config.

        The use of this function requires the gin binding 'dataset.name' to be
        specified as that determines the data set used for training.

        Args:
          model: GaussianEncoderModel that should be trained and exported.
          training_steps: Integer with number of training steps.
          random_seed: Integer with random seed used for training.
          batch_size: Integer with the batch size.
          name: Optional string with name of the model (can be used to name models).
          model_num: Optional integer with model number (can be used to identify
            models).
    """

    def __init__(self,
                 model=gin.REQUIRED,
                 training_steps=gin.REQUIRED,
                 random_seed=gin.REQUIRED,
                 batch_size=gin.REQUIRED,
                 opt_name=torch.optim.Adam,
                 lr=5e-4,
                 eval_numbers=10,
                 name="",
                 model_num=None):
        super().__init__()
        self.training_steps = training_steps
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.lr = lr
        self.name = name
        self.model_num = model_num
        self.eval_numbers = eval_numbers
        wandb.config['dataset'] = gin.query_parameter('dataset.name')
        self.save_hyperparameters()
        self.opt_name = opt_name
        self.data = named_data.get_named_ground_truth_data()
        img_shape = np.array(self.data.observation_shape)[[2, 0, 1]].tolist()
        # img_shape = [1,64,64]
        self.ae = model(img_shape)

    def training_step(self, batch, batch_idx):
        if (self.global_step + 1) % (self.training_steps // self.eval_numbers) == 0:
            self.evaluate()
        x = batch
        loss, summary = self.ae.model_fn(x.float(), None)
        self.log_dict(summary)
        return loss

    def evaluate(self) -> None:
        model = self.ae
        model.cpu()
        model.eval()
        dic_log = {}
        dic_log.update(self.visualize_model(model))
        wandb.log(dic_log)
        model.cuda()
        model.train()

    def visualize_model(self, model) -> dict:
        _encoder, _decoder = convert_model(model)
        num_latent = self.ae.num_latent
        mu = torch.zeros(1, num_latent)
        fig = plt_sample_traversal(mu, _decoder, 8, range(num_latent), 2)
        return {'traversal': wandb.Image(fig)}

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

    def save_model(self, file):
        dir = '/tmp/models/' + str(np.random.randint(99999))
        file_path = os.path.join(dir, file)
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.ae.state_dict(), file_path)
        wandb.save(file_path, base_path=dir)
