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

from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.cli import LightningCLI

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.data.ground_truth import util
from disentanglement_lib.data.ground_truth.ground_truth_data import *
from disentanglement_lib.methods.shared import architectures, losses
from disentanglement_lib.methods.unsupervised import model 

import numpy as np
import logging

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import gin
import pathlib, shutil
import wandb

def config_dict():
    configuration_object = gin.config._CONFIG
    macros = {}
    for (scope, selector), config in configuration_object.items():
        selector1 = selector.split('.')[-1]
        for k, v in config.items():
            macros[f"{selector1}.{k}"] = v
    return macros

class DataModule(LightningDataModule):
    def __init__(self, dataset="dsprites_tiny", batch_size=256):
        super().__init__()
        self.dataset = named_data.get_named_ground_truth_data(dataset)
        obs_np = self.dataset.observation_shape
        self._dims = obs_np[2], obs_np[0], obs_np[1]
        self.batch_size = batch_size
        
    # def add_argparse_args(self, parser):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('dataset_test',default=20,type=int)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, num_workers=4,
                                           shuffle=True, batch_size=self.batch_size, pin_memory=True)

    def val_dataloader(self):
        return self.train_dataloader()
        

@gin.configurable("model", blacklist=[])
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
                 input_shape = [1, 64, 64],
                 num_latent=10,
                 encoder_fn=architectures.conv_encoder,
                 decoder_fn=architectures.deconv_decoder,
                 regularizers=[],
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(config_dict())
        self.encode = encoder_fn(input_shape=input_shape, num_latent=num_latent)
        self.decode = decoder_fn(num_latent=num_latent, output_shape=input_shape)
        self.num_latent = num_latent
        self.input_shape = input_shape
        self.summary = {}
        if isinstance(regularizers,list):
            self.regularizers = nn.Sequential(*[i() for i in regularizers])
        else:
            self.regularizers = nn.Sequential(regularizers())

 
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, summary = self.model_fn(x.float(), y, self.global_step)
        # self.global_step += 1
        self.log_dict(summary)
        return loss
        
    
    def model_fn(self, features, labels, global_step):
        """Training compatible model function."""
        self.summary = {}
        z_mean, z_logvar = self.encode(features)
        z_sampled = model.sample_from_latent_distribution(z_mean, z_logvar)

        self.z_sampled = z_sampled
        reconstructions = self.decode(z_sampled)
        per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
        reconstruction_loss = torch.mean(per_sample_loss)
        self.summary['reconstruction_loss'] = reconstruction_loss

        kl = model.compute_gaussian_kl(z_mean, z_logvar)
        kl_loss = kl.sum()
        self.summary['kl_loss'] = kl_loss

        regularizer_loss = 0
        for regularizer in self.regularizers:
            regularizer_loss = regularizer_loss + regularizer((features, labels), self, kl, z_mean, z_logvar, z_sampled)

        loss = reconstruction_loss + regularizer_loss
        elbo = torch.add(reconstruction_loss, kl_loss)

        self.summary['elbo'] = -elbo
        self.summary['loss'] = loss

        for i in range(kl.shape[0]):
            self.summary[f"kl/{i}"] = kl[i]

        return loss, self.summary

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),5e-4)
 

    def save_model(self, file, dir):
        file_path = os.path.join(dir, file)
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.ae.state_dict(), file_path)


class MyLightningCLI(LightningCLI):
    def __init__(self,override_args=None, **kwargs, ) -> None:
        self.override_args=[] if override_args is None else override_args
        super().__init__(**kwargs)
        
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.Adam)

    # def parse_arguments(self) -> None:
    #     """Parses command line arguments and stores it in self.config"""
    #     self.config = self.parser.parse_args(self.override_args)

    def instantiate_trainer(self) -> None:
        """Implement to run some code before instantiating the classes"""
        logger = self.config_init['trainer'].get('logger')
        print('init', logger.experiment)
        super().instantiate_trainer()
        
# note: 未实现, trainer_args model_dir
def train_with_gin(model_dir,
                   overwrite=False,
                   trainer_args = None,
                   gin_config_files=None,
                   gin_bindings=None):
    """Trains a model based on the provided gin configuration.

    This function will set the provided gin bindings, call the train() function
    and clear the gin config. Please see train() for required gin bindings.

    Args:
      model_dir: String with path to directory where model output should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      gin_config_files: List of gin config files to load.
      gin_bindings: List of gin bindings to use.
    """
    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []
    
    gin.clear_config()
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    logging.info(gin.operative_config_str())
    # model_path = pathlib.Path(model_dir)
    # # Delete the output directory if it already exists.
    # if model_path.exists():
    #     if overwrite:
    #         shutil.rmtree(model_path)
    #     else:
    #         raise FileExistsError("Directory already exists and overwrite is False.")
    # model_path.mkdir(parents=True, exist_ok=True)

    cli = MyLightningCLI(
        override_args=trainer_args,
        model_class=Train,
        datamodule_class = DataModule,
        save_config_callback=None,
        env_parse=True,
        parser_kwargs={
            "default_config_files": ["cli_training.yaml", "/etc/cli_training.yaml"],
            })
    return cli.model
