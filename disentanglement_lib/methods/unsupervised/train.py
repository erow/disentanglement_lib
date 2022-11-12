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

from disentanglement_lib.data import named_data
from disentanglement_lib.data.ground_truth import util
from disentanglement_lib.data.ground_truth.ground_truth_data import *
from disentanglement_lib.methods.shared import architectures, losses
from disentanglement_lib.methods.unsupervised import model 
from torch.utils.data import IterableDataset

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
    def __init__(self, batch_size=256):
        super().__init__()
        self.dataset = named_data.get_named_ground_truth_data()
        obs_np = self.dataset.observation_shape
        self._dims = obs_np[2], obs_np[0], obs_np[1]
        self.batch_size = batch_size
        
    # def add_argparse_args(self, parser):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('dataset_test',default=20,type=int)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, num_workers=4,
                                           shuffle=True, batch_size=self.batch_size, pin_memory=True)

        

@gin.configurable("model", denylist=[])
class PLModel(pl.LightningModule):
    def __init__(self,
                 input_shape = [1, 64, 64],
                 num_latent=10,
                 encoder_fn=architectures.conv_encoder,
                 decoder_fn=architectures.deconv_decoder,
                 regularizers=[],
                 seed = 99,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(config_dict())
        
        self.encode = encoder_fn(input_shape=input_shape, num_latent=num_latent)
        self.decode = decoder_fn(num_latent=num_latent, output_shape=input_shape)
        self.num_latent = num_latent
        self.input_shape = input_shape
        self.summary = {}
        if isinstance(regularizers,list):
            self.regularizers = regularizers
        else:
            self.regularizers = [regularizers]

    def forward(self,x):
        mu,logvar = self.encode(x)
        return self.decode(mu)
 
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

        # for i in range(kl.shape[0]):
        #     self.summary[f"kl/{i}"] = kl[i]

        return loss, self.summary

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),1e-4,betas=[0.9,0.999])
 

    def save_model(self, file, dir):
        file_path = os.path.join(dir, file)
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), file_path)

    def convert(self, device='cpu'):
        """Convert model to fit numpy format inputs and outputs.

        Args:
            device (str, optional): Defaults to 'cpu'.
        """
        def _decoder(latent_vectors,*args):
            with torch.no_grad():
                z = torch.FloatTensor(latent_vectors).to(device)
                imgs = self.decode(z,*args).cpu().numpy()
                return imgs.transpose((0, 2, 3, 1))

        def _encoder(obs,*args):
            with torch.no_grad():
                # if isinstance(obs,torch.Tensor):
                obs = torch.FloatTensor(obs.transpose((0, 3, 1, 2))).to(device)  # convert tf format to torch's
                mu, logvar = self.encode(obs,*args)
                mu, logvar = mu.cpu().numpy(), logvar.cpu().numpy()
                return mu, logvar

        return _encoder, _decoder

    def train_dataloader(self):
        dataset = named_data.get_named_ground_truth_data()
        return torch.utils.data.DataLoader(
            dataset, num_workers=4,
            shuffle=True, batch_size=128, pin_memory=True)
        
class Iterate(IterableDataset):
    def __init__(self,source:GroundTruthData) -> None:
        super(IterableDataset, self).__init__()
        self.source = source
        for k,v in source.__dict__.items():
            self.__dict__[k] = v
        for m in source.__dict__.keys():
             if not m.startswith('__'):
                self.__setattr__(m,getattr(source,m))
        self.length = len(self.source)
        self.index = 0
        self.sample = source.sample
        self.rs = np.random.RandomState(0)


    def __iter__(self):
        return self

    def __next__(self):
        # return self.source[np.random.randint(self.length)]
        factors, obs = self.source.sample(1,self.rs)
        return obs[0].transpose(2,0,1), factors[0]
