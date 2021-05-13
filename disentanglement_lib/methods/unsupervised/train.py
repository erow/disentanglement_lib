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

from disentanglement_lib.methods.unsupervised.evaluate import Evaluate

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.data.ground_truth import util
from disentanglement_lib.data.ground_truth.ground_truth_data import *
from disentanglement_lib.methods.shared import losses
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from disentanglement_lib.methods.unsupervised import model  # pylint: disable=unused-import

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
                 dataset=gin.REQUIRED,
                 model=gin.REQUIRED,
                 training_steps=gin.REQUIRED,
                 random_seed=gin.REQUIRED,
                 batch_size=gin.REQUIRED,
                 opt_name=torch.optim.Adam,
                 lr=5e-4,
                 eval_numbers=1,
                 name="",
                 model_num=None):
        super().__init__()
        self.dir = wandb.run.dir
        self.training_steps = training_steps
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.lr = lr
        self.name = name
        self.model_num = model_num
        self.eval_numbers = eval_numbers

        self.save_hyperparameters(config_dict())
        self.opt_name = opt_name
        self.data = named_data.get_named_ground_truth_data(dataset)
        self.evaluator = Evaluate()
        img_shape = np.array(self.data.observation_shape)[[2, 0, 1]].tolist()
        # img_shape = [1,64,64]
        self.ae = model(img_shape)


    def training_step(self, batch, batch_idx):
        if self.eval_numbers > 0 and \
                (self.global_step + 1) % (self.training_steps // self.eval_numbers) == 0:
	        self.evaluator.evaluate(self.ae)
        x, y = batch
        loss, summary = self.ae.model_fn(x.float(), y, self.global_step)
        self.log_dict(summary)
        return loss

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
        dir = str(self.dir)
        file_path = os.path.join(dir, file)
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.ae.state_dict(), file_path)
        wandb.save(file_path, base_path=dir)



def train_with_gin(model_dir,
                   overwrite=False,
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
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    logging.info(gin.operative_config_str())

    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger()

    model_path = pathlib.Path(model_dir)
    # Delete the output directory if it already exists.
    if model_path.exists():
        if overwrite:
            shutil.rmtree(model_path)
        else:
            raise FileExistsError("Directory already exists and overwrite is False.")
    model_path.mkdir(parents=True, exist_ok=True)

    gpus = torch.cuda.device_count()
    print(logger.experiment.url)
    logger.experiment.save(model.__file__, os.path.dirname(model.__file__))
    pl_model = Train()
    if gpus > 0:
        trainer = pl.Trainer(logger,
                             progress_bar_refresh_rate=0,  # disable progress bar
                             max_steps=pl_model.training_steps,
                             checkpoint_callback=False,
                             gpus=1,
                             default_root_dir=model_dir)
    else:
        trainer = pl.Trainer(logger,
                             progress_bar_refresh_rate=0,
                             max_steps=pl_model.training_steps,
                             checkpoint_callback=False,
                             tpu_cores=8,
                             default_root_dir=model_dir)

    trainer.fit(pl_model)
    # pl_model.save_model('model.pt')
    from disentanglement_lib.utils.results import save_gin
    save_gin(f"{model_dir}/train.gin")
    wandb.save(f"{model_dir}/train.gin", base_path=model_dir)

    gin.clear_config()
    return pl_model
