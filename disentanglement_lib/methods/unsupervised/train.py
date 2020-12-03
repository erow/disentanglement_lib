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
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from disentanglement_lib.methods.unsupervised import vae  # pylint: disable=unused-import
from disentanglement_lib.utils import results
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gin.torch
import pathlib, shutil
import pytorch_lightning as pl


class torch_dataset(Dataset):
    def __init__(self, tf_dataset, random_seed):
        self.random_state = np.random.RandomState(random_seed)
        self.tf_dataset = tf_dataset

    def __len__(self):
        return np.prod(self.tf_dataset.factors_num_values)

    def __getitem__(self, item):
        factors, observations = self.tf_dataset.sample(1, self.random_state)
        observations = torch.tensor(observations, dtype=torch.float32)
        factors = torch.tensor(factors, dtype=torch.float32)
        observations = observations.permute(0, 3, 1, 2)
        return observations[0], factors

@gin.configurable("model", blacklist=["model_dir", "overwrite"])
def train(model_dir,
          overwrite=False,
          model=gin.REQUIRED,
          training_steps=gin.REQUIRED,
          random_seed=gin.REQUIRED,
          batch_size=gin.REQUIRED,
          eval_steps=1000,
          name="",
          model_num=None):
    """Trains the estimator and exports the snapshot and the gin config.

    The use of this function requires the gin binding 'dataset.name' to be
    specified as that determines the data set used for training.

    Args:
      model_dir: String with path to directory where model output should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      model: GaussianEncoderModel that should be trained and exported.
      training_steps: Integer with number of training steps.
      random_seed: Integer with random seed used for training.
      batch_size: Integer with the batch size.
      eval_steps: Optional integer with number of steps used for evaluation.
      name: Optional string with name of the model (can be used to name models).
      model_num: Optional integer with model number (can be used to identify
        models).
    """
    # We do not use the variables 'name' and 'model_num'. Instead, they can be
    # used to name results as they will be part of the saved gin config.
    del name, model_num

    model_path = pathlib.Path(model_dir)
    # Delete the output directory if it already exists.
    if model_path.exists():
        if overwrite:
            shutil.rmtree(model_path)
        else:
            raise ValueError("Directory already exists and overwrite is False.")

    # Create a numpy random state. We will sample the random seeds for training
    # and evaluation from this.

    # Obtain the dataset. tf format
    dataset = named_data.get_named_ground_truth_data()
    tf_data_shape = dataset.observation_shape
    dataset = torch_dataset(dataset, random_seed)
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=8)

    # We create a TPUEstimator based on the provided model. This is primarily so
    # that we could switch to TPU training in the future. For now, we train
    # locally on GPUs.
    save_checkpoints_steps = training_steps // 10
    input_shape = [tf_data_shape[2], tf_data_shape[0], tf_data_shape[1]]
    autoencoder = model(input_shape)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{global_step:05d}',
        period=save_checkpoints_steps)
    trainer = pl.Trainer(max_steps=training_steps, num_processes=4, gpus=1,
                         default_root_dir=model_dir,
                         callbacks=[checkpoint_callback])
    trainer.fit(autoencoder, dl)
    trainer.save_checkpoint("result.ckpt")

    # Save model as a TFHub module.


    # Save the results. The result dir will contain all the results and config
    # files that we copied along, as we progress in the pipeline. The idea is that
    # these files will be available for analysis at the end.


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
    print(gin.operative_config_str())
    train(model_dir, overwrite)
    gin.clear_config()
