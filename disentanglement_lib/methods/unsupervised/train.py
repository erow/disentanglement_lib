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
from disentanglement_lib.data.ground_truth.ground_truth_data import TorchData
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from disentanglement_lib.methods.unsupervised import vae  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised.gaussian_encoder_model import GaussianModel, load
from disentanglement_lib.utils import results
from disentanglement_lib.visualize import visualize_model
from .optimizer import *
import numpy as np
# import torch
from torch.utils.data import Dataset, DataLoader
import gin.torch
import pathlib, shutil
import wandb


@gin.configurable("train", blacklist=["model_dir", "overwrite"])
def train(model_dir,
          overwrite=False,
          model=gin.REQUIRED,
          training_steps=gin.REQUIRED,
          random_seed=gin.REQUIRED,
          batch_size=gin.REQUIRED,
          opt_name=torch.optim.Adam,
          lr=1e-4,
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
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)
    model_path = pathlib.Path(model_dir)
    # Delete the output directory if it already exists.
    if model_path.exists():
        if overwrite:
            shutil.rmtree(model_path)
        else:
            print("Directory already exists and overwrite is False.")
    model_path.mkdir(parents=True, exist_ok=True)
    # Create a numpy random state. We will sample the random seeds for training
    # and evaluation from this.

    # Obtain the dataset. tf format
    dataset = named_data.get_named_ground_truth_data()
    tf_data_shape = dataset.observation_shape
    torch_ds = TorchData(dataset)
    dl = DataLoader(torch_ds,
                    batch_size=batch_size,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=True)

    test_dl = DataLoader(dataset,
                         batch_size=batch_size,
                         num_workers=0,
                         pin_memory=True)
    # Set up time to keep track of elapsed time in results.
    experiment_timer = time.time()

    # We create a TPUEstimator based on the provided model. This is primarily so
    # that we could switch to TPU training in the future. For now, we train
    # locally on GPUs.
    save_checkpoints_steps = training_steps // 10
    input_shape = [tf_data_shape[2], tf_data_shape[0], tf_data_shape[1]]
    autoencoder = model(input_shape)

    device = 'cuda'

    autoencoder.to(device).train()
    from disentanglement_lib.methods.shared.architectures import fractional_conv_encoder
    if isinstance(autoencoder.encode, fractional_conv_encoder):
        opt = opt_name(autoencoder.decode.parameters(), lr)
        for conv in autoencoder.encode.convs:
            if conv.activate:
                opt.add_param_group({'params': conv.parameters(), 'lr': lr})
            else:
                opt.add_param_group({'params': conv.parameters(), 'lr': lr * 0})
    else:
        opt = opt_name(autoencoder.parameters(), lr)

    global_step = 0

    summary = {}
    while global_step < training_steps:
        for imgs, labels in dl:
            imgs, labels = imgs.to(device), labels.to(device)
            autoencoder.global_step = global_step
            summary = autoencoder.model_fn(imgs, labels)
            loss = summary['loss']

            if (global_step + 1) % save_checkpoints_steps == 0:
                autoencoder.save(model_dir, f'ckp-{global_step:06d}.pth')
                mean, std, factors = [], [], []
                with torch.no_grad():
                    for imgs, labels in test_dl:
                        imgs = imgs.cuda()
                        mu, logvar = autoencoder.encode(imgs)
                        mean.append(mu)
                        factors.append(labels)
                mean = torch.cat(mean, 0).cpu()
                fig = visualize_model.plot_latent_vs_ground(mean, latnt_sizes=[16, 16])
                summary['projection'] = wandb.Image(fig)

            wandb.log(summary)

            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step = global_step + 1


            if global_step >= training_steps:
                break

    # Save model as a TFHub module.
    autoencoder.eval()
    autoencoder.save(model_dir)
    wandb.save(f'{model_dir}/ckp.pth')

    # Save the results. The result dir will contain all the results and config
    # files that we copied along, as we progress in the pipeline. The idea is that
    # these files will be available for analysis at the end.
    summary.pop('projection')
    results_dir = os.path.join(model_dir, "results")
    results_dict = summary
    results_dict["elapsed_time"] = time.time() - experiment_timer
    results.update_result_directory(results_dir, "train", results_dict)


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
