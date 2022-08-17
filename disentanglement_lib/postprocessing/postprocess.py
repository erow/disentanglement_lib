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

"""Postprocessing step that extracts representation from trained model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pathlib
import shutil
import time

import wandb
from torch.utils.data import DataLoader, Dataset

from disentanglement_lib.data import named_data
from disentanglement_lib.methods.unsupervised.gaussian_encoder_model import GaussianModel
from disentanglement_lib.utils import results
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import gin


def postprocess_with_gin(model_dir,
                         output_dir,
                         overwrite=False,
                         gin_config_files=None,
                         gin_bindings=None):
    """Postprocess a trained model based on the provided gin configuration.

    This function will set the provided gin bindings, call the postprocess()
    function and clear the gin config. Please see the postprocess() for required
    gin bindings.

    Args:
      model_dir: String with path to directory where the model is saved.
      output_dir: String with the path where the representation should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      gin_config_files: List of gin config files to load.
      gin_bindings: List of gin bindings to use.
    """
    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    postprocess(model_dir, output_dir, overwrite)
    gin.clear_config()


@gin.configurable(
    "postprocess", denylist=["model_dir", "output_dir", "overwrite"])
def postprocess(model_dir,
                output_dir,
                overwrite=False,
                random_seed=0,
                name=""):
    """Loads a trained Gaussian encoder and extracts representation.

    Args:
      model_dir: String with path to directory where the model is saved.
      output_dir: String with the path where the representation should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
        for examples).
      random_seed: Integer with random seed used for postprocessing (may be
        unused).
      name: Optional string with name of the representation (can be used to name
        representations).
    """
    # We do not use the variable 'name'. Instead, it can be used to name
    # representations as it will be part of the saved gin config.
    del name
    np.random.seed(random_seed)
    # Delete the output directory if it already exists.
    if os.path.isdir(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise ValueError("Directory already exists and overwrite is False.")
    pathlib.Path(output_dir).mkdir(parents=True)
    # Set up timer to keep track of elapsed time in results.
    experiment_timer = time.time()

    from disentanglement_lib.methods.unsupervised.train import Train
    train = Train()
    dl = train.train_dataloader()
    model = train.ae
    model.load_state_dict(torch.load(f"{model_dir}/model.pt"))
    model = model.cuda()
    # Run the postprocessing function which returns a transformation function
    # that can be used to create the representation from the mean and log
    # variance of the Gaussian distribution given by the encoder. Also returns
    # path to a checkpoint if the transformation requires variables.

    mean, std, factors = [], [], []
    with torch.no_grad():
        for imgs, labels in dl:
            imgs = imgs.cuda()
            mu, logvar = model.encode(imgs)
            mu, logvar = mu.cpu(), logvar.cpu()
            mean.append(mu.numpy())
            std.append((logvar / 2).exp().numpy())
            factors.append(labels.numpy())
    mean = np.concatenate(mean)
    std = np.concatenate(std)
    factors = np.concatenate(factors)
    representation = {'mean': mean,
                      'std': std,
                      'factor': factors}
    file_path = os.path.join(output_dir, 'representation.npy')
    np.save(file_path, representation)
    # We first copy over all the prior results and configs.
    results_dir = os.path.join(output_dir, "results")
    results_dict = dict(elapsed_time=time.time() - experiment_timer)
    results.update_result_directory(results_dir, "postprocess", results_dict)
    return representation
