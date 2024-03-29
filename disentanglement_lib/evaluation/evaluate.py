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

"""Evaluation protocol to compute metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import shutil
import time
import warnings

import torch

from disentanglement_lib.data import named_data
from disentanglement_lib.evaluation import representation_fn  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import beta_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import dci  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import factor_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import fairness  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import irs  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import mig  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import modularity_explicitness  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import reduced_downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import sap_score  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import strong_downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import unified_scores  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import unsupervised_metrics  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import udr
from disentanglement_lib.evaluation.representation_fn import mean_representation
from disentanglement_lib.methods.unsupervised.train import *
from disentanglement_lib.postprocessing.convert_representation import concat_representation
from disentanglement_lib.utils import results
import numpy as np
import gin

from disentanglement_lib.utils.hub import get_model


def evaluate_with_gin(model_dir,
                      output_dir,
                      overwrite=False,
                      gin_config_files=None,
                      gin_bindings=None):
    """Evaluate a representation based on the provided gin configuration.

    This function will set the provided gin bindings, call the evaluate()
    function and clear the gin config. Please see the evaluate() for required
    gin bindings.

    Args:
      model_dir: String with path to directory where the representation is saved.
      output_dir: String with the path where the evaluation should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      gin_config_files: List of gin config files to load.
      gin_bindings: List of gin bindings to use.
    """
    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings,
                                        skip_unknown=True)
    evaluate(model_dir, output_dir, overwrite)
    gin.clear_config()


@gin.configurable(
    "evaluation", denylist=["output_dir", "output_dir", "overwrite"])
def evaluate(checkpoint,
             output_dir,
             overwrite=False,
             representation_fn = mean_representation,
             evaluation_fn=gin.REQUIRED,
             random_seed=gin.REQUIRED,
             name=""):
    """Loads a representation TFHub module and computes disentanglement metrics.

    Args:
      model_dir: String with path to directory where the representation function
        is saved.
      output_dir: String with the path where the results should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      evaluation_fn: Function used to evaluate the representation (see metrics/
        for examples).
      random_seed: Integer with random seed used for training.
      name: Optional string with name of the metric (can be used to name metrics).
    """
    # Delete the output directory if it already exists.
    if os.path.isdir(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise ValueError("Directory already exists and overwrite is False.")

    # Set up time to keep track of elapsed time in results.
    experiment_timer = time.time()

    dataset = named_data.get_named_ground_truth_data()
    shape = dataset.observation_shape


    # Using the learned distribution (mu, std) instead of the model
    # variables = np.load(os.path.join(model_dir, 'representation.npy'), allow_pickle=True)
    # variables = variables[()]
    # distributions = concat_representation(dataset, variables)
    model = get_model(checkpoint)
    _encode, _decoder = model.convert()

    results_dict = evaluation_fn(
        dataset,
        lambda x: mean_representation(_encode(x)),
        random_state=np.random.RandomState(random_seed),
        )

    # Save the results (and all previous results in the pipeline) on disk.
    results_dir = os.path.join(output_dir, "results")
    results_dict["elapsed_time"] = time.time() - experiment_timer
    results.update_result_directory(results_dir, "evaluation", results_dict)
    return results_dict
    


if __name__ == '__main__':
    import gin
    from disentanglement_lib.evaluation import evaluate

    mig_conf = 'disentanglement_lib/config/unsupervised_study_v1/metric_configs/mig.gin'
    gin.parse_config_files_and_bindings(["train.gin", mig_conf], [], False, True)
    evaluate.evaluate('./', 'metric', True)
