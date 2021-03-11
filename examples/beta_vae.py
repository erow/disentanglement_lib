# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import torch
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import gin

base_path = "example_output"
overwrite = True
# We save the results in a `vae` subfolder.
path_vae = os.path.join(base_path, "vae")

# 2. Train beta-VAE from the configuration at model.gin.
train.train_with_gin(os.path.join(path_vae, "model"), overwrite, ["model.gin"])
gin.parse_config_file("model.gin")
# 3. Extract the mean representation for both of these models.
for path in [path_vae]:
    representation_path = os.path.join(path, "representation")
    model_path = os.path.join(path, "model")
    postprocess.postprocess(model_path, representation_path, overwrite)

# 4. Compute the Mutual Information Gap (already implemented) for both models.
gin_bindings = [
    "evaluation.evaluation_fn = @mig",
    "dataset.name='dsprites_full'",
    "evaluation.random_seed = 0",
    "mig.num_train=1000",
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20"
]
for path in [path_vae]:
    result_path = os.path.join(path, "metrics", "mig")
    representation_path = os.path.join(path, "representation")
    evaluate.evaluate_with_gin(
        representation_path, result_path, overwrite, gin_bindings=gin_bindings)
