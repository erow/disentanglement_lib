# coding=utf-8

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from shutil import copyfile

from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
# from disentanglement_lib.methods.shared.architectures import conv_encoder
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.methods.unsupervised.gaussian_encoder_model import load
from disentanglement_lib.methods.unsupervised.vae import load_model
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import torch
from torch import nn
import gin
import numpy as np
import wandb

beta = None
wandb.init(job_type='train', project='examples')

# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = "example_output"
overwrite = True
# We save the results in a `vae` subfolder.
path_vae = os.path.join(base_path, "fvae")

phase = 0
steps = 11520 * 2
gin_bindings = [
    f"fractional_conv_encoder.active={phase}",
    "model.encoder_fn = @fractional_conv_encoder",
    "model.num_latent = 12",
    "vae.beta=100",
    f"train.training_steps={steps}"
]

# The main training protocol of disentanglement_lib is defined in the
# disentanglement_lib.methods.unsupervised.train module. To configure
# training we need to provide a gin config. For a standard VAE, you may have a
# look at model.gin on how to do this.

model_dir = os.path.join(path_vae, "model")
gin.parse_config_files_and_bindings(["model.gin"], gin_bindings)
train_model = vae.BetaVAE


def init_model(input_shape):
    if phase == 0:
        return train_model(input_shape)
    else:
        return load_model(model_dir, f"{phase}.pth")


train.train(model_dir, False, init_model)
copyfile(os.path.join(model_dir, "ckp.pth"), os.path.join(model_dir, f"{phase}.pth"))
gin.clear_config()

# 3. Extract the mean representation for both of these models.
# ------------------------------------------------------------------------------
# To compute disentanglement metrics, we require a representation function that
# takes as input an image and that outputs a vector with the representation.
# We extract the mean of the encoder from both models using the following code.
for path in [path_vae]:
    representation_path = os.path.join(path, "representation")
    model_path = os.path.join(path, "model")
    postprocess_gin = ["postprocess.gin"]  # This contains the settings.
    # postprocess.postprocess_with_gin defines the standard extraction protocol.
    postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                     postprocess_gin)

gin_bindings = [
    "evaluation.evaluation_fn = @mig",
    "dataset.name='auto'",
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
