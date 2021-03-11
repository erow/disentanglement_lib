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
from disentanglement_lib.methods.unsupervised import model
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import torch
from torch import nn
import gin
import numpy as np
import wandb

from disentanglement_lib.visualize.visualize_model import visualize


def run_model(model_dir, gin_bindings, train_model, overwrite=True):
    def init_model(input_shape):
        global phase
        print("load ", phase)
        if phase == 0:
            model = train_model(input_shape)
        else:
            model = load_model(model_dir, f"{phase - 1}.pth")
        wandb.watch(model, log="all")
        return model

    gin.parse_config_files_and_bindings(["model.gin"], gin_bindings)
    train.train(model_dir, False, init_model)
    copyfile(os.path.join(model_dir, "ckp.pth"), os.path.join(model_dir, f"{phase}.pth"))
    gin.clear_config()



# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = "example_output"
# We save the results in a `vae` subfolder.
output_directory = os.path.join(base_path, "fvae")

# The main training protocol of disentanglement_lib is defined in the
# disentanglement_lib.methods.unsupervised.train module. To configure
# training we need to provide a gin config. For a standard VAE, you may have a
# look at model.gin on how to do this.

model_dir = os.path.join(output_directory, "model")

wandb.init(job_type='train', project='examples')

phase = 0
bs = 64
steps = 11520 * 64 / bs * 1
gin_bindings = [
    f"train.batch_size ={bs}",
    f"fractional_conv_encoder.active={phase}",
    "model.encoder_fn = @fractional_conv_encoder",
    "model.num_latent = 12",
    "vae.beta=120",
    f"train.training_steps={steps}"
]
run_model(model_dir, gin_bindings, model.BetaVAE)

# phase = 1
# steps = steps * 2
# gin_bindings = [
#     f"train.batch_size ={bs}",
#     f"fractional_conv_encoder.active={phase}",
#     "model.encoder_fn = @fractional_conv_encoder",
#     "model.num_latent = 12",
#     "vae.beta=20",
#     f"train.training_steps={steps}"
# ]
# run_model(model_dir,gin_bindings,vae.BetaVAE)

# 3. Extract the mean representation for both of these models.
# ------------------------------------------------------------------------------
# To compute disentanglement metrics, we require a representation function that
# takes as input an image and that outputs a vector with the representation.
# We extract the mean of the encoder from both models using the following code.

representation_path = os.path.join(output_directory, "representation")
postprocess_gin = ["postprocess.gin"]  # This contains the settings.
# postprocess.postprocess_with_gin defines the standard extraction protocol.
postprocess.postprocess_with_gin(model_dir, representation_path, True,
                                 postprocess_gin)

gin_bindings = [
    "evaluation.evaluation_fn = @mig",
    "dataset.name='auto'",
    "evaluation.random_seed = 0",
    "mig.num_train=1000",
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20"
]

result_path = os.path.join(output_directory, "metrics", "mig")
evaluate.evaluate_with_gin(
    representation_path, result_path, True, gin_bindings=gin_bindings)

visualize(model_dir, os.path.join(output_directory, "visualization"), True)
