# coding=utf-8

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import model
from disentanglement_lib.methods.unsupervised.model import total_correlation, anneal
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import torch
import gin
import numpy as np

from disentanglement_lib.visualize.visualize_model import visualize
from examples.TC import *
import wandb

beta = None

base_path = "example_output"
overwrite = True
# We save the results in a `vae` subfolder.
path_vae = os.path.join(base_path, "AnnealedTCVAE")


@gin.configurable("AnnealedTCVAE")  # This will allow us to reference the model.
class AnnealedTCVAE(model.BaseVAE):
    """AnnealedTCVAE model."""

    def __init__(self, input_shape,
                 beta=gin.REQUIRED,
                 gamma=gin.REQUIRED,
                 c_max=gin.REQUIRED,
                 iteration_threshold=gin.REQUIRED, **kwargs):
        """
        Args:
          gamma: Hyperparameter for the regularizer.
          c_max: Maximum capacity of the bottleneck.
          iteration_threshold: How many iterations to reach c_max.
        """
        super().__init__(input_shape, beta=beta, gamma=gamma,
                         c_max=c_max,
                         iteration_threshold=iteration_threshold, **kwargs)
        self.beta = beta
        self.gamma = gamma
        self.c_max = c_max
        self.iteration_threshold = iteration_threshold

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        c = anneal(self.gamma, self.global_step, 100000)

        log_qzCx = model.gaussian_log_density(z_sampled, z_mean, z_logvar).sum(1)
        log_pz = model.gaussian_log_density(z_sampled,
                                            torch.zeros_like(z_mean),
                                            torch.zeros_like(z_mean)).sum(1)
        _, log_qz, log_qz_product = model.decompose(z_sampled, z_mean, z_logvar)

        mi = torch.mean(log_qzCx - log_qz)
        tc = torch.mean(log_qz - log_qz_product)
        dw_kl_loss = torch.mean(log_qz_product - log_pz)
        self.summary['mi'] = mi
        self.summary['tc'] = tc
        self.summary['dw'] = dw_kl_loss
        self.summary['c'] = c
        return 500 * (-self.gamma + c - mi).abs() + self.beta * tc + dw_kl_loss


@gin.configurable("AnnealedTCVAE1")  # This will allow us to reference the model.
class AnnealedTCVAE1(model.BaseVAE):
    """AnnealedTCVAE model."""

    def __init__(self, input_shape,
                 beta=gin.REQUIRED,
                 gamma=gin.REQUIRED,
                 c_max=gin.REQUIRED,
                 iteration_threshold=gin.REQUIRED, **kwargs):
        """
        Args:
          gamma: Hyperparameter for the regularizer.
          c_max: Maximum capacity of the bottleneck.
          iteration_threshold: How many iterations to reach c_max.
        """
        super().__init__(input_shape, beta=beta, gamma=gamma,
                         c_max=c_max,
                         iteration_threshold=iteration_threshold, **kwargs)
        self.beta = beta
        self.gamma = gamma
        self.N = 3 * 6 * 40 * 32 * 32

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        c = self.gamma - self.gamma * anneal(1, self.global_step, 1000)

        log_qzCx = model.gaussian_log_density(z_sampled, z_mean, z_logvar).sum(1)
        zeros = torch.zeros_like(z_mean)
        log_pz = model.gaussian_log_density(z_sampled, zeros, zeros).sum(1)
        _, log_qz, log_qz_product = model.decompose(z_sampled, z_mean, z_logvar)
        # 常数矫正，但是常数不影响结果
        batch_size = z_mean.size(0)
        log_qz = log_qz - np.log(batch_size * self.N)
        log_qz_product = log_qz_product - math.log(batch_size * self.N) * z_mean.size(1)

        mi = torch.mean(log_qzCx - log_qz)
        tc = torch.mean(log_qz - log_qz_product)
        dw_kl_loss = torch.mean(log_qz_product - log_pz)
        self.summary['mi'] = mi
        self.summary['tc'] = tc
        self.summary['dw'] = dw_kl_loss
        self.summary['c'] = c
        return mi * c + self.beta * tc + dw_kl_loss


gin_bindings = [
    "train.model = @AnnealedTCVAE",
    "AnnealedTCVAE1.beta=6",
    "AnnealedTCVAE1.gamma=5",
    "AnnealedTCVAE1.c_max=50.",
    "AnnealedTCVAE1.iteration_threshold=1000",
    "AnnealedTCVAE.beta=6",
    "AnnealedTCVAE.gamma=5",
    "AnnealedTCVAE.c_max=50.",
    "AnnealedTCVAE.iteration_threshold=1000"
]

# The main training protocol of disentanglement_lib is defined in the
# disentanglement_lib.methods.unsupervised.train module. To configure
# training we need to provide a gin config. For a standard VAE, you may have a
# look at model.gin on how to do this.
wandb.init(project='examples', reinit=True)

train.train_with_gin(
    os.path.join(path_vae, "model"), overwrite, ["model.gin"],
    gin_bindings)

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

# 4. Compute the Mutual Information Gap (already implemented) for both models.
# ------------------------------------------------------------------------------
# The main evaluation protocol of disentanglement_lib is defined in the
# disentanglement_lib.evaluation.evaluate module. Again, we have to provide a
# gin configuration. We could define a .gin config file; however, in this case
# we show how all the configuration settings can be set using gin bindings.
# We use the Mutual Information Gap (with a low number of samples to make it
# faster). To learn more, have a look at the different scores in
# disentanglement_lib.evaluation.evaluate.metrics and the predefined .gin
# configuration files in
# disentanglement_lib/config/unsupervised_study_v1/metrics_configs/(...).
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
