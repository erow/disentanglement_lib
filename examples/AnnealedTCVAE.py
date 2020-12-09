# coding=utf-8

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.methods.unsupervised.vae import total_correlation, anneal
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import torch
import gin
import numpy as np

beta = None

# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = "example_output"
overwrite = True
# We save the results in a `vae` subfolder.
path_vae = os.path.join(base_path, "fvae")


@gin.configurable("FractionalVAE")  # This will allow us to reference the model.
class AnnealedTCVAE(vae.BaseVAE):
    """AnnealedTCVAE model."""

    def __init__(self, input_shape, beta=gin.REQUIRED,
                 gamma=gin.REQUIRED, c_max=gin.REQUIRED, iteration_threshold=gin.REQUIRED, **kwargs):
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
        del z_mean, z_logvar, z_sampled
        c = anneal(self.c_max, self.global_step, self.iteration_threshold)

        tc = total_correlation(z_sampled, z_mean, z_logvar)
        return self.gamma * torch.abs(mi - c) + self.beta * tc +


n = 300000
gin_bindings = [
    "encoder.encoder_fn = @conv_group_encoder",
    "model.model = @FractionalVAE()",
    f"model.training_steps = {n}"
]

k = n // 10
stas = np.zeros(n, dtype=np.int)
stas[k:3 * k] = 1
stas[3 * k:6 * k] = 2
stas[6 * k:] = 3
# stas[3*k: 6*k] =1
# stas[6*k: ] =2
# The main training protocol of disentanglement_lib is defined in the
# disentanglement_lib.methods.unsupervised.train module. To configure
# training we need to provide a gin config. For a standard VAE, you may have a
# look at model.gin on how to do this.
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

# 6. Aggregate the results.
# ------------------------------------------------------------------------------
# In the previous steps, we saved the scores to several output directories. We
# can aggregate all the results using the following command.
pattern = os.path.join(base_path,
                       "*/metrics/*/results/aggregate/evaluation.json")
results_path = os.path.join(base_path, "results.json")
aggregate_results.aggregate_results_to_json(
    pattern, results_path)

# 7. Print out the final Pandas data frame with the results.
# ------------------------------------------------------------------------------
# The aggregated results contains for each computed metric all the configuration
# options and all the results captured in the steps along the pipeline. This
# should make it easy to analyze the experimental results in an interactive
# Python shell. At this point, note that the scores we computed in this example
# are not realistic as we only trained the models for a few steps and our custom
# metric always returns 1.
model_results = aggregate_results.load_aggregated_json_results(results_path)
print(model_results)
