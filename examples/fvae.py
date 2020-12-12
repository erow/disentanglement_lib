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
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import torch
import gin
import gin
import numpy as np

beta = None


@gin.configurable("conv_group_encoder", allowlist=[])
def conv_group_encoder(input_tensor, num_latent, is_training=True):
    """
    Args:
      input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
        build encoder on.
      num_latent: Number of latent variables to output.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      means: Output tensor of shape (batch_size, num_latent) with latent variable
        means.
      log_var: Output tensor of shape (batch_size, num_latent) with latent
        variable log variances.
    """
    global beta
    BETA = tf.constant([100, 40, 20, 4], dtype=tf.float32)
    STAGE = tf.constant(stas, dtype=tf.int32)

    step = tf.train.get_global_step()
    if step is None:
        stage = tf.constant(3, name='stage')
    else:
        stage = tf.gather(STAGE, step, name='stage')
        beta = tf.gather(BETA, stage, name='beta')
        tf.summary.scalar('stage', stage)
        tf.summary.scalar('beta', beta)

    mean_list, log_var_list = [], []
    dim = num_latent // 4
    for i in range(4):
        mu, lvar = conv_encoder(input_tensor, dim, is_training)
        mu = tf.cond(i <= stage,
                     lambda: mu,
                     lambda: tf.stop_gradient(mu))
        lvar = tf.cond(i <= stage,
                       lambda: lvar,
                       lambda: tf.stop_gradient(lvar))

        lr = tf.cond((tf.equal(stage, tf.Variable(i, dtype=tf.int32))), lambda: tf.constant(1.0),
                     lambda: tf.constant(0.1))
        lr_fun = lr_mult(lr)
        mean_list.append(lr_fun(mu))
        log_var_list.append(lr_fun(lvar))

    means = tf.concat(mean_list, 1, name='means')
    log_var = tf.concat(log_var_list, 1, name='log_var')

    # means = tf.concat(mean_list,1)
    # log_var = tf.concat(log_var_list,1)
    return means, log_var


# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = "example_output"
overwrite = True
# We save the results in a `vae` subfolder.
path_vae = os.path.join(base_path, "fvae")


@gin.configurable("FractionalVAE")  # This will allow us to reference the model.
class FractionalVAE(vae.BaseVAE):
    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        # This is how we customize BaseVAE. To learn more, have a look at the
        # different models in vae.py.
        del z_mean, z_logvar, z_sampled
        return kl_loss * beta


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
