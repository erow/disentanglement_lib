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
import tensorflow.compat.v1 as tf
import gin.tf
import gin


def conv_encoder(input_tensor, num_latent, is_training=True):
    del is_training
    filters = 16
    e1 = tf.layers.conv2d(
        inputs=input_tensor,
        filters=filters,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same"
    )
    e2 = tf.layers.conv2d(
        inputs=e1,
        filters=filters,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
    )
    e3 = tf.layers.conv2d(
        inputs=e2,
        filters=filters * 2,
        kernel_size=2,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
    )
    e4 = tf.layers.conv2d(
        inputs=e3,
        filters=filters * 2,
        kernel_size=2,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
    )
    flat_e4 = tf.layers.flatten(e4)
    e5 = tf.layers.dense(flat_e4, 256, activation=tf.nn.relu, )
    means = tf.layers.dense(e5, num_latent, activation=None)
    log_var = tf.layers.dense(e5, num_latent, activation=None)
    return means, log_var


@gin.configurable("conv_group_encoder", whitelist=[])
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
    del is_training

    e1 = tf.layers.separable_conv2d(
        inputs=input_tensor,
        depth_multiplier=4,
        filters=16,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
        name="e1",

    )
    e2 = tf.layers.separable_conv2d(
        inputs=e1,
        depth_multiplier=4,
        filters=16,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
        name="e2",
    )
    e3 = tf.layers.separable_conv2d(
        inputs=e2,
        depth_multiplier=4,
        filters=32,
        kernel_size=2,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
        name="e3",
    )
    e4 = tf.layers.separable_conv2d(
        inputs=e3,
        depth_multiplier=4,
        filters=32,
        kernel_size=2,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
        name="e4",
    )
    group_e4 = tf.split(e4, 4, axis=1)
    mean_list, log_var_list = [], []
    for i, sep_e4 in enumerate(group_e4):
        flat_e4 = tf.layers.flatten(sep_e4)
        e5 = tf.layers.dense(flat_e4, 256, activation=tf.nn.relu, name=f"e5_{i}")
        mean_list.append(tf.layers.dense(e5, num_latent // 4, activation=None, name=f"means_{i}"))
        log_var_list.append(tf.layers.dense(e5, num_latent // 4, activation=None, name=f"log_var_{i}"))

    session = tf.get_default_session()
    step = tf.train.get_global_step()
    stage_steps = (gin.query_parameter('model.training_steps')) // 4  #
    stage = 0

    means = tf.concat(mean_list[:stage + 1] +
                      [tf.stop_gradient(t) for t in mean_list[stage + 1:]]
                      , 1)
    log_var = tf.concat(log_var_list[:stage + 1] +
                        [tf.stop_gradient(t) for t in log_var_list[stage + 1:]]
                        , 1)
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

gin_bindings = [
    "encoder.encoder_fn = @conv_group_encoder"
]
# The main training protocol of disentanglement_lib is defined in the
# disentanglement_lib.methods.unsupervised.train module. To configure
# training we need to provide a gin config. For a standard VAE, you may have a
# look at model.gin on how to do this.
train.train_with_gin(
    os.path.join(path_vae, "model"), overwrite, ["model.gin"],
    gin_bindings)
