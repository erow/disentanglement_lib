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
import tensorflow as tf
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


def lr_mult(alpha):
    '''
    https://vimsky.com/article/4317.html
    :param alpha:
    :return:
    '''

    @tf.custom_gradient
    def _lr_mult(x):
        def grad(dy):
            return dy * alpha * tf.ones_like(x)

        return x, grad

    return _lr_mult


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
    step = tf.train.get_global_step()
    if step is None:  # evaluation
        stage = tf.constant(-1, dtype=tf.int64)
    else:
        stage_steps = (gin.query_parameter('model.training_steps')) // 4  #
        stage = step // stage_steps

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

        lr = tf.cond((tf.equal(stage, tf.Variable(i, dtype=tf.int64))), lambda: tf.constant(1.0),
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
    def __init__(self, beta=gin.REQUIRED):
        self.beta = tf.constant(beta)

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        # This is how we customize BaseVAE. To learn more, have a look at the
        # different models in vae.py.
        del z_mean, z_logvar, z_sampled
        step = tf.train.get_global_step()
        if step is None:  # evaluation
            stage = tf.constant(0, dtype=tf.int64)
        else:
            stage_steps = (gin.query_parameter('model.training_steps')) // 4  #
            stage = step // stage_steps
        # beta = tf.to_float(self.beta[stage])
        beta = 100 * (1.1 - tf.to_float(step) / gin.query_parameter('model.training_steps'))
        return kl_loss * beta

gin_bindings = [
    "encoder.encoder_fn = @conv_group_encoder",
    "model.model = @FractionalVAE()",
    "FractionalVAE.beta = [100,32,5,1]",
]
# The main training protocol of disentanglement_lib is defined in the
# disentanglement_lib.methods.unsupervised.train module. To configure
# training we need to provide a gin config. For a standard VAE, you may have a
# look at model.gin on how to do this.
train.train_with_gin(
    os.path.join(path_vae, "model"), overwrite, ["model.gin"],
    gin_bindings)
