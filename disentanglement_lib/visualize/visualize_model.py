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

"""Visualization module for disentangled representations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numbers
import os
import pathlib
import shutil

import torch

from disentanglement_lib.utils import results
from disentanglement_lib.utils.hub import get_model, retrive_model
from disentanglement_lib.data.named_data import get_named_ground_truth_data
from disentanglement_lib.visualize import visualize_util
from disentanglement_lib.visualize.visualize_irs import vis_all_interventional_effects
import numpy as np
from scipy import stats
from six.moves import range
import brewer2mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gin

bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
colors = bmap.mpl_colors
VAR_THRESHOLD = 5e-2


def sigmoid(x):
    if isinstance(x, torch.Tensor):
        return sigmoid(x.data.numpy())
    return stats.logistic.cdf(x)


def tanh(x):
    if isinstance(x, torch.Tensor):
        return tanh(x.numpy())
    return np.tanh(x) / 2. + .5


def visualize(model_dir,
              output_dir,
              overwrite=False):
    """Takes trained model from model_dir and visualizes it in output_dir.

    Args:
      model_dir: Path to directory where the trained model is saved.
      output_dir: Path to output directory.
      overwrite: Boolean indicating whether to overwrite output directory.
      num_animations: Integer with number of distinct animations to create.
      num_frames: Integer with number of frames in each animation.
      fps: Integer with frame rate for the animation.
      num_points_irs: Number of points to be used for the IRS plots.
    """
    # Fix the random seed for reproducibility.
    random_state = np.random.RandomState(0)

    # Create the output directory if necessary.
    if os.path.isdir(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise ValueError("Directory already exists and overwrite is False.")
    # Automatically set the proper data set if necessary. We replace the active
    # gin config as this will lead to a valid gin config file where the data set
    # is present.
    # Obtain the dataset name from the gin config of the previous step.
    from exps.decrement import Decrement
    model = retrive_model(model_dir,model_fn=Decrement)
    # model = retrive_model(model_dir)
    num_latent = model.num_latent

    # Automatically infer the activation function from gin config.
    try:
        activation_str = gin.query_parameter["reconstruction_loss.activation"]
    except:
        activation_str = "'logits'"

    if activation_str == "'logits'" or activation_str == None:
        activation = sigmoid
    elif activation_str == "'tanh'":
        activation = tanh
    else:
        raise ValueError(
            "Activation function  could not be infered from gin config.")


    # Save samples.
    _encoder, _decoder = model.convert()
    dataset = get_named_ground_truth_data()
    visualize_reconstructions(output_dir, dataset, model, activation=activation)
    visualize_samples(output_dir, num_latent, _decoder, activation=activation)
    visualize_traversal(output_dir, dataset, _encoder, _decoder, activation=activation)
    visualize_intervention(output_dir, dataset, _encoder)
    # Finally, we clear the gin config that we have set.
    gin.clear_config()


def visualize_reconstructions(output_dir, dataset, model,
                              num_pics=64,
                              activation=sigmoid,
                              random_state=np.random.RandomState()):
    # Save reconstructions.
    real_pics = dataset.sample_observations(num_pics, random_state)

    real_pics1 = torch.Tensor(real_pics.transpose((0, 3, 1, 2)))  # convert tf format to torch's
    raw_pics = model(real_pics1)

    pics = activation(raw_pics)
    pics = pics.transpose([0, 2, 3, 1])
    print(real_pics.shape, pics.shape)
    paired_pics = np.concatenate((real_pics, pics), axis=0)
    paired_pics = [paired_pics[i, :, :, :] for i in range(paired_pics.shape[0])]
    results_dir = os.path.join(output_dir, "reconstructions")
    print(results_dir)
    if not os.path.isdir(results_dir):
        pathlib.Path(results_dir).mkdir(parents=True)
    visualize_util.grid_save_images(
        paired_pics, os.path.join(results_dir, "reconstructions.jpg"))


def visualize_samples(output_dir, num_latent, _decoder,
                      activation=sigmoid,
                      random_state=np.random.RandomState()):
    num_pics = 64
    random_codes = random_state.normal(0, 1, [num_pics, num_latent])
    pics = activation(_decoder(random_codes))
    results_dir = os.path.join(output_dir, "sampled")
    if not os.path.isdir(results_dir):
        pathlib.Path(results_dir).mkdir(parents=True)
    visualize_util.grid_save_images(pics,
                                    os.path.join(results_dir, "samples.jpg"))


def visualize_traversal(output_dir, dataset, _encoder, _decoder,
                        num_pics=64,
                        num_animations=5,
                        num_frames=20,
                        fps=10,
                        activation=sigmoid,
                        random_state=np.random.RandomState()):
    # Save latent traversals.
    means, logvars = _encoder(dataset.sample_observations(num_pics, random_state))

    results_dir = os.path.join(output_dir, "traversals")
    if not os.path.isdir(results_dir):
        pathlib.Path(results_dir).mkdir(parents=True)
    for i in range(means.shape[1]):
        pics = activation(
            latent_traversal_1d_multi_dim(_decoder, means[i, :], None))
        file_name = os.path.join(results_dir, "traversals{}.jpg".format(i))
        visualize_util.grid_save_images([pics], file_name)

    # Save the latent traversal animations.
    results_dir = os.path.join(output_dir, "animated_traversals")
    if not os.path.isdir(results_dir):
        pathlib.Path(results_dir).mkdir(parents=True)

    # Cycle through quantiles of a standard Gaussian.
    for i, base_code in enumerate(means[:num_animations]):
        images = []
        for j in range(base_code.shape[0]):
            code = np.repeat(np.expand_dims(base_code, 0), num_frames, axis=0)
            code[:, j] = visualize_util.cycle_gaussian(base_code[j], num_frames)
            images.append(np.array(activation(_decoder(code))))
        filename = os.path.join(results_dir, "std_gaussian_cycle%d.gif" % i)
        visualize_util.save_animation(np.array(images), filename, fps)

    # Cycle through quantiles of a fitted Gaussian.
    for i, base_code in enumerate(means[:num_animations]):
        images = []
        for j in range(base_code.shape[0]):
            code = np.repeat(np.expand_dims(base_code, 0), num_frames, axis=0)
            loc = np.mean(means[:, j])
            total_variance = np.mean(np.exp(logvars[:, j])) + np.var(means[:, j])
            code[:, j] = visualize_util.cycle_gaussian(
                base_code[j], num_frames, loc=loc, scale=np.sqrt(total_variance))
            images.append(np.array(activation(_decoder(code))))
        filename = os.path.join(results_dir, "fitted_gaussian_cycle%d.gif" % i)
        visualize_util.save_animation(np.array(images), filename, fps)

    # Cycle through [-2, 2] interval.
    for i, base_code in enumerate(means[:num_animations]):
        images = []
        for j in range(base_code.shape[0]):
            code = np.repeat(np.expand_dims(base_code, 0), num_frames, axis=0)
            code[:, j] = visualize_util.cycle_interval(base_code[j], num_frames,
                                                       -2., 2.)
            images.append(np.array(activation(_decoder(code))))
        filename = os.path.join(results_dir, "fixed_interval_cycle%d.gif" % i)
        visualize_util.save_animation(np.array(images), filename, fps)

    # Cycle linearly through +-2 std dev of a fitted Gaussian.
    for i, base_code in enumerate(means[:num_animations]):
        images = []
        for j in range(base_code.shape[0]):
            code = np.repeat(np.expand_dims(base_code, 0), num_frames, axis=0)
            loc = np.mean(means[:, j])
            total_variance = np.mean(np.exp(logvars[:, j])) + np.var(means[:, j])
            scale = np.sqrt(total_variance)
            code[:, j] = visualize_util.cycle_interval(base_code[j], num_frames,
                                                       loc - 2. * scale, loc + 2. * scale)
            images.append(np.array(activation(_decoder(code))))
        filename = os.path.join(results_dir, "conf_interval_cycle%d.gif" % i)
        visualize_util.save_animation(np.array(images), filename, fps)

    # Cycle linearly through minmax of a fitted Gaussian.
    for i, base_code in enumerate(means[:num_animations]):
        images = []
        for j in range(base_code.shape[0]):
            code = np.repeat(np.expand_dims(base_code, 0), num_frames, axis=0)
            code[:, j] = visualize_util.cycle_interval(base_code[j], num_frames,
                                                       np.min(means[:, j]),
                                                       np.max(means[:, j]))
            images.append(np.array(activation(_decoder(code))))
        filename = os.path.join(results_dir, "minmax_interval_cycle%d.gif" % i)
        visualize_util.save_animation(np.array(images), filename, fps)


def visualize_intervention(output_dir, dataset, _encoder,
                           num_points_irs=10000,
                           random_state=np.random.RandomState()):
    # Interventional effects visualization.
    factors = dataset.sample_factors(num_points_irs, random_state)
    obs = dataset.sample_observations_from_factors(factors, random_state)

    latents, _ = _encoder(obs)
    vis_all_interventional_effects(factors, latents, os.path.join(output_dir, "interventional_effects"))


def latent_traversal_1d_multi_dim(generator_fn,
                                  latent_vector,
                                  dimensions=None,
                                  values=None,
                                  transpose=False):
    """Creates latent traversals for a latent vector along multiple dimensions.

    Creates a 2d grid image where each grid image is generated by passing a
    modified version of latent_vector to the generator_fn. In each column, a
    fixed dimension of latent_vector is modified. In each row, the value in the
    modified dimension is replaced by a fixed value.

    Args:
      generator_fn: Function that computes (fixed size) images from latent
        representation. It should accept a single Numpy array argument of the same
        shape as latent_vector and return a Numpy array of images where the first
        dimension corresponds to the different vectors in latent_vectors.
      latent_vector: 1d Numpy array with the base latent vector to be used.
      dimensions: 1d Numpy array with the indices of the dimensions that should be
        modified. If an integer is passed, the dimensions 0, 1, ...,
        (dimensions - 1) are modified. If None is passed, all dimensions of
        latent_vector are modified.
      values: 1d Numpy array with the latent space values that should be used for
        modifications. If an integer is passed, a linear grid between -1 and 1
        with that many points is constructed. If None is passed, a default grid is
        used (whose specific design is not guaranteed).
      transpose: Boolean which indicates whether rows and columns of the 2d grid
        should be transposed.

    Returns:
      Numpy array with image.
    """
    if latent_vector.ndim != 1:
        raise ValueError("Latent vector needs to be 1-dimensional.")

    if dimensions is None:
        # Default case, use all available dimensions.
        dimensions = np.arange(latent_vector.shape[0])
    elif isinstance(dimensions, numbers.Integral):
        # Check that there are enough dimensions in latent_vector.
        if dimensions > latent_vector.shape[0]:
            raise ValueError("The number of dimensions of latent_vector is less than"
                             " the number of dimensions requested in the arguments.")
        if dimensions < 1:
            raise ValueError("The number of dimensions has to be at least 1.")
        dimensions = np.arange(dimensions)
    if dimensions.ndim != 1:
        raise ValueError("Dimensions vector needs to be 1-dimensional.")

    if values is None:
        # Default grid of values.
        values = np.linspace(-1., 1., num=11)
    elif isinstance(values, numbers.Integral):
        if values <= 1:
            raise ValueError("If an int is passed for values, it has to be >1.")
        values = np.linspace(-1., 1., num=values)
    if values.ndim != 1:
        raise ValueError("Values vector needs to be 1-dimensional.")

    # We iteratively generate the rows/columns for each dimension as different
    # Numpy arrays. We do not preallocate a single final Numpy array as this code
    # is not performance critical and as it reduces code complexity.
    num_values = len(values)
    row_or_columns = []
    for dimension in dimensions:
        # Creates num_values copy of the latent_vector along the first axis.
        latent_traversal_vectors = np.tile(latent_vector, [num_values, 1])
        # Intervenes in the latent space.
        latent_traversal_vectors[:, dimension] = values
        # Generate the batch of images
        images = generator_fn(latent_traversal_vectors)
        # Adds images as a row or column depending whether transpose is True.
        axis = (1 if transpose else 0)
        row_or_columns.append(np.concatenate(images, axis))
    axis = (0 if transpose else 1)
    return np.concatenate(row_or_columns, axis)


def vis_projection(factors, latents, results_dir):
    std = latents.std(0)
    l1, l2 = np.argsort(std)[-2:]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.scatter(latents[:, l1], latents[:, l2])
    plt.savefig(os.path.join(results_dir, 'projection.png'))


def plot_latent_vs_ground(z,
                          z_inds=None,
                          latnt_sizes=None):
    if latnt_sizes is None:
        latnt_sizes = [3, 6, 40, 32, 32]
    K = z.shape[-1]
    num_factor = len(latnt_sizes)
    qz_means = z.reshape(*(latnt_sizes + [K])).cpu().data
    var = torch.std(qz_means.reshape(-1, K), dim=0).pow(2)

    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, K))

    if z_inds is None:
        z_inds = active_units
    num_active = len(z_inds)
    if num_active == 0:
        return plt.figure()
    fig, axes = plt.subplots(num_active, num_factor,
                             figsize=(num_factor * 3, (num_active + 1) * 3),
                             squeeze=False)  # default is (8,6)

    for j in range(num_factor):
        mean_latent = qz_means.mean(dim=[dim for dim in range(num_factor) if dim != j])
        for ax, i in zip(axes[:, j], z_inds):
            ax.plot(mean_latent[:, i].numpy(), )
            # ax.set_xticks([])
            # ax.set_yticks([])
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
            if i == z_inds[-1]:
                ax.set_xlabel(f'factor_{j}')
            if j == 0:
                ax.set_ylabel(f'z_{i}')
                # ax.yaxis.tick_right()

    fig.text(0.5, 0.03, 'Ground Truth', ha='center')
    fig.text(0.01, 0.5, 'Learned Latent Variables ', va='center', rotation='vertical')

    return fig


def plot_projection_2d(z, latnt_sizes, sample_nums=5):
    K = z.shape[-1]
    num_factor = len(latnt_sizes)
    assert num_factor == 2
    qz_means = z.reshape(*(latnt_sizes + [K])).cpu().data
    var = torch.std(qz_means.reshape(-1, K), dim=0).pow(2)

    _, z_inds = var.topk(2)
    fig = plt.figure()
    for pos in np.random.randint([0, 0], latnt_sizes, [sample_nums, 2]):
        line1 = qz_means[pos[0], :, z_inds]
        plt.plot(line1[:, 0], line1[:, 1], 'r')

        line2 = qz_means[:, pos[1], z_inds]
        plt.plot(line2[:, 0], line2[:, 1], 'g')
        point = qz_means[pos[0], pos[1], z_inds]
        plt.scatter(point[0], point[1])
    return fig
