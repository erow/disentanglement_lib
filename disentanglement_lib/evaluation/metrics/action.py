# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
import numpy as np
from six.moves import range
from sklearn import linear_model
import gin.tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from disentanglement_lib.data.ground_truth.ground_truth_data import *


def build_model(input_size):
    model = keras.Sequential([
        layers.Dense(1024, activation=tf.nn.relu, input_shape=[input_size]),
        layers.Dense(1024, activation=tf.nn.relu),
        layers.Dense(input_size)
    ])

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_squared_error'])
    return model


class ActionSequence(keras.utils.Sequence):
    def __init__(self, ground_truth_data: GroundTruthData,
                 factor_index,
                 representation_function):
        super().__init__()
        self.data = ground_truth_data
        self.action_index = factor_index
        self.representation_function = representation_function
        self.length = 1000
        self.p = 0.2  # 遮住20%的序列

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        del item
        factor = sample_factor(self.data)
        obs = action(self.data, factor, self.action_index)
        representation = self.representation_function(obs)
        seq_len, dim = representation.shape
        mask_rep = representation * (np.random.rand(seq_len, 1) > self.p)

        mask_seq = mask_rep.reshape(1, -1)
        sequence = representation.reshape(1, -1)
        return mask_seq, sequence


@gin.configurable(
    "action",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_action(ground_truth_data: GroundTruthData,
                   representation_function,
                   random_state,
                   artifact_dir=None):
    """Computes action.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      random_state: Numpy random state used for randomness.
      artifact_dir: Optional path to directory where artifacts can be saved.

    Returns:
      Dictionary with scores:
    """
    del artifact_dir
    scores_dict = {}
    score = 0
    dim =10
    for i in range(ground_truth_data.num_factors):
        logging.info(f"Computing action {i}.")

        model = build_model(dim * ground_truth_data.factors_num_values[i])
        action_sequence = ActionSequence(ground_truth_data, i, representation_function)
        history = model.fit(action_sequence, epochs=20)
        scores_dict[f'action_history_{i}'] = history.history['loss']
        score += history.history['loss'][-1]
    scores_dict['action_score'] = score
    return scores_dict
