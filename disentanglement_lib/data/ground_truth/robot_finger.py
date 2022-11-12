"""
This tar archive contains this README and the datasets used in the paper:

    A. Dittadi, F. Träuble, F. Locatello, M. Wüthrich, V. Agrawal, O. Winther,
    S. Bauer, B. Schölkopf. On the Transfer of Disentangled Representations in
    Realistic Settings. ICLR 2021.

If you use any of these datasets, please cite our paper as:

@inproceedings{
    dittadi2021transfer,
    title={On the Transfer of Disentangled Representations in Realistic Settings},
    author={Andrea Dittadi and Frederik Tr{\"a}uble and Francesco Locatello and Manuel W{\"u}thrich and Vaibhav Agrawal and Ole Winther and Stefan Bauer and Bernhard Sch{\"o}lkopf},
    booktitle={International Conference on Learning Representations},
    year={2021},
}
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

from PIL import Image
from numba import njit
from sklearn.metrics import pairwise_distances

from disentanglement_lib.data.ground_truth import ground_truth_data
import numpy as np


class Finger(ground_truth_data.GroundTruthData):
    """
    There are three modes:
    finger, finger_heldout_colors, finger_real

    There are nine factors, and the number of factors are:
    [ 1,  8,  1, 30, 30, 30, 30, 30, 10]

    The ground-truth factors of variation in the dataset are:
    [('shape', ['cube']),
     ('color',
      [[1.0, 0.0, 0.0],
       [1.0, 0.0, 0.5],
       [1.0, 0.0, 1.0],
       [0.5, 0.0, 1.0],
       [0.0, 0.5, 1.0],
       [0.0, 1.0, 1.0],
       [0.0, 1.0, 0.5],
       [0.0, 1.0, 0.0]]),
     ('size', [0.065]),
     ('joint0',
      array([-0.65      , -0.60517241, -0.56034483, -0.51551724, -0.47068966,
             -0.42586207, -0.38103448, -0.3362069 , -0.29137931, -0.24655172,
             -0.20172414, -0.15689655, -0.11206897, -0.06724138, -0.02241379,
              0.02241379,  0.06724138,  0.11206897,  0.15689655,  0.20172414,
              0.24655172,  0.29137931,  0.3362069 ,  0.38103448,  0.42586207,
              0.47068966,  0.51551724,  0.56034483,  0.60517241,  0.65      ])),
     ('joint1',
      array([-0.5       , -0.46551724, -0.43103448, -0.39655172, -0.36206897,
             -0.32758621, -0.29310345, -0.25862069, -0.22413793, -0.18965517,
             -0.15517241, -0.12068966, -0.0862069 , -0.05172414, -0.01724138,
              0.01724138,  0.05172414,  0.0862069 ,  0.12068966,  0.15517241,
              0.18965517,  0.22413793,  0.25862069,  0.29310345,  0.32758621,
              0.36206897,  0.39655172,  0.43103448,  0.46551724,  0.5       ])),
     ('joint2',
      array([-0.8       , -0.74482759, -0.68965517, -0.63448276, -0.57931034,
             -0.52413793, -0.46896552, -0.4137931 , -0.35862069, -0.30344828,
             -0.24827586, -0.19310345, -0.13793103, -0.08275862, -0.02758621,
              0.02758621,  0.08275862,  0.13793103,  0.19310345,  0.24827586,
              0.30344828,  0.35862069,  0.4137931 ,  0.46896552,  0.52413793,
              0.57931034,  0.63448276,  0.68965517,  0.74482759,  0.8       ])),
     ('x',
      array([-0.11      , -0.10241379, -0.09482759, -0.08724138, -0.07965517,
             -0.07206897, -0.06448276, -0.05689655, -0.04931034, -0.04172414,
             -0.03413793, -0.02655172, -0.01896552, -0.01137931, -0.0037931 ,
              0.0037931 ,  0.01137931,  0.01896552,  0.02655172,  0.03413793,
              0.04172414,  0.04931034,  0.05689655,  0.06448276,  0.07206897,
              0.07965517,  0.08724138,  0.09482759,  0.10241379,  0.11      ])),
     ('y',
      array([-0.11      , -0.10241379, -0.09482759, -0.08724138, -0.07965517,
             -0.07206897, -0.06448276, -0.05689655, -0.04931034, -0.04172414,
             -0.03413793, -0.02655172, -0.01896552, -0.01137931, -0.0037931 ,
              0.0037931 ,  0.01137931,  0.01896552,  0.02655172,  0.03413793,
              0.04172414,  0.04931034,  0.05689655,  0.06448276,  0.07206897,
              0.07965517,  0.08724138,  0.09482759,  0.10241379,  0.11      ])),
     ('angle',
      array([0.        , 0.15707963, 0.31415927, 0.4712389 , 0.62831853,
             0.78539816, 0.9424778 , 1.09955743, 1.25663706, 1.41371669]))]
    """

    def __init__(self, mode="finger"):
        if mode == "finger":
            ds_path = os.path.join(
                os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "robot_finger_datasets", "finger")
            if not os.path.exists(ds_path):
                raise ValueError(
                    "Dataset '{}' not found. Make sure the dataset is publicly available and downloaded correctly."
                        .format(mode))
            self.info = np.load(os.path.join(ds_path, "finger_info.npz"), allow_pickle=True)
            self.tarfile = tarfile.open(os.path.join(ds_path, 'finger_images.tar'))
            self.labels = np.load(os.path.join(ds_path, "finger_labels.npz"), allow_pickle=True)['labels']

            sorted_files = [m for m in self.tarfile.getmembers() if m.isfile()]
            sorted_files.sort(key=lambda m: int(m.name.split('/')[-1][:-4]))
            self.files = sorted_files

        elif mode == "finger_heldout_colors":
            ds_path = os.path.join(os.environ.get("DISENTANGLEMENT_LIB_DATA", "."),
                                   "robot_finger_datasets", "finger_heldout_colors")
            if not os.path.exists(ds_path):
                raise ValueError(
                    "Dataset '{}' not found. Make sure the dataset is publicly available and downloaded correctly."
                        .format(mode))
            self.info = np.load(os.path.join(ds_path, "finger_heldout_colors_info.npz"), allow_pickle=True)
            self.tarfile = tarfile.open(os.path.join(ds_path, 'finger_heldout_colors_images.tar'))
            self.labels = np.load(os.path.join(ds_path, "finger_heldout_colors_labels.npz"), allow_pickle=True)[
                'labels']

            sorted_files = [m for m in self.tarfile.getmembers() if m.isfile()]
            sorted_files.sort(key=lambda m: int(m.name.split('/')[-1][:-4]))
            self.files = sorted_files

        elif mode == "finger_real":
            ds_path = os.path.join(os.environ.get("DISENTANGLEMENT_LIB_DATA", "."),
                                   "robot_finger_datasets", "finger_real")
            if not os.path.exists(ds_path):
                raise ValueError(
                    "Dataset '{}' not found. Make sure the dataset is publicly available and downloaded correctly."
                        .format(mode))
            self.info = np.load(os.path.join(ds_path, "finger_real_info.npz"), allow_pickle=True)
            self.images = np.load(os.path.join(ds_path, 'finger_real_images.npz'), allow_pickle=True)['images']
            self.labels = np.load(os.path.join(ds_path, "finger_real_labels.npz"), allow_pickle=True)['labels']

            self.images = np.array(self.images, dtype=np.float32) / 255
        else:
            raise ValueError("Unknown mode provided.")

        self.factor_sizes = self.info["num_factor_values"].tolist()
        self.mode = mode

    @property
    def num_factors(self):
        return len(self.factor_sizes)

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return [3, 128, 128]

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.labels[np.random.randint(0, len(self), size=(num,))]

    def sample_observations_from_factors(self, factors, _random_state):
        indices = np.nonzero(pairwise_distances(factors, self.labels) == 0)[1]
        return np.stack([self.read_image(index) for index in indices]).transpose(0,2,3,1)

    def read_image(self, item):
        if self.mode == "finger_real":
            return self.images[item].transpose(2,0,1)
        else:
            file_fp = self.tarfile.extractfile(self.files[item])
            pic = Image.open(file_fp)
            return np.asarray(pic, dtype=np.float32).transpose(2,0,1) / 255

    def __getitem__(self, item):
        factor = self.labels[item]
        return self.read_image(item), factor

    def __len__(self):
        return len(self.labels)
