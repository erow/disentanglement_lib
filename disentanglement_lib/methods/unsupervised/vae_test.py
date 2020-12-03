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

"""Tests for vae.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized, absltest
from disentanglement_lib.methods.unsupervised import vae
import numpy as np
import torch


def _make_symmetric_psd(matrix):
    return 0.5 * (matrix + matrix.T) + np.diag(np.ones(10)) * 10.


class VaeTest(parameterized.TestCase):

    @parameterized.parameters((np.zeros([10, 10]), np.zeros([10, 10]), 0., 0.01),
                              (np.ones([10, 10]), np.zeros([10, 10]), 5., 5.01),
                              (np.ones([10, 10]), np.ones([10, 10]), 8.58, 8.6))
    def test_compute_gaussian_kl(self, mean, logvar, target_low, target_high):
        mean_t = torch.FloatTensor(mean)
        logvar_t = torch.FloatTensor(logvar)
        test_value = vae.compute_gaussian_kl(mean_t, logvar_t)
        self.assertBetween(test_value, target_low, target_high)

    @parameterized.parameters((0, 0., 0.01), (10, 10., 10.01),
                              (100, 100., 100.01), (101, 100., 100.01))
    def test_anneal(self, step, target_low, target_high):
        c_max = 100.
        iteration_threshold = 100
        test_value = (vae.anneal(c_max, step, iteration_threshold))
        self.assertBetween(test_value, target_low, target_high)

    @parameterized.parameters(
        (True, 0., 1.), (True, 0., 4.), (True, 1., 1.),
        (False, np.zeros(10), np.ones([10, 10])),
        (False, np.zeros(10), _make_symmetric_psd(np.random.random((10, 10)))))
    def test_compute_covariance_z_mean(self, diagonal, mean, cov):
        if diagonal:
            samples = torch.FloatTensor(np.random.normal(mean, np.sqrt(cov), size=(100000, 10)))
            cov = np.diag(np.ones([10])) * cov
        else:
            samples = torch.FloatTensor(
                np.random.multivariate_normal(mean, cov, size=(1000000)))

        test_value = (vae.compute_covariance_z_mean(samples)).numpy()
        self.assertBetween(np.sum((test_value - cov) ** 2), 0., 0.1)

    @parameterized.parameters(
        (np.ones([10, 10]), 90., 90.1), (np.zeros([10, 10]), 10., 10.1),
        (np.diag(np.ones(10)), 0., 0.1), (2. * np.diag(np.ones(10)), 10., 10.1))
    def test_regularize_diag_off_diag_dip(self, matrix, target_low, target_high):
        matrix_tf = torch.FloatTensor(matrix)
        test_value = (vae.regularize_diag_off_diag_dip(matrix_tf, 1, 1)).numpy()
        self.assertBetween(test_value, target_low, target_high)

    @parameterized.parameters((0., -1.4190, -1.4188), (1., -0.92, -0.91))
    def test_gaussian_log_density(self, z_mean, target_low, target_high):
        matrix = torch.ones(1)
        test_value = (vae.gaussian_log_density(matrix, z_mean, torch.FloatTensor([0.])))[0].numpy()
        self.assertBetween(test_value, target_low, target_high)

    @parameterized.parameters(
        (1, 0., 0.1), (10, -82.9, -82.89))  # -82.893 = (10 - 1) * ln(10000)
    def test_total_correlation(self, num_dim, target_low, target_high):
        # Since there is no dataset, the constant should be (num_latent - 1)*log(N)
        z = torch.randn(10000, num_dim)
        z_mean = torch.zeros(10000, num_dim)
        z_logvar = torch.zeros(10000, num_dim)
        test_value = (vae.total_correlation(z, z_mean, z_logvar)).numpy()
        self.assertBetween(test_value, target_low, target_high)


if __name__ == "__main__":
    absltest.main()
