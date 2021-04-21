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

"""Main training protocol used for unsupervised disentanglement models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.data.ground_truth import util
from disentanglement_lib.data.ground_truth.ground_truth_data import *
from disentanglement_lib.methods.shared import losses
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from disentanglement_lib.methods.unsupervised import model  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised.gaussian_encoder_model import GaussianModel
from disentanglement_lib.methods.unsupervised.model import gaussian_log_density
from disentanglement_lib.utils import results
from disentanglement_lib.evaluation.metrics import mig

import numpy as np
import logging

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import gin
import pathlib, shutil
import wandb

from disentanglement_lib.utils.hub import convert_model
from disentanglement_lib.utils.mi_estimators import estimate_entropies
from disentanglement_lib.visualize.visualize_util import plt_sample_traversal


@gin.configurable("train", blacklist=[])
class Train(pl.LightningModule):
    """Trains the estimator and exports the snapshot and the gin config.

        The use of this function requires the gin binding 'dataset.name' to be
        specified as that determines the data set used for training.

        Args:
          model: GaussianEncoderModel that should be trained and exported.
          training_steps: Integer with number of training steps.
          random_seed: Integer with random seed used for training.
          batch_size: Integer with the batch size.
          name: Optional string with name of the model (can be used to name models).
          model_num: Optional integer with model number (can be used to identify
            models).
    """

    def __init__(self,
                 model=gin.REQUIRED,
                 training_steps=gin.REQUIRED,
                 random_seed=gin.REQUIRED,
                 batch_size=gin.REQUIRED,
                 opt_name=torch.optim.Adam,
                 lr=5e-4,
                 eval_numbers=1,
                 name="",
                 model_num=None,
                 data_fun=named_data.get_named_ground_truth_data,
                 dir=None):
        super().__init__()
        self.dir = '/tmp/models/' + str(np.random.randint(99999)) if dir is None else dir
        self.training_steps = training_steps
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.lr = lr
        self.name = name
        self.model_num = model_num
        self.eval_numbers = eval_numbers
        wandb.config['dataset'] = gin.query_parameter('dataset.name')
        self.save_hyperparameters()
        self.opt_name = opt_name
        self.data = data_fun()
        img_shape = np.array(self.data.observation_shape)[[2, 0, 1]].tolist()
        # img_shape = [1,64,64]
        self.ae = model(img_shape)

    def training_step(self, batch, batch_idx):
        if self.eval_numbers > 0 and \
                (self.global_step + 1) % (self.training_steps // self.eval_numbers) == 0:
            self.evaluate()
        if self.data.supervision:
            x, y = batch
            loss, summary = self.ae.model_fn(x.float(), y)
        else:
            x, y = batch
            loss, summary = self.ae.model_fn(x.float(), None)
        self.log_dict(summary)
        return loss

    def train_dataloader(self) -> DataLoader:
        dl = DataLoader(self.data,
                        batch_size=self.batch_size,
                        num_workers=4,
                        shuffle=True,
                        pin_memory=True)
        return dl

    def configure_optimizers(self):
        optimizer = self.opt_name(self.parameters(), lr=self.lr)
        return optimizer

    def save_model(self, file):
        dir = str(self.dir)
        file_path = os.path.join(dir, file)
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.ae.state_dict(), file_path)
        wandb.save(file_path, base_path=dir)

    def estimate_decomposition(self, model, dataset_loader):
        """
        reference: https://github.com/rtqichen/beta-tcvae/blob/master/elbo_decomposition.py
        :param model:
        :param dataset_loader:
        :return: dict(): TC, MI, DWKL
        """
        N = len(dataset_loader.dataset)  # number of data samples
        K = model.num_latent  # number of latent variables
        S = 1  # number of latent variable samples
        nparams = 2

        print('Computing q(z|x) distributions.')
        # compute the marginal q(z_j|x_n) distributions
        qz_params = torch.Tensor(N, K, nparams).cuda()
        n = 0
        for samples in dataset_loader:
            if self.data.supervision:
                xs, labels = samples
            else:
                xs, _ = samples
            batch_size = xs.size(0)
            xs = xs.view(batch_size, -1, 64, 64).cuda()
            mu, logvar = model.encode(xs)
            qz_params[n:n + batch_size, :, 0] = mu.data
            qz_params[n:n + batch_size, :, 1] = logvar.data
            n += batch_size
        z_sampled = model.sample_from_latent_distribution(qz_params[..., 0], qz_params[..., 1])

        # pz = \sum_n p(z|n) p(n)
        logpz = gaussian_log_density(z_sampled, torch.zeros_like(z_sampled), torch.zeros_like(z_sampled)).mean(0)
        logqz_condx = gaussian_log_density(z_sampled, qz_params[..., 0], qz_params[..., 1]).mean(0)

        z_sampled = z_sampled.transpose(0, 1).contiguous().view(K, N * S)
        marginal_entropies, joint_entropy = estimate_entropies(z_sampled, qz_params)

        # Independence term
        # KL(q(z)||prod_j q(z_j)) = log q(z) - sum_j log q(z_j)
        dependence = (- joint_entropy + marginal_entropies.sum()).item()

        # Information term
        # KL(q(z|x)||q(z)) = log q(z|x) - log q(z) = H(z) - H(z|x)
        H_zCx = -logqz_condx.sum().item()
        H_qz = joint_entropy.item()
        information = (joint_entropy - H_zCx).item()
        z_information = (marginal_entropies + logqz_condx).cpu().numpy().round(2)

        # Dimension-wise KL term
        # sum_j KL(q(z_j)||p(z_j)) = sum_j (log q(z_j) - log p(z_j))
        dimwise_kl = (marginal_entropies - logpz).sum().item()

        # Compute sum of terms analytically
        # KL(q(z|x)||p(z)) = log q(z|x) - log p(z)
        analytical_cond_kl = (logqz_condx - logpz).sum().item()

        print('Dependence: {}'.format(dependence))
        print('Information: {}'.format(information))
        print('Dimension-wise KL: {}'.format(dimwise_kl))
        print('Analytical E_p(x)[ KL(q(z|x)||p(z)) ]: {}'.format(analytical_cond_kl))
        log = dict(TC=dependence,
                   MI=information,
                   ZMI=z_information,
                   DWKL=dimwise_kl,
                   KL=analytical_cond_kl,
                   H_q_zCx=H_zCx,
                   H_q_z=H_qz)
        return log

    def evaluate(self) -> None:
        self.save_model(f"model_{self.global_step}.pt")
        model = self.ae
        model.eval()
        dic_log = self.estimate_decomposition(model, self.train_dataloader())
        model.cpu()
        dic_log.update(self.visualize_model(model))
        dic_log.update(self.compute_mig(model))
        wandb.log(dic_log)
        model.cuda()
        model.train()

    def compute_mig(self, model, device='cpu') -> dict:
        if self.data.supervision == False:
            return {}
        _encoder, _decoder = convert_model(model, device=device)
        result = mig.compute_mig(self.data, lambda x: _encoder(x)[0], np.random.RandomState(), )
        return result

    def visualize_model(self, model) -> dict:
        _encoder, _decoder = convert_model(model)
        latent_dim = self.ae.num_latent
        mu = torch.zeros(1, latent_dim)
        fig = plt_sample_traversal(mu, _decoder, 8, range(latent_dim), 2)
        return {'traversal': wandb.Image(fig)}



def train_with_gin(model_dir,
                   overwrite=False,
                   gin_config_files=None,
                   gin_bindings=None):
    """Trains a model based on the provided gin configuration.

    This function will set the provided gin bindings, call the train() function
    and clear the gin config. Please see train() for required gin bindings.

    Args:
      model_dir: String with path to directory where model output should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      gin_config_files: List of gin config files to load.
      gin_bindings: List of gin bindings to use.
    """
    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    logging.info(gin.operative_config_str())

    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(project='dlib')

    model_path = pathlib.Path(model_dir)
    # Delete the output directory if it already exists.
    if model_path.exists():
        if overwrite:
            shutil.rmtree(model_path)
        else:
            raise FileExistsError("Directory already exists and overwrite is False.")
    model_path.mkdir(parents=True, exist_ok=True)

    gpus = torch.cuda.device_count()
    print(logger.experiment.url)
    logger.experiment.save(model.__file__, os.path.dirname(model.__file__))
    pl_model = Train(dir=model_path)
    if gpus > 0:
        trainer = pl.Trainer(logger,
                             progress_bar_refresh_rate=0,  # disable progress bar
                             max_steps=pl_model.training_steps,
                             checkpoint_callback=False,
                             gpus=1,
                             default_root_dir=model_dir)
    else:
        trainer = pl.Trainer(logger,
                             progress_bar_refresh_rate=0,
                             max_steps=pl_model.training_steps,
                             checkpoint_callback=False,
                             tpu_cores=8,
                             default_root_dir=model_dir)

    trainer.fit(pl_model)
    pl_model.save_model('model.pt')
    from disentanglement_lib.utils.results import save_gin
    save_gin(f"{model_dir}/train.gin")
    wandb.save(f"{model_dir}/train.gin", base_path=model_dir)

    gin.clear_config()
    return pl_model
