#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import torch
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from disentanglement_lib.methods.shared import losses
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils, mig
from disentanglement_lib.methods.unsupervised.model import BaseVAE, compute_gaussian_kl
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results, mi_estimators
import gin
import argparse

from disentanglement_lib.utils.hub import convert_model

parser = argparse.ArgumentParser()

args, unknown = parser.parse_known_args()


class ProjectionSet(Dataset):
    def __init__(self, data, encoder,
                 factor_indices,
                 length=1000, random_seed=None):
        self.data = data
        self.encoder = encoder
        self.length = length
        self.factor_indices = factor_indices
        self.random_state = np.random.RandomState(random_seed)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        factors, obs = self.data.sample(1, self.random_state)
        return self.encoder(obs)[0], factors[0, self.factor_indices]


@gin.configurable("anneal")
class Annealing(BaseVAE):
    def __init__(self, input_shape, beta_h=40):
        super().__init__(input_shape)
        self.beta_h = beta_h
        self.total_steps = gin.query_parameter('train.training_steps')
        wandb.config['beta_h'] = self.beta_h

    def regularizer(self, kl, z_mean, z_logvar, z_sampled):
        beta = max((1 - self.global_step / self.total_steps) * self.beta_h, 1)
        self.summary['beta'] = beta
        return beta * (kl.sum())


class TraceMITrain(train.Train):

    def evaluate(self) -> None:
        model = self.ae

        model.eval()
        model.cpu()

        direction = wandb.run.dir
        file = os.path.join(direction, f"_{self.global_step}.pt")
        torch.save(model.state_dict(), file)
        wandb.save(file, base_path=direction)

        _encoder, _ = convert_model(model, device='cpu')

        def encoder(obs):
            mu, logvar = _encoder(obs)
            return mu

        bs = 64
        log = dict()
        for c in range(self.data.num_factors):
            factor_indices = [c]
            projected_data = ProjectionSet(self.data, encoder, factor_indices, length=4000 * bs)
            dl = DataLoader(projected_data, batch_size=bs, num_workers=4)

            # res_dict = mi_estimators.estimate_mutual_information(mi_estimators.CLUB,
            #                                                      model.num_latent, len(factor_indices),
            #                                                      dl, device='cpu')
            # mi_values = res_dict['values'][-100:]
            # log[f'MI/CLUB.{c}'] = np.mean(mi_values)
            # log[f'MI/CLUB_std.{c}'] = np.std(mi_values)

            res_dict = mi_estimators.estimate_mutual_information(mi_estimators.InfoNCE,
                                                                 model.num_latent, len(factor_indices),
                                                                 dl, device='cpu')
            mi_values = res_dict['values'][-100:]
            log[f'MI/InfoNCE.{c}'] = np.mean(mi_values)
            log[f'MI/InfoNCE_std.{c}'] = np.std(mi_values)

        if self.ae.num_latent > 1:
            log.update(self.compute_mig(model))

        wandb.log(log)
        model.train()
        model.cuda()

    def on_fit_end(self) -> None:
        model = self.ae
        model.cpu()
        model.eval()
        log = self.visualize_model(model)
        if self.ae.num_latent > 1:
            log.update(self.compute_mig(model))
        wandb.log(log)


# 2. Train beta-VAE from the configuration at model.gin.
bindings = [] + [i[2:] for i in unknown]
gin.parse_config_files_and_bindings(["shared.gin"], bindings, skip_unknown=True)

logger = WandbLogger(project='dlib', tags=['IB'])
print(logger.experiment.url)

pl_model = TraceMITrain(eval_numbers=50)
trainer = pl.Trainer(logger,
                     max_steps=pl_model.training_steps,
                     checkpoint_callback=False,
                     progress_bar_refresh_rate=0,
                     gpus=1, )

trainer.fit(pl_model)
