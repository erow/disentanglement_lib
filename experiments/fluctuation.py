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

from disentanglement_lib.evaluation.metrics.metrics.sampling import estimate_JEMMIG_cupy
from disentanglement_lib.methods.shared import losses
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils, mig, factor_vae
from disentanglement_lib.methods.unsupervised.model import BaseVAE, compute_gaussian_kl
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results, mi_estimators
import gin
import argparse

from disentanglement_lib.utils.hub import convert_model

parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()


class TraceMITrain(train.Train):
    def evaluate(self) -> None:
        model = self.ae
        state = np.random.RandomState(self.random_seed)
        model.eval()
        model.cpu()

        direction = '/tmp/'
        file = os.path.join(direction, f"_{self.global_step}.pt")
        torch.save(model.state_dict(), file)
        wandb.save(file, base_path=direction)

        _encoder, _ = convert_model(model, device='cpu')

        def encoder(obs):
            mu, logvar = _encoder(obs)
            return mu

        bs = 64
        log = dict()

        log.update(mig.compute_mig(self.data, encoder, state))
        log.update(factor_vae.compute_factor_vae(self.data, encoder, state))

        # num_samples = 10000
        # factors, obs = self.data.sample(num_samples, state)
        # mu, logvar = _encoder(obs)
        # log.update(estimate_JEMMIG_cupy(mu, np.exp(logvar / 2), factors, num_samples=num_samples))

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

logger = WandbLogger(project='dlib', tags=['fluctuation'])
print(logger.experiment.url)

pl_model = TraceMITrain(eval_numbers=10)
trainer = pl.Trainer(logger,
                     max_steps=pl_model.training_steps,
                     checkpoint_callback=False,
                     progress_bar_refresh_rate=0,
                     gpus=1, )

trainer.fit(pl_model)
