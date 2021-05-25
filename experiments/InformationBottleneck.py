#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import sklearn.manifold as manifold
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from disentanglement_lib.data.ground_truth.translation import Translation
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
from disentanglement_lib.visualize.visualize_model import visualize_intervention

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


@gin.configurable("IB")
class IB(BaseVAE):
    def __init__(self, input_shape, C_max=6, gamma=300):
        super().__init__(input_shape)
        self.gamma = gamma
        self.C_max = C_max
        self.delta = C_max / 1e4
        self.C = 0

    def regularizer(self, kl, z_mean, z_logvar, z_sampled):
        KL = kl.sum()
        # self.C = min(self.C_max, self.C+self.delta)
        self.c = self.C_max
        return KL + self.gamma * torch.max(KL - self.C, torch.zeros(1, device=KL.device))


class TraceMITrain(train.Train):

    def evaluate(self, model) -> None:
        model.eval()
        log = {}
        log.update(self.evaluator.estimate_decomposition(model, self.evaluator.dl))
        model.cpu()
        _encoder, _ = convert_model(model, device='cpu')

        log.update(self.evaluator.compute_mig(model))

        visualize_intervention(self.dir, self.data, _encoder, min(len(self.data), 10000))
        fig = plt.gcf()
        log['irs'] = wandb.Image(fig)

        mean = []
        for obs, _ in self.evaluator.dl:
            mean.append(model.encode(obs)[0].data)
        mean = torch.cat(mean, 0).numpy()
        indices = np.argsort(mean.std(0))
        # tsne = manifold.TSNE()
        # embedding = tsne.fit_transform(mean)
        fig, axes = plt.subplots()
        plt.scatter(*mean[:, indices[-2:]].T)
        log['codes'] = wandb.Image(fig)
        log['latents'] = mean.tolist()

        log.update(self.evaluator.visualize_model(model))
        wandb.log(log)

        model.train()
        model.cuda()


# 2. Train beta-VAE from the configuration at model.gin.
bindings = [
               "train.model=@IB",
               "IB.C_max=0.68",
               "train.dataset = 'translation'",
               "translation.pos_type=2",
               "translation.radius=5",
               "evaluate.dataset = 'translation'",
               "evaluate.random_seed=99",
               "train.training_steps = 10000"
           ] + [i[2:] for i in unknown]
gin.parse_config_files_and_bindings(["shared.gin"], bindings, skip_unknown=True)

logger = WandbLogger(project='IB', tags=['IB'])
print(logger.experiment.url)

pl_model = TraceMITrain(eval_numbers=10, dataset="translation")

trainer = pl.Trainer(logger,
                     max_steps=pl_model.training_steps,
                     checkpoint_callback=False,
                     progress_bar_refresh_rate=0,
                     gpus=1, )

trainer.fit(pl_model)
