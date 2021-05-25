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


def attention_score(K, Q):
    score = K.matmul(Q.T) / np.sqrt(K.size(1))
    return torch.softmax(score, dim=1)


@gin.configurable("Attention")
class Attention(BaseVAE):
    def __init__(self, input_shape, ):
        super().__init__(input_shape)
        self.Q = torch.nn.Linear(2, self.num_latent)

    def model_fn(self, features, labels, global_step):
        self.summary = {}
        bs = features.size(0)
        z_mean, z_logvar = self.encode(features)
        z = self.Q(labels.float())
        z_sampled = self.sample_from_latent_distribution(torch.zeros_like(z_mean), torch.zeros_like(z_mean))

        reconstructions = self.decode(z_mean)
        expand_recons = reconstructions.unsqueeze(0).repeat([bs, 1, 1, 1, 1])
        att_s = attention_score(z_mean, z_sampled).reshape(bs, bs, 1, 1, 1)

        weighted_recons = (att_s * expand_recons).sum(1)
        att_loss = losses.make_reconstruction_loss(features, weighted_recons).mean()

        per_sample_loss = losses.make_reconstruction_loss(features, reconstructions.data).mean()
        kl = compute_gaussian_kl(z_mean, z_logvar)
        gamma = 10 * (torch.exp(z_logvar) - z_logvar).mean()
        kl_loss = kl.sum()
        for i in range(kl.shape[0]):
            self.summary[f"kl/{i}"] = kl[i]
        self.summary['reconstruction_loss'] = per_sample_loss
        self.summary['loss'] = att_loss
        return att_loss + kl_loss, self.summary


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
        # indices = np.argsort(mean.std(0))
        tsne = manifold.TSNE()
        embedding = tsne.fit_transform(mean)
        fig, axes = plt.subplots()
        # plt.scatter(*mean[:, indices[-2:]].T)
        plt.scatter(*embedding.T)
        log['codes'] = wandb.Image(fig)
        log['latents'] = mean.tolist()

        log.update(self.evaluator.visualize_model(model))
        wandb.log(log)

        model.train()
        model.cuda()


# 2. Train beta-VAE from the configuration at model.gin.
bindings = [
               "train.model=@Attention",
               "train.batch_size=64",
               "train.dataset = 'translation'",
               "translation.pos_type=0",
               "translation.radius=5",
               "evaluate.dataset = 'translation'",
               "evaluate.random_seed=99",
               "train.training_steps = 10000"
           ] + [i[2:] for i in unknown]
gin.parse_config_files_and_bindings(["shared.gin"], bindings, skip_unknown=True)

logger = WandbLogger(project='Attention')
print(logger.experiment.url)

pl_model = TraceMITrain(eval_numbers=5, dataset="translation")

trainer = pl.Trainer(logger,
                     max_steps=pl_model.training_steps,
                     checkpoint_callback=False,
                     progress_bar_refresh_rate=0,
                     gpus=1, )

trainer.fit(pl_model)
