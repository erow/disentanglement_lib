#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import cv2
import torch
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from disentanglement_lib.config.unsupervised_study_v1.sweep import UnsupervisedStudyV1
from disentanglement_lib.methods.shared import losses
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils, mig
from disentanglement_lib.methods.unsupervised.model import BaseVAE, compute_gaussian_kl
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results, mi_estimators
from disentanglement_lib.data.ground_truth import ground_truth_data
import gin
import argparse

from disentanglement_lib.utils.hub import convert_model

study = UnsupervisedStudyV1()
parser = argparse.ArgumentParser()
parser.add_argument('--metrics', nargs='+', default=[])
args, unknown = parser.parse_known_args()


class Correlation(ground_truth_data.GroundTruthData):
    @property
    def num_factors(self):
        return 3

    @property
    def factors_num_values(self):
        return [32, 32, 32]

    @property
    def observation_shape(self):
        return [64, 64, 1]

    def __len__(self):
        return 32 * 32

    def sample_factors(self, num, random_state):
        pos = np.random.randint([32, 32], size=(num, 2))
        p = pos - np.array([[16, 16]])
        theta = np.arctan2(p[:, 1], p[:, 0]) / np.pi * 180 + 180
        t = np.digitize(theta, self.angles).reshape(-1, 1)
        return np.concatenate([t, pos], 1)

    def latent_factor(self, index):
        """Get a latent factor from index."""

        x = index // 32
        y = index % 32
        theta = np.arctan2(y - 16, x - 16) / np.pi * 180 + 180
        t = np.digitize(theta, self.angles)
        return np.array([(t, x, y)], dtype=np.int64)

    def sample_observations_from_factors(self, factors, random_state):
        pos = factors[:, 1:]
        theta = self.angles[factors[:, 0] - 1] - 90
        imgs = np.zeros((len(factors), 64, 64, 1))
        for i in range(len(factors)):
            matRotate = cv2.getRotationMatrix2D((5, 5), theta[i], 1)
            dst = cv2.warpAffine(self.img.copy(), matRotate, (11, 11))
            imgs[i,
            10 + pos[i, 0]:21 + pos[i, 0],
            10 + pos[i, 1]:21 + pos[i, 1],
            0] = dst
        return imgs

    def __init__(self, random_seed=None):
        self.random_state = np.random.RandomState(random_seed)
        img = np.zeros((11, 11))
        self.img = cv2.fillPoly(img,
                                np.array([[[0, 2], [10, 5], [0, 8]]], dtype=np.int),
                                1)
        self.angles = np.linspace(0, 360, 33)


# 2. Train beta-VAE from the configuration at model.gin.
gin_bindings = ['train.training_steps=10000', 'model.num_latent=5'] + [i[2:] for i in unknown]
_, share_conf = study.get_model_config()
gin.parse_config_files_and_bindings([share_conf], gin_bindings, skip_unknown=True)

logger = WandbLogger(project='dlib', tags=['correlation'])
print(logger.experiment.url)

pl_model = train.Train(data_fun=Correlation)

trainer = pl.Trainer(logger,
                     max_steps=pl_model.training_steps,
                     checkpoint_callback=False,
                     progress_bar_refresh_rate=0,
                     gpus=1, )

trainer.fit(pl_model)
pl_model.save_model("model.pt")

from disentanglement_lib.utils.results import save_gin

# metrics
confs = study.get_eval_config_files()
print(confs)
model_path = pl_model.dir
save_gin(f"{model_path}/train.gin")

print(model_path)

for metric_conf in confs:
    gin.clear_config()
    metric = metric_conf.split('/')[-1][:-4]
    if metric in args.metrics:
        result_path = os.path.join(model_path, "metrics", metric)
        evaluate.evaluate_with_gin(
            model_path, result_path, True, [metric_conf, share_conf], gin_bindings)
