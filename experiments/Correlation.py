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

if __name__ == '__main__':

    study = UnsupervisedStudyV1()
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', nargs='+', default=[])
    args, unknown = parser.parse_known_args()

    # 2. Train beta-VAE from the configuration at model.gin.
    gin_bindings = [i[2:] for i in unknown]
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
