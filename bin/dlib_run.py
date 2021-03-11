#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import pathlib
import shutil

import torch

from disentanglement_lib.config.unsupervised_study_v1.sweep import UnsupervisedStudyV1
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import gin

study = UnsupervisedStudyV1()

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--output', default='output', help="Output path.")
parser.add_argument('--model', type=str, choices=['beta_vae', 'annealed_vae', 'beta_tc_vae'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--s', default=1, type=float, help="Disentanglement pressure strength.")
parser.add_argument('--a', default=0, type=float, help="Reconstruction regularization.")
parser.add_argument('--dataset', choices=[
    "dsprites_full", "color_dsprites", "noisy_dsprites",
    "scream_dsprites", "smallnorb", "cars3d", "shapes3d"
], default="dsprites_full")
parser.add_argument('--overwrite', default=True, type=bool)


def get_model_bindings(model, s):
    if model == "beta_vae":
        return ["train.model=@vae",
                f"vae.beta={s * 16}"]
    elif model == "annealed_vae":
        return ["train.model=@" + model,
                f"annealed_vae.c_max={100 * s}",
                f"annealed_vae.gamma=1000",
                f"annealed_vae.iteration_threshold=30000"]
    elif model == "beta_tc_vae":
        return ["train.model=@" + model,
                f"beta_tc_vae.beta={s * 16}"]
    else:
        raise NotImplementedError(model)


if __name__ == '__main__':
    args = parser.parse_args()
    path = args.output
    overwrite = args.overwrite

    if pathlib.Path(path).exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(path)
    pathlib.Path(path).mkdir(parents=True)

    gin_bindings = get_model_bindings(args.model, args.s) + [
        # "train.training_steps=1500",
        f"model.alpha={args.a}",
        f"train.random_seed={args.seed}"
    ]
    _, share_conf = study.get_model_config()

    train.train_with_gin(os.path.join(path, "model"), overwrite, [share_conf],
                         gin_bindings=gin_bindings)
    gin.parse_config_file(share_conf)
    gin.parse_config(gin_bindings)

    # 3. Extract the mean representation for both of these models.
    representation_path = os.path.join(path, "representation")
    model_path = os.path.join(path, "model")
    postprocess.postprocess(model_path, representation_path, overwrite)
    confs = study.get_eval_config_files()
    for metric_conf in [confs[5], confs[8]]:
        metric = metric_conf.split('/')[-1][:-4]
        result_path = os.path.join(path, "metrics", metric)
        representation_path = os.path.join(path, "representation")
        evaluate.evaluate_with_gin(
            representation_path, result_path, overwrite, [metric_conf])
