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
from disentanglement_lib.methods.unsupervised import train, model
from disentanglement_lib.evaluation import evaluate
import gin

study = UnsupervisedStudyV1()

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--output', default='output', help="Output path.")
parser.add_argument('--dataset', choices=[
    "dsprites_full", "color_dsprites", "noisy_dsprites",
    "dsprites_noshape",
    "scream_dsprites", "smallnorb", "cars3d", "shapes3d"
], default="dsprites_noshape")
parser.add_argument('--overwrite', default=True, type=bool)

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    path = args.output
    overwrite = args.overwrite
    if pathlib.Path(path).exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(path)
    pathlib.Path(path).mkdir(parents=True)

    gin_bindings = [
                       f"dataset.name='{args.dataset}'",
                       "mig.num_train=10000"
                   ] + [i[2:] for i in unknown]
    _, share_conf = study.get_model_config()
    train.train_with_gin(os.path.join(path, "model"), overwrite, [share_conf],
                         gin_bindings=gin_bindings)

    # 3. Extract the mean representation for both of these models.
    representation_path = os.path.join(path, "representation")
    model_path = os.path.join(path, "model")
    # postprocess.postprocess(model_path, representation_path, overwrite)
    confs = study.get_eval_config_files()
    confs = [confs[i] for i in [4, 7]]
    print(confs)
    for metric_conf in confs:
        metric = metric_conf.split('/')[-1][:-4]
        result_path = os.path.join(path, "metrics", metric)
        evaluate.evaluate_with_gin(
            model_path, result_path, overwrite, [metric_conf, share_conf], gin_bindings)
