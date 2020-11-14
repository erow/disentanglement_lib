#!/usr/bin/env python
# coding=utf-8
import os
import wandb
import argparse

hyperparameter_defaults = dict(
    random_seed=0,
    param=0,
    model=0,
    architecture=0,
    dataset=0
)
parser = argparse.ArgumentParser()
for key, value in hyperparameter_defaults.items():
    parser.add_argument(f'--{key}', default=value, type=type(value))
args = parser.parse_args()

wandb.init(config=args, resume=True)
config = wandb.config
num_pair = dict(
    random_seed=1,
    param=50,
    model=50 * 6,
    architecture=50 * 36,
    dataset=50 * 36
)
model_num = sum([config[k] * num for k, num in num_pair.items()])
print(config)
os.system(f'dlib_reproduce --model_num={model_num} --only_train')
