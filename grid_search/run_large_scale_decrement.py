#!/usr/bin/env python
import argparse
import disentanglement_lib.utils.hyperparams as h
import os 
import logging
os.environ['WANDB_NOTES']="""Decremental model"""
os.environ['WANDB_PROJECT']='Decrement'
os.environ['WANDB_ENTITY']="dlib"
os.environ['WANDB_TAGS']="scale"

from disentanglement_lib.evaluation.metrics.mig import compute_mig
from disentanglement_lib.methods.unsupervised.train import PLModel
from disentanglement_lib.methods.unsupervised import callbacks
from disentanglement_lib.data.named_data import get_named_ground_truth_data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import os
import argparse
import gin
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from disentanglement_lib.utils.results import save_gin


seeds = h.sweep("model.seed",h.categorical(list(range(0,5))))


datasets = h.sweep("configs", ["disentanglement_lib/config/data/dsprites.gin",
    #"disentanglement_lib/config/data/shapes3d.gin",
    ])

model_setting1 = h.sweep("decrement.betas",h.discrete([
    #"[1.0,10.0,20,30,40,50,60,70,80]",
    "[1.0,10.0,40.0]"]))

model_setting2 = h.sweep("decrement.scale", h.discrete([1.0]))

general_setting=[{
#    'configs':"disentanglement_lib/config/data/dsprites.gin",
    'model.num_latent':1024
}]

all_experiemts = h.product([seeds, datasets, model_setting1,model_setting2,general_setting])


training_steps = int(3e5)

program = "python exps/decrement.py"
logging.basicConfig(filename='log.txt',filemode='a',level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

for i,args in enumerate(all_experiemts[1:]):

    args = " ".join(map(lambda x:f"--{x[0]}={x[1]}",args.items()))
    cmd = f"{program} {args} --max_steps {training_steps} " 
    print(cmd)
    continue
    ret = os.system(cmd)
    logging.info(f"[{__file__}:{i}] {cmd} -> {ret}")
    if ret!=0:
        print('error! Stop at ', i )
        break
