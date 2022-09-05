# disentanglement_lib-PyTorch Yes!
![Sample visualization](https://github.com/erow/disentanglement_lib/blob/pytorch/sample.gif?raw=true)

Most codes of **disentanglement_lib-PyTorch** are migrated from google's [disentanglement_lib](https://github.com/google-research/disentanglement_lib). 

This project applies pytorch lightning to train the model in `disentanglement_lib.methods.unsupervised.train`. All disentanglement methods in `disentanglement_lib.methods.unsupervised.model` can be used independently.
# What's new

## Flexible configuration
`dlib_run --configs config1 config2 --key value` 

Run all experiments by one program! 

For example, `dlib_run --configs disentanglement_lib/config/data/imagenet100.gin model.gin --max_steps=200000 --model.num_latent=100`

## Experiment tracking and visulization
Wandb is an experiment tracking tool for machine learning. It's cool to select, tag, and manage experiments in one platform. I prefer to use wandb, but you can use any tools by passing the argument of trainer.

It's convenient to use `disentanglement_lib.methods.unsupervised.callbacks` to track the intermediateness status of the model. Supported callbacks:

1. Decomposition: 
2. Traversal
3. ShowSamples
4. ComputeMetric


## New models
1. ControlVAE---Rethinking Controllable Variational Autoencoders, CVPR.
2. DistillVAE---Deft: distilling entangled factors by preventing information diffusion

## New metrics

## New datasets

# Install

``pip install git+https://github.com/erow/disentanglement_lib``

Copy config files into current direction.
``dlib_config``

# Dataset

## Dowload

``dlib_download_data``

## Visualization

`` dlib_visualize_dataset --name=shapes3d --path=outputs/shapes3d``

## Load dataset

The avaliable datasets in `named_data` are:
- "dsprites_full"
- "dsprites_noshape"
- "dsprites_tiny"
- "dsprites_test"
- "color_dsprites"
- "noisy_dsprites"
- "scream_dsprites"
- "smallnorb"
- "cars3d"
- "mpi3d_toy"
- "mpi3d_realistic"
- "mpi3d_real"
- "shapes3d"
- "dummy_data"
- "translation"
- "chairs"

Call `get_named_ground_truth_data(name)` or set `dataset.name="name"` in **config.gin**.

# RUN
The hyperparameters are passed by [gin][!https://github.com/google/gin-config] globally.
You can write these paramters on a file (*.gin) or on arguments.
There are several predefined configurations in 'disentanglement_lib/config/reproduce'.

You can test your model or finetune hyperparameters by passing arguments:
`dlib_run --model.regularizers="[@exp_annealing()]" --exp_annealing.total_steps=500000 --exp_annealing.beta_h=50 -c disentanglement_lib/config/data/scream.gin`

# Result
By default, all results will be in outputs.

# Reproduce

To reproduce models, you need to setup the environment.
```bash
pip install git+https://github.com/erow/disentanglement_lib.git
dlib_download_data
dlib_config
```

Then run the models.
```python

import argparse
import disentanglement_lib.utils.hyperparams as h
import os 
import logging
logging.basicConfig(filename='log.txt',filemode='a',level=logging.INFO,
)
logging.getLogger().addHandler(logging.StreamHandler())

training_steps = int(5e5)
program = "dlib_run"

seeds = h.sweep("model.seed",h.categorical(list(range(10))))

datasets = h.sweep("configs", [ "config/data/dsprites.gin"])

model_name = h.fixed("model.regularizers", "'[@vae()]'")
betas = h.sweep("vae.beta", h.discrete([1., 6. ]))
config_beta_vae = h.zipit([model_name, betas])

model_name = h.fixed("model.regularizers", "'[@beta_tc_vae()]'")
betas = h.sweep("beta_tc_vae.beta", h.discrete([ 12.]))
config_beta_tc_vae = h.zipit([model_name, betas])

all_models = h.chainit([config_beta_tc_vae,config_beta_vae])

all_experiemts = h.product([seeds, datasets, all_models])


for i,args in enumerate(all_experiemts):
    
    args = " ".join(map(lambda x:f"--{x[0]}={x[1]}",args.items()))
    cmd = f"{program} {args} --max_steps {training_steps}"
    # print("Run: ", cmd)
    ret = os.system(cmd)
    logging.info(f"[reproduce:{i}] {cmd} -> {ret}")
    if ret!=0:
        print('error! Stop at ', i)
        break
```