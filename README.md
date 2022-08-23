# disentanglement_lib-PyTorch!
![Sample visualization](https://github.com/erow/disentanglement_lib/blob/pytorch/sample.gif?raw=true)

Most codes of **disentanglement_lib-PyTorch** are migrated from google's [disentanglement_lib](https://github.com/google-research/disentanglement_lib). 

# What's new

## PyTorch Yes!
This project applies pytorch lightning to train the model in `disentanglement_lib.methods.unsupervised.train`. All disentanglement methods in `disentanglement_lib.methods.unsupervised.model` can be used independently.

## Flexible configuration
`dlib_run --configs config1 config2 --key vale` 

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
```bash
git clone https://github.com/erow/disentanglement_lib
cd disentanglement_lib
pip install .
```

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

`dlib_run --model.regularizers="[@exp_annealing()]" --exp_annealing.total_steps=500000 --exp_annealing.beta_h=50 -c disentanglement_lib/config/data/scream.gin`