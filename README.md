# disentanglement_lib-PyTorch!
![Sample visualization](https://github.com/google-research/disentanglement_lib/blob/master/sample.gif?raw=true)

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

# dlib_download_data
