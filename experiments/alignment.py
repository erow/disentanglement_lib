"""
实验目的：探究样本overlap对解耦的影响。
Data 中的translation可以调整stride和图像的高度h，宽度固定为4。
当 stride>=4, h<=4时没有overlap。
实验预期：
1. 没有overlap很难解耦。
2. 加大2个action的显著度有利于解耦。（h增大）
"""
import os
import pathlib

import gin.torch
import torch
from absl import app
from absl import flags
from absl import logging


from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import model
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.data.ground_truth.translation import Translation
from disentanglement_lib.visualize import visualize_model

import numpy as np
import wandb

experiment = __file__.split('/')[-1][:-3]
base_directory = f'experiment_results/{experiment}'

for random_seed in range(3):
    for stride in [1, 2, 4]:
        for h in [2, 4, 8]:  # 2, 4
            for method in ['AnnealedTCVAE']:  # ['beta_tc_vae', 'vae']:
                wandb.init(project='experiments', tags=[experiment], reinit=True,
                           config={
                               'method': method,
                               'stride': stride,
                               'h': h,
                               'random_seed': random_seed
                           })

                output_directory = os.path.join(base_directory, method, str(h), str(stride))
                output_directory = pathlib.Path(output_directory)
                output_directory.mkdir(parents=True, exist_ok=True)

                ds_bindings = [
                    "dataset.name='translation'",
                    f"translation.img_size=({h},4,1)",
                    f"translation.stride={stride}",
                    "train.training_steps = 5000"
                ]

                train_bindings = [
                    "train.model = @{}".format(method),
                    f"train.random_seed={random_seed}"
                ]

                model_dir = os.path.join(output_directory, "model")
                model_bindings = train_bindings + ds_bindings


                train.train_with_gin(model_dir, True, ['shared.gin'],
                                     model_bindings)

                representation_dir = os.path.join(output_directory, "representation")

                postprocess.postprocess_with_gin(model_dir, representation_dir, True,
                                                 gin_bindings=["dataset.name='translation'",
                                                               "discriminator.discriminator_fn=@fc_discriminator",
                                                               "postprocess.random_seed=0"])
                wandb.save(os.path.join(representation_dir, 'representation.npy'))
                representation = np.load(os.path.join(representation_dir, 'representation.npy'), allow_pickle=True)
                representation = representation[()]
                z = torch.Tensor(representation['mean'])
                fig = visualize_model.plot_latent_vs_ground(z, latnt_sizes=[16, 16])
                wandb.log({'projection': wandb.Image(fig)})
                fig.savefig(os.path.join(output_directory, 'projection.png'))

                # visualize projection

                fig = visualize_model.plot_projection_2d(z, [16, 16], 10)
                wandb.log({'projection_d': wandb.Image(fig)})
