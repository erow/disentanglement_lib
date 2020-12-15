import os
import pathlib

import gin.torch
import torch
from absl import app
from absl import flags
from absl import logging

from disentanglement_lib.config.unsupervised_study_v1.sweep import UnsupervisedStudyV1
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.data.ground_truth.translation import Translation
from disentanglement_lib.visualize import visualize_model

import numpy as np
import wandb

experiment = __file__.split('/')[-1][:-3]
base_directory = 'experiment_results'

for stride in [1, 2, 4]:
    for h in [2, 4, 8]:  # 2, 4
        for method in ['factor_vae']:  # 'annealed_vae', 'factor_vae'

            output_directory = os.path.join(base_directory, method, str(h), str(stride))
            output_directory = pathlib.Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)

            ds_bindings = [
                "dataset.name='translation'",
                f"translation.img_size=({h},4,1)",
                f"translation.stride={stride}"
            ]

            train_bindings = [
                "train.model = @{}".format(method)
            ]

            model_bingdings = [
                "factor_vae.gamma=30.",
                "discriminator.discriminator_fn=@fc_discriminator",
                "vae.beta=1",
                "annealed_vae.iteration_threshold=5000",
                "annealed_vae.c_max=5.",
                "annealed_vae.gamma=1000"
            ]

            model_dir = os.path.join(output_directory, "model")
            model_bindings = model_bingdings + \
                             train_bindings + ds_bindings

            wandb.init(project='experiments', tags=[experiment], reinit=True,
                       config={
                           'method': method,
                           'stride': stride,
                           'h': h
                       })
            if not os.path.exists(model_dir):

                train.train_with_gin(model_dir, False, ['shared.gin'],
                                     model_bindings)

            representation_dir = os.path.join(output_directory, "representation")

            if not os.path.exists(representation_dir):
                postprocess.postprocess_with_gin(model_dir, representation_dir, False,
                                                 gin_bindings=["dataset.name='translation'",
                                                               "discriminator.discriminator_fn=@fc_discriminator",
                                                               "postprocess.random_seed=0"])

            representation = np.load(os.path.join(representation_dir, 'representation.npy'), allow_pickle=True)
            representation = representation[()]
            z = torch.Tensor(representation['mean'])
            fig = visualize_model.plot_latent_vs_ground(z, latnt_sizes=[16, 16])
            wandb.log({'projection': wandb.Image(fig)})
            fig.savefig(os.path.join(output_directory, 'projection.png'))

            # evaluate
            eval_configs = sorted(UnsupervisedStudyV1().get_eval_config_files())
            eval_configs = eval_configs[1:]  # TODO remove DCI for taking too much time.

            for post_name in ['mean', 'sample']:
                post_dir = os.path.join(output_directory, "representation")
                # Now, we compute all the specified scores.
                for gin_eval_config in eval_configs:
                    metric_name = os.path.basename(gin_eval_config).replace(".gin", "")
                    logging.info("Computing metric '%s' on '%s'...", metric_name, post_name)
                    metric_dir = os.path.join(output_directory, "metrics", post_name,
                                              metric_name)
                    eval_bindings = [
                        "evaluation.random_seed = {}".format(0),
                        "evaluation.name = '{}'".format(metric_name),
                        f"evaluation.representation_fn =@{post_name}_representation"

                    ]
                    evaluate.evaluate_with_gin(post_dir, metric_dir, True,
                                               [gin_eval_config], eval_bindings)

            # visualize_model.visualize(model_dir, os.path.join(output_directory, 'visualization')
            #                           , True, num_points_irs=10000)
