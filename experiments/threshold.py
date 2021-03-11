"""
实验目的：发现action、β、KL 之间的关系。
预期：
1. KL与β成反比，且存在一threshold使得，当β>threshold时，KL<0.1。
2. KL与action的entropy成正比。
3. KL与action的entropy有关而与action无关。（设计不同的动作但是entropy相同）
"""
import os
import time
from disentanglement_lib.data.ground_truth import dsprites
from disentanglement_lib.data.ground_truth import util
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from disentanglement_lib.methods.unsupervised import train, model  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised.gaussian_encoder_model import GaussianModel, load
from disentanglement_lib.utils import results

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gin.torch
import pathlib, shutil
import wandb

experiment = __file__.split('/')[-1][:-3]
base_directory = 'experiment_results'


def train1(model_dir, model, action,
           overwrite=True,
           training_steps=22300,
           random_seed=0,
           batch_size=64,
           opt_name=torch.optim.Adam):
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)
    model_path = pathlib.Path(model_dir)
    # Delete the output directory if it already exists.
    if model_path.exists():
        if overwrite:
            shutil.rmtree(model_path)
        else:
            print("Directory already exists and overwrite is False.")
    model_path.mkdir(parents=True, exist_ok=True)
    # Create a numpy random state. We will sample the random seeds for training
    # and evaluation from this.

    # Obtain the dataset. tf format
    dataset = dsprites.DSprites([action])
    tf_data_shape = dataset.observation_shape
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    # Set up time to keep track of elapsed time in results.
    experiment_timer = time.time()


    input_shape = [tf_data_shape[2], tf_data_shape[0], tf_data_shape[1]]
    autoencoder = model(input_shape)

    device = 'cuda'

    autoencoder.to(device).train()
    opt = opt_name(autoencoder.parameters())
    global_step = 0

    summary = {}
    while global_step < training_steps:
        for imgs, labels in dl:
            imgs, labels = imgs.to(device), labels.to(device)
            autoencoder.global_step = global_step
            summary = autoencoder.model_fn(imgs, labels)
            loss = summary['loss']
            wandb.log(summary)

            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step = global_step + 1

            # if (global_step + 1) % save_checkpoints_steps == 0:
            #     torch.save(autoencoder.state_dict(), f'{model_dir}/ckp-{global_step:06d}.pth')
            if global_step >= training_steps:
                break

    # Save model as a TFHub module.
    autoencoder.save(model_dir)
    wandb.save(f'{model_dir}/ckp.pth')

    # Save the results. The result dir will contain all the results and config
    # files that we copied along, as we progress in the pipeline. The idea is that
    # these files will be available for analysis at the end.
    results_dir = os.path.join(model_dir, "results")
    results_dict = summary
    results_dict["elapsed_time"] = time.time() - experiment_timer
    results.update_result_directory(results_dir, "train", results_dict)


if __name__ == "__main__":
    for random_seed in range(3, 6):
        for trail, beta in enumerate(np.linspace(1, 23, 10)):
            steps = int(3e4)
            output_directory = os.path.join(base_directory, experiment, str(random_seed), str(beta))
            wandb.init(project='experiments', tags=[experiment], reinit=True,
                       config={
                           'dataset': 'translation',
                           'beta': beta,
                           'random_seed': random_seed
                       })

            gin_bindings = [
                'dataset.name = "translation"',
                f"translation.img_size=(2,8,1)",
                f"translation.stride=1",
                "train.model=@vae",
                f"vae.beta={beta}",
                f"train.training_steps = {steps}"
            ]
            train.train_with_gin(os.path.join(output_directory, 'model'), True,
                                 ['shared.gin'],
                                 gin_bindings=gin_bindings)
    exit()
    for random_seed in range(3):
        for trail, beta in enumerate(np.linspace(1, 300, 10)):
            for action in range(1, 5):
                steps = int(3e4)
                output_directory = os.path.join(base_directory, experiment, str(random_seed), str(beta), str(action))
                wandb.init(project='experiments', tags=[experiment], reinit=True,
                           config={
                               'action': action,
                               'beta': beta,
                               'random_seed': random_seed
                           })
                model = model.BetaVAE
                gin_bindings = [
                    f"vae.beta={beta}",
                    f"train.training_steps = {steps}"
                ]
                gin.parse_config_files_and_bindings(['shared.gin'], gin_bindings)
                train1(os.path.join(output_directory, 'model'),
                       model,
                       action,
                       training_steps=steps, random_seed=random_seed)
                wandb.join()
                gin.clear_config()
