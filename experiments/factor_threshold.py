"""
实验目的：发现action、β、KL 之间的关系。
预期：
1. KL与β成反比，且存在一threshold使得，当β>threshold时，KL<0.1。
2. 这个阈值是否稳定。
"""
import os
import time
from disentanglement_lib.data.ground_truth import dsprites, ground_truth_data, norb, mpi3d, cars3d
from disentanglement_lib.data.ground_truth import util
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from disentanglement_lib.methods.unsupervised import train, vae  # pylint: disable=unused-import
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


def train1(model_dir, model, dataset,
           overwrite=True,
           training_steps=10000,
           random_seed=0,
           batch_size=1,
           opt_name=torch.optim.Adam):
    # torch.random.manual_seed(random_seed)
    # np.random.seed(random_seed)
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

    dl = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)


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

    obs = torch.Tensor(np.stack([dataset[i][0] for i in range(len(dataset))],0)).to(device)
    projection,_ = autoencoder.encode(obs)
    _,index = projection.cpu().std(0).sort()

    data = [[i, projection[i,index[-1]],projection[i,index[-2]]] for i in range(len(dataset))]
    table = wandb.Table(data=data, columns=["c", index[-1],index[-2]])
    wandb.log({f'projection': wandb.plot.line(table, "c", index[-1])})
    return summary


def compute_threthold(output_directory, action):
    ds_name = str(action.data)
    steps = int(1e4)
    wandb.init(project='experiments', tags=[experiment], reinit=True,
               config={
                   'action': action.action_index,
                   'beta': beta,
                   'factor': action.factor,
                   "ds": ds_name
               })
    model = vae.BetaVAE
    gin_bindings = [
        f"vae.beta={beta}",
        f"train.training_steps = {steps}"
    ]
    gin.parse_config_files_and_bindings(['shared.gin'], gin_bindings)
    summary = train1(os.path.join(output_directory, 'model'),
                     model,
                     action,
                     training_steps=steps, random_seed=0)
    wandb.join()
    gin.clear_config()
    return summary


if __name__ == "__main__":
    # dsprites.DSprites,norb.SmallNORB
    for ds in [ dsprites.DSprites,norb.SmallNORB, mpi3d.MPI3D]:
        dataset = ds()
        data_shape = dataset.observation_shape
        input_shape = [data_shape[2], data_shape[0], data_shape[1]]
        for s in range(4):  # 每个action 多次采样
            for a in range(dataset.num_factors):
                action = ground_truth_data.RandomAction(dataset, a)
                if len(action) <= 1:
                    continue

                count = 0 #early stopping
                for trail, beta in enumerate(np.linspace(1, 200, 10)):
                    output_directory = os.path.join(base_directory, experiment)
                    summary = compute_threthold(output_directory, action)

                    # 早停
                    if summary['kl_loss'] < 0.1:
                        count += 1
                    else:
                        count = 0
                    if count > 2:
                        break
