import os
import sys
import time, gin, wandb
from typing import *

import pytorch_lightning as pl
import numpy as np, torch, matplotlib.pyplot as plt, torch.nn as nn
import torch.nn.functional as F

from disentanglement_lib.data.ground_truth.ground_truth_data import sample_factor
from disentanglement_lib.methods.shared import architectures, losses
from disentanglement_lib.data.ground_truth import named_data, ground_truth_data
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import argparse

from disentanglement_lib.visualize.visualize_util import array_animation


def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dsprites_full')
    parser.add_argument('--visualize', default=True, type=bool)

    args = parser.parse_args()
    # pl_model = main(args)

    data = named_data.get_named_ground_truth_data(args.data)
    factor_vec = np.array([2, 3, 30, 15, 15])  # sample_factor(data)
    wandb.init(project='factor_prob', tags=['v5'], config=args)
    print(wandb.run.url)
    hit_codes = []
    for factor in range(3, 4):
        img_shape = np.array(data.observation_shape)[[2, 0, 1]].tolist()

        action = ground_truth_data.RandomAction(data, factor, factor_vec)
        x, y = zip(*[action[i] for i in range(len(action))])
        x, y = torch.FloatTensor(np.stack(x)).cuda(), torch.FloatTensor(np.stack(y)).cuda()
        real_z = ((y - y.mean(0)) / torch.clamp(y.std(0), 0.01))

        decoder = architectures.deconv_decoder(real_z.size(1), img_shape).cuda()
        # wandb.watch(decoder, log_freq=50, log='all')

        optimizer = torch.optim.Adam(decoder.parameters(), 1e-4)
        epochs = 2000
        last_loss = None
        for e in range(epochs):
            z = torch.randn_like(real_z)
            m = 1
            zs = z.repeat(2, 1)
            recons = decoder(zs)
            xs = x.repeat(1, 2, 1, 1).reshape(-1, 1, 64, 64)
            loss1 = losses.l2_loss(xs, recons, "logits")
            loss1 = loss1.reshape(-1, m)
            if last_loss is None:
                last_loss = loss1.data.mean()

            gap = (loss1[:, 0] - last_loss).abs()
            loss = (loss1.sum(1) - gap * 1).mean(0)
            if loss1.mean() < last_loss:
                hit_codes.append(z)
            last_loss = loss1.data.mean()

            wandb.log({f"loss/{factor}": loss,
                       f"gap/{factor}": gap.mean()})

            indices = loss1.argmin(1)

            # backward
            optimizer.zero_grad()
            loss.backward()
            # apply and clear grads
            optimizer.step()
            if (e + 1) % 500 == 0 and args.visualize:
                # decoder.cpu()
                zs = torch.stack(hit_codes[-100:]).cpu()
                # var_i = zs.mean(0).std(0).argmax()
                for var_i in range(real_z.size(1)):
                    z = torch.zeros(10, real_z.size(1)).cpu()
                    z[:, var_i] = torch.linspace(-2, 2, 10)
                    recons = (decoder(z.cuda()).data.squeeze(1).sigmoid()).cpu().numpy()
                    ani = array_animation(recons, 5)
                    wandb.log({f'action/{factor}_{var_i}':
                                   wandb.Html(ani.to_html5_video(), False)})
                print(zs.mean(0).std(0))
