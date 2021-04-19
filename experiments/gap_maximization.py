import os
import sys
import time, gin, wandb
from typing import *

import pytorch_lightning as pl
import numpy as np, torch, matplotlib.pyplot as plt, torch.nn as nn
import torch.nn.functional as F

from disentanglement_lib.data.ground_truth.ground_truth_data import sample_factor
from disentanglement_lib.methods.shared import architectures, losses
from disentanglement_lib.data.ground_truth import named_data, ground_truth_data, dsprites
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import argparse

from disentanglement_lib.visualize.visualize_util import array_animation


def main(args):
    pass


def get_recons_loss(z):
    codes = (z.reshape(-1, 1, num_latent)
             .repeat(1, m, 1)
             .reshape(-1, num_latent))

    recons = decoder(codes)

    return losses.bernoulli_loss(x.repeat(len(z), 1, 1, 1),
                                 recons, "logits").reshape(-1, m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dsprites_full')
    parser.add_argument('--visualize', default=True, type=bool)

    args = parser.parse_args()
    # pl_model = main(args)

    data = dsprites.DSprites([1])
    bs = 128
    epochs = 2000
    alpha = 0.9

    dl = DataLoader(data, batch_size=bs, pin_memory=True, shuffle=False)
    wandb.init(project='gap_maximization', tags=['v3'], config=args)
    print(wandb.run.url)

    num_latent = 2
    img_shape = np.array(data.observation_shape)[[2, 0, 1]].tolist()
    decoder = architectures.deconv_decoder(num_latent, img_shape).cuda()

    optimizer = torch.optim.Adam(decoder.parameters(), 1e-4)
    hit_codes = []

    m = 2
    buffer_codes = torch.randn(len(data), m, num_latent, device='cuda')
    for x, y in dl:
        break
    x = x.cuda()
    for e in range(epochs):

        recons1 = decoder(torch.randn(len(x), num_latent, device='cuda'))
        recons2 = decoder(torch.randn(len(x), num_latent, device='cuda'))

        recons_loss1 = losses.bernoulli_loss(x, recons1, "logits")
        recons_loss2 = losses.bernoulli_loss(x, recons2, "logits")

        gap = (recons_loss1 - recons_loss2).abs()
        loss = (recons_loss1 + recons_loss2 - alpha * gap).mean()

        log = {}
        log['loss'] = loss.item()
        log['gap'] = gap.mean()
        wandb.log(log)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e + 1) % (epochs // 10) == 0:
            for var_i in range(num_latent):
                z = torch.zeros(10, num_latent).cpu()
                z[:, var_i] = torch.linspace(-2, 2, 10)
                recons = (decoder(z.cuda()).data.squeeze(1).sigmoid()).cpu().numpy()
                ani = array_animation(recons, 5)
                wandb.log({f'action/{var_i}':
                               wandb.Html(ani.to_html5_video(), False)})

        # zs = torch.stack(hit_codes[-300:]).cpu()
    # z = torch.randn(100, num_latent).cuda()
    # recons_loss = get_recons_loss(z).cpu().data
    # s_recons_loss, s_order = recons_loss.sort(1)
    # fig = plt.figure()
    # z = z.cpu()
    # plt.scatter(z[:, 0], z[:, 1], c=s_order[:, 0])
    # wandb.log({'action/c': wandb.Image(fig)})
