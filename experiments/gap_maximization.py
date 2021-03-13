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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dsprites_full')
    parser.add_argument('-p', default=0.0, type=float, help='real codes')
    parser.add_argument('--visualize', default=True, type=bool)

    args = parser.parse_args()
    # pl_model = main(args)

    data = dsprites.DSprites([5, 4])
    bs = 128
    epochs = 300
    alpha = 0.5

    dl = DataLoader(data, batch_size=bs, pin_memory=True, shuffle=True)
    wandb.init(project='gap_maximization', tags=['v1'], config=args)
    print(wandb.run.url)

    img_shape = np.array(data.observation_shape)[[2, 0, 1]].tolist()
    decoder = architectures.deconv_decoder(5, img_shape).cuda()
    # wandb.watch(decoder, log_freq=50, log='all')

    optimizer = torch.optim.Adam(decoder.parameters(), 1e-4)
    hit_codes = []
    for e in range(epochs):
        for x, y in dl:
            x = x.cuda()
            m = 2
            zs = torch.cat([torch.randn(len(x), 5).cuda() for _ in range(m)])
            recons = decoder(zs)
            xs = x.repeat(1, m, 1, 1).reshape(-1, 1, 64, 64)
            loss1 = losses.l2_loss(xs, recons, "logits")
            loss1 = loss1.reshape(m, -1)

            gap = (loss1[1] - loss1[0]).abs()
            loss = (loss1.sum(0) - gap * alpha).mean(0)

            wandb.log({f"loss": loss,
                       f"gap": gap.mean()})

            indices = loss1.argmin(1)

            # backward
            optimizer.zero_grad()
            loss.backward()
            # apply and clear grads
            optimizer.step()

        if (e + 1) % 50 == 0:
            for var_i in range(5):
                z = torch.zeros(10, 5).cpu()
                z[:, var_i] = torch.linspace(-2, 2, 10)
                recons = (decoder(z.cuda()).data.squeeze(1).sigmoid()).cpu().numpy()
                ani = array_animation(recons, 5)
                wandb.log({f'action/{var_i}':
                               wandb.Html(ani.to_html5_video(), False)})
