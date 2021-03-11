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


class CNN(nn.Module):
    """
    Simple MLP network
    """

    def __init__(self, input_shape: Tuple[int], n_actions: int):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4),
            nn.ReLU(), nn.MaxPool2d(4, 4),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.head = nn.Sequential(nn.Linear(conv_out_size, n_actions))

    def _get_conv_out(self, shape) -> int:
        """
        Calculates the output size of the last conv layer

        Args:
            shape: input dimensions
        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))

    def forward(self, input_x):
        """
        Forward pass through network

        Args:
            x: input to network
        Returns:
            output of network
        """
        conv_out = self.conv(input_x).view(input_x.size()[0], -1)
        return self.head(conv_out)


def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dsprites_full')
    parser.add_argument('-p', default=0.0, type=float, help='real codes')
    parser.add_argument('--visualize', default=True, type=bool)

    args = parser.parse_args()
    # pl_model = main(args)

    data = named_data.get_named_ground_truth_data(args.data)
    factor_vec = np.array([2, 3, 30, 15, 15])  # sample_factor(data)
    wandb.init(project='factor_prob', tags=['v5'], config=args)
    print(wandb.run.url)
    hit_codes = []
    for factor in range(3, data.num_factors):
        img_shape = np.array(data.observation_shape)[[2, 0, 1]].tolist()

        action = ground_truth_data.RandomAction(data, factor, factor_vec)
        x, y = zip(*[action[i] for i in range(len(action))])
        x, y = torch.FloatTensor(np.stack(x)).cuda(), torch.FloatTensor(np.stack(y)).cuda()
        real_z = ((y - y.mean(0)) / torch.clamp(y.std(0), 0.01))[:, factor:factor + 1]

        decoder = architectures.deconv_decoder(1, img_shape).cuda()
        # wandb.watch(decoder, log_freq=50, log='all')

        optimizer = torch.optim.Adam(decoder.parameters(), 1e-4)
        epochs = 2000
        last_loss = 1e3
        # for e in range(epochs):
        e = 0
        while True:
            if e > epochs:
                break
            m = 100
            true_z = False
            if np.random.rand() < args.p:  # (e % m) < int(args.p * m):
                z = real_z
                true_z = True
            else:
                z = torch.randn_like(real_z)
            recon = decoder(z)
            loss1 = losses.bernoulli_loss(x, recon, "logits").mean()

            loss = loss1 * 1.1 - (loss1 - last_loss).abs()

            wandb.log({f"loss/{factor}": loss})
            if loss1.item() < last_loss:
                hit_codes.append(z)
                e += 1
            last_loss = loss1.item()
            # if true_z:
            #     wandb.log({f"loss_t/{factor}": loss1})
            # else:
            #     wandb.log({f"loss_f/{factor}": loss1})

            # backward
            optimizer.zero_grad()
            loss.backward()
            # apply and clear grads
            optimizer.step()
            if (e + 1) % 100 == 0:
                # decoder.cpu()
                z = torch.linspace(-2, 2, 10).unsqueeze(1)
                recons = (decoder(z.cuda()).data.squeeze(1).sigmoid()).cpu().numpy()
                ani = array_animation(recons, 5)
                wandb.log({f'action/{factor}':
                               wandb.Html(ani.to_html5_video(), False)})

        # classifier = CNN(img_shape, len(action)).cuda()
        # optimizer = torch.optim.Adam(classifier.parameters(), 1e-4)
        #
        # labels = torch.arange(len(real_z)).cuda()
        # for e in range(10000):
        #     preds = classifier(x)
        #     c = (preds.data.argmax(1) == labels).float().mean()
        #     if c > 0.99:
        #         break
        #     loss = F.cross_entropy(preds, labels)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #
        # recons = decoder(real_z).detach().sigmoid()
        # preds = classifier(recons)
        # c = (preds.data.argmax(1) == labels).float().mean()
        # #     wandb.log({f"c_loss/{factor}": loss})
        #
        # wandb.summary[f"c/{factor}"] =c
        fig = plt.figure()
        zs = torch.stack(hit_codes).squeeze(2).cpu()
        for i, x in enumerate(zs[1000:, ::4].T):
            plt.hist(x, normed=True, alpha=0.3, label=i)
        plt.legend()
        wandb.log({'hist': wandb.Image(fig)})
        break
