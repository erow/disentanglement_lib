# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from disentanglement_lib.methods.shared import losses
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised.model import BaseVAE, compute_gaussian_kl
from disentanglement_lib.methods.shared import architectures
import gin
import argparse

from disentanglement_lib.visualize.visualize_util import plt_sample_traversal

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--alpha', type=float, default=2)
parser.add_argument('-s', '--seed', type=int, default=999)
args, unknown = parser.parse_known_args()
alpha = args.alpha

base_path = "example_output"
overwrite = True
# We save the results in a `vae` subfolder.
path_vae = os.path.join(base_path, "vae")


@gin.configurable("deft_s")
class DEFTVAES(BaseVAE):
    # shared decoder and simultaneously update
    def __init__(self, input_shape):
        super().__init__(input_shape)
        self.betas = torch.FloatTensor([1, 1, 1, 1, 1])
        self.active_latent = self.num_latent

    def get_decoder(self, i):
        return self.decode

    def regularizer(self, kl, z_mean, z_logvar, z_sampled):
        # 防止未惩罚的维度增加信息
        return kl.sum() * 1e-6

    def model_fn(self, features, labels):
        """Training compatible model function."""
        del labels
        self.summary = {}
        z_mean, z_logvar = self.encode(features)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
        self.stage = min(self.global_step // self.stage_steps, self.num_latent - 1)
        if self.global_step % self.stage_steps == 0:
            print(self.stage)
        # z_sampled.register_hook(self.z_hook)
        kl = compute_gaussian_kl(z_mean, z_logvar)

        loss = self.regularizer(kl, z_mean, z_logvar, z_sampled)
        for i in range(self.active_latent):
            r = torch.randn_like(z_sampled)
            r[:, :i + 1] = z_sampled[:, :i + 1]

            reconstructions = self.get_decoder(i)(r)
            per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
            reg_recon = torch.mean(per_sample_loss)
            self.summary[f'reg_recon/{i}'] = reg_recon
            reg_loss = reg_recon + kl[:i + 1].sum() * self.betas[i]
            # print(loss,reg_loss,self.alpha)
            loss = loss + reg_loss * pow(self.alpha, i)
        if self.alpha == 1:
            s = self.num_latent
        else:
            s = (1 - pow(self.alpha, self.num_latent)) / (1 - self.alpha)
        loss = loss / s

        self.summary['loss'] = loss
        self.summary['kl_loss'] = kl.sum()

        for i in range(kl.shape[0]):
            self.summary[f"kl/{i}"] = kl[i]

        self.global_step += 1
        return loss, self.summary


@gin.configurable("deft_i")
class DEFTIVAEI(DEFTVAES):
    def __init__(self, input_shape):
        super().__init__(input_shape)
        # self.betas = torch.FloatTensor([10, 4, 1, 1, 1])
        self.decodes = \
            torch.nn.Sequential(*[architectures.lite_decoder(self.num_latent, input_shape)
                                  for _ in range(self.num_latent - 1)])

    def get_decoder(self, i):
        if i == self.num_latent - 1:
            return self.decode
        return self.decodes[i]


@gin.configurable("deft_ii")
class DEFTIVAESI(DEFTVAES):
    # shared decoder and iteratively update
    def regularizer(self, kl, z_mean, z_logvar, z_sampled):
        self.active_latent = self.stage
        return kl.sum() * 1e-6


@gin.configurable("deft_iii")
class DEFTIVAEII(DEFTIVAEI):
    # independent decoders and iteratively update
    def regularizer(self, kl, z_mean, z_logvar, z_sampled):
        self.active_latent = self.stage
        return kl.sum() * 1e-6


class TestTrain(train.Train):
    def visualize_model(self, model) -> dict:
        log = super().visualize_model(model)
        latent_dim = self.ae.num_latent

        def to_decoder(decode):
            def _decoder(latent_vectors):
                with torch.no_grad():
                    z = torch.Tensor(latent_vectors)
                    torch_imgs = decode(z).numpy()
                    return torch_imgs.transpose((0, 2, 3, 1))

            return _decoder

        if hasattr(model, 'decodes'):
            for i, decode in enumerate(model.decodes):
                mu = torch.zeros(1, latent_dim)
                fig = plt_sample_traversal(mu, to_decoder(decode), 8, range(latent_dim), 2)
                log.update({f'traversal/{i}': wandb.Image(fig)})

        return log

    def evaluate(self) -> None:
        model = self.ae
        model.cpu()
        model.eval()
        dic_log = {}
        dic_log.update(self.visualize_model(model))
        dic_log.update(self.compute_mig(model))
        dic_log.update(self.compute_zmi(model))
        wandb.log(dic_log)
        model.cuda()
        model.train()


bindings = ["mig.num_train=10000",
            f"train.random_seed={args.seed}",
            "train.training_steps = 22000",
            f"model.alpha={alpha}",
            "model.stage_steps=4000"] + [i[2:] for i in unknown]
gin.parse_config_files_and_bindings(["model.gin"], bindings, skip_unknown=True)

logger = WandbLogger(project='experiments', tags=['v2', 'hyperparameter'], save_dir='/tmp/wandb')
print(logger.experiment.url)
pl_model = TestTrain()
trainer = pl.Trainer(logger,
                     progress_bar_refresh_rate=0,
                     max_steps=pl_model.training_steps,
                     checkpoint_callback=False,
                     gpus=1, )

trainer.fit(pl_model)
