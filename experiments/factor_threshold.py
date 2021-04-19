"""
实验目的：发现action、β、KL 之间的关系。
预期：
1. KL与β成反比，且存在一threshold使得，当β>threshold时，KL<0.1。
2. 这个阈值是否稳定。
"""
import argparse
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from disentanglement_lib.config.unsupervised_study_v1.sweep import UnsupervisedStudyV1
from disentanglement_lib.data.ground_truth import dsprites
from disentanglement_lib.methods.shared import losses
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils, mig
from disentanglement_lib.methods.unsupervised.model import BaseVAE, compute_gaussian_kl, gaussian_log_density
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gin
import pathlib, shutil
import wandb

from disentanglement_lib.utils.mi_estimators import estimate_entropies


class AnnealingTest(train.Train):
    def evaluate(self) -> None:
        dataset_loader = self.train_dataloader()
        model = self.ae
        model.eval()
        self.save_model(f"_{self.global_step}.pt")

        N = len(dataset_loader.dataset)  # number of data samples
        K = model.num_latent  # number of latent variables
        S = 1  # number of latent variable samples
        nparams = 2

        print('Computing q(z|x) distributions.')
        # compute the marginal q(z_j|x_n) distributions
        qz_params = torch.Tensor(N, K, nparams).cuda()
        n = 0
        for xs, labels in dataset_loader:
            batch_size = xs.size(0)
            xs = xs.view(batch_size, -1, 64, 64).cuda()
            mu, logvar = model.encode(xs)
            qz_params[n:n + batch_size, :, 0] = mu.data
            qz_params[n:n + batch_size, :, 1] = logvar.data
            n += batch_size
        z_sampled = model.sample_from_latent_distribution(qz_params[..., 0], qz_params[..., 1])

        # pz = \sum_n p(z|n) p(n)
        logpz = gaussian_log_density(z_sampled, torch.zeros_like(z_sampled), torch.zeros_like(z_sampled)).mean(0)
        logqz_condx = gaussian_log_density(z_sampled, qz_params[..., 0], qz_params[..., 1]).mean(0)

        z_sampled = z_sampled.transpose(0, 1).contiguous().view(K, N * S)
        marginal_entropies, joint_entropy = estimate_entropies(z_sampled, qz_params)

        # Independence term
        # KL(q(z)||prod_j q(z_j)) = log q(z) - sum_j log q(z_j)
        dependence = (- joint_entropy + marginal_entropies.sum()).item()

        # Information term
        # KL(q(z|x)||q(z)) = log q(z|x) - log q(z) = H(z) - H(z|x)
        H_zCx = -logqz_condx.sum().item()
        H_qz = joint_entropy.item()
        information = (joint_entropy - H_zCx).item()
        z_information = (marginal_entropies + logqz_condx).cpu().numpy()

        # Dimension-wise KL term
        # sum_j KL(q(z_j)||p(z_j)) = sum_j (log q(z_j) - log p(z_j))
        dimwise_kl = (marginal_entropies - logpz).sum().item()

        # Compute sum of terms analytically
        # KL(q(z|x)||p(z)) = log q(z|x) - log p(z)
        analytical_cond_kl = (logqz_condx - logpz).sum().item()

        print('Dependence: {}'.format(dependence))
        print('Information: {}'.format(information))
        print('Dimension-wise KL: {}'.format(dimwise_kl))
        print('Analytical E_p(x)[ KL(q(z|x)||p(z)) ]: {}'.format(analytical_cond_kl))
        log = dict(TC=dependence,
                   MI=information,
                   ZMI=z_information,
                   DWKL=dimwise_kl,
                   KL=analytical_cond_kl,
                   H_q_zCx=H_zCx,
                   H_q_z=H_qz)

        wandb.log(log)
        model.train()
        model.cuda()

    def on_fit_start(self) -> None:
        self.evaluate()

    def on_fit_end(self) -> None:
        # self.evaluate()
        model = self.ae
        model.cpu()
        model.eval()
        log = self.visualize_model(model)
        if self.ae.num_latent > 1:
            log.update(self.compute_mig(model))
        wandb.log(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()

    dataset = dsprites.DSprites([3])
    dl = DataLoader(dataset, batch_size=16)
    bindings = ['mig.num_train=10000',
                'train.model=@annealed',
                'annealed.beta_h=40',
                'train.training_steps=30000'] + [i[2:] for i in unknown]
    study = UnsupervisedStudyV1()
    _, share_conf = study.get_model_config()
    gin.parse_config_files_and_bindings([share_conf], bindings, skip_unknown=True)

    logger = WandbLogger(project='dlib', tags=['ICP'])
    print(logger.experiment.url)
    pl_model = AnnealingTest(eval_numbers=10)
    trainer = pl.Trainer(logger,
                         max_steps=pl_model.training_steps,
                         checkpoint_callback=False,
                         progress_bar_refresh_rate=0,
                         gpus=1, )
    trainer.fit(pl_model, dl)
