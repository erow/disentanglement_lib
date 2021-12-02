import gin
import numpy as np
import torch
import wandb
from torch import nn

from disentanglement_lib.methods.shared import losses
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.evaluation.metrics import mig
from disentanglement_lib.methods.unsupervised.model import gaussian_log_density, sample_from_latent_distribution

from disentanglement_lib.utils.mi_estimators import estimate_entropies
import torch.nn.functional as F

class Evaluation(Callback):
    def __init__(self, every_n_step):
        self.every_n_step = every_n_step
        self.log={}

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            if (trainer.global_step+1) % self.every_n_step == 0:
                log = self.compute(pl_module, trainer.train_dataloader)
                self.log=log
                wandb.log(log, step=trainer.global_step)

    @torch.no_grad()
    def compute(self, model, train_dl=None):
        raise NotImplementedError()

class Decomposition(Evaluation):
    @torch.no_grad()
    def compute(self, model, train_dl=None):
        """
        reference: https://github.com/rtqichen/beta-tcvae/blob/master/elbo_decomposition.py
        :param model:
        :param dataset_loader:
        :return: dict(): TC, MI, DWKL
        """
        device = model.device
        model.eval()

        dataset_loader = train_dl if train_dl else self.dl
        N = len(dataset_loader.dataset)  # number of data samples
        K = model.num_latent  # number of latent variables
        S = 1  # number of latent variable samples
        nparams = 2

        print('Computing q(z|x) distributions.')
        # compute the marginal q(z_j|x_n) distributions
        qz_params = torch.Tensor(N, K, nparams).to(device)
        n = 0
        for samples in dataset_loader:
            xs, _ = samples
            batch_size = xs.size(0)
            xs = xs.view(batch_size, -1, 64, 64).to(device)
            mu, logvar = model.encode(xs)
            qz_params[n:n + batch_size, :, 0] = mu.data
            qz_params[n:n + batch_size, :, 1] = logvar.data
            n += batch_size
        z_sampled = sample_from_latent_distribution(
            qz_params[..., 0], qz_params[..., 1])

        # pz = \sum_n p(z|n) p(n)
        logpz = gaussian_log_density(z_sampled, torch.zeros_like(
            z_sampled), torch.zeros_like(z_sampled)).mean(0)
        logqz_condx = gaussian_log_density(
            z_sampled, qz_params[..., 0], qz_params[..., 1]).mean(0)

        z_sampled = z_sampled.transpose(0, 1).contiguous().view(K, N * S)
        marginal_entropies, joint_entropy = estimate_entropies(
            z_sampled, qz_params)

        # Independence term
        # KL(q(z)||prod_j q(z_j)) = log q(z) - sum_j log q(z_j)
        dependence = (- joint_entropy + marginal_entropies.sum()).item()

        # Information term
        # KL(q(z|x)||q(z)) = log q(z|x) - log q(z) = H(z) - H(z|x)
        H_zCx = -logqz_condx.sum().item()
        H_qz = joint_entropy.item()
        information = (joint_entropy - H_zCx).item()
        z_information = (marginal_entropies +
                         logqz_condx).cpu().numpy().round(2)

        # Dimension-wise KL term
        # sum_j KL(q(z_j)||p(z_j)) = sum_j (log q(z_j) - log p(z_j))
        dimwise_kl = (marginal_entropies - logpz).sum().item()

        # Compute sum of terms analytically
        # KL(q(z|x)||p(z)) = log q(z|x) - log p(z)
        analytical_cond_kl = (logqz_condx - logpz).sum().item()

        print('Dependence: {}'.format(dependence))
        print('Information: {}'.format(information))
        print('Dimension-wise KL: {}'.format(dimwise_kl))
        print('Analytical E_p(x)[ KL(q(z|x)||p(z)) ]: {}'.format(
            analytical_cond_kl))
        log = dict(TC=dependence,
                   MI=information,
                   ZMI=z_information,
                   DWKL=dimwise_kl,
                   KL=analytical_cond_kl,
                   H_q_zCx=H_zCx,
                   H_q_z=H_qz)
        model.train()
        return log


class ComputeMetric(Evaluation):
    def __init__(self, every_n_step, metric_fn):
        self.every_n_step = every_n_step
        self.metric_fn = metric_fn

    @torch.no_grad()
    def compute(self, model, train_dl) -> dict:
        device = model.device
        model.eval()
        model.cpu()
        _encoder, _decoder = model.convert()
        
        dataset = train_dl.dataset.datasets
        result = self.metric_fn(dataset, lambda x: _encoder(x)[
                                 0], np.random.RandomState(), )
        model.to(device)
        model.train()
        return result


class Visualization(Evaluation):
    def __init__(self, every_n_step):
        self.every_n_step = every_n_step

    @torch.no_grad()
    def compute(self, model, train_dl) -> dict:
        from disentanglement_lib.visualize.visualize_util import plt_sample_traversal
        device = model.device
        model.eval()
        model.cpu()
        _encoder, _decoder = model.convert()
        num_latent = model.num_latent
        mu = torch.zeros(1, num_latent)
        fig = plt_sample_traversal(mu, _decoder, 8, range(num_latent), 2)
        model.to(device)
        model.train()
        return {'traversal': wandb.Image(fig)}


# note: not finished
# class FactorMI(Evaluation):
#     def compute(self, model, train_dl=None):
#         """
#         reference: https://github.com/rtqichen/beta-tcvae/blob/master/elbo_decomposition.py
#         :param model:
#         :param dataset_loader:
#         :return: dict(): TC, MI, DWKL
#         """
#         dataset_loader = train_dl if train_dl else self.dl
#         N = len(dataset_loader.dataset)  # number of data samples
#         K = model.num_latent  # number of latent variables
#         S = 1  # number of latent variable samples
#         nparams = 2
#         qz_params = torch.Tensor(N, K, nparams)
#         y = np.zeros((N, train_dl.dataset.num_factors), dtype=np.int)
#         n = 0
#         for samples in dataset_loader:
#             xs, labels = samples
#             batch_size = xs.size(0)
#             xs = xs.view(batch_size, -1, 64, 64).to(device)
#             mu, logvar = model.encode(xs)
#             qz_params[n:n + batch_size, :, 0] = mu.data.cpu()
#             qz_params[n:n + batch_size, :, 1] = logvar.data.cpu()
#             y[n:n + batch_size] = labels
#             n += batch_size

#         from disentanglement_lib.evaluation.metrics import RMIG
#         log = RMIG.estimate_JEMMIG_cupy(qz_params[..., 0].numpy(), qz_params[..., 1].numpy(), y)
#         return log
