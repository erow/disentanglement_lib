import gin
import numpy as np
import torch
import wandb
from torch import nn

from disentanglement_lib.methods.shared import losses
from torch.utils.data import DataLoader

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.evaluation.metrics import mig
from disentanglement_lib.methods.unsupervised.model import gaussian_log_density
from disentanglement_lib.utils.hub import convert_model
from disentanglement_lib.utils.mi_estimators import estimate_entropies
from disentanglement_lib.visualize.visualize_util import plt_sample_traversal
import torch.nn.functional as F


class Evaluation():
    def __init__(self, evaluation_steps):
        self.evaluation_steps = evaluation_steps
        self.global_steps = 0

    def __call__(self, model, train_dl, global_steps):
        if (global_steps + 1) % self.evaluation_steps == 0:
            self.global_steps = global_steps
            return self.compute(model, train_dl)
        return {}

    def compute(self, model, train_dl):
        return NotImplemented


@gin.configurable('eval_decomposition')
class Decomposition(Evaluation):
    def __init__(self, evaluation_steps, dataset=None, random_seed=99, batch_size=256):
        super().__init__(evaluation_steps)
        if dataset:
            self.data = named_data.get_named_ground_truth_data(dataset)
            self.dl = DataLoader(self.data, batch_size=batch_size, num_workers=4, pin_memory=True)
            self.random_state = np.random.RandomState(random_seed)

    def compute(self, model, train_dl=None):
        """
        reference: https://github.com/rtqichen/beta-tcvae/blob/master/elbo_decomposition.py
        :param model:
        :param dataset_loader:
        :return: dict(): TC, MI, DWKL
        """
        dataset_loader = train_dl if train_dl else self.dl
        N = len(dataset_loader.dataset)  # number of data samples
        K = model.num_latent  # number of latent variables
        S = 1  # number of latent variable samples
        nparams = 2

        print('Computing q(z|x) distributions.')
        # compute the marginal q(z_j|x_n) distributions
        qz_params = torch.Tensor(N, K, nparams).cuda()
        n = 0
        for samples in dataset_loader:
            if self.data.supervision:
                xs, labels = samples
            else:
                xs, _ = samples
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
        z_information = (marginal_entropies + logqz_condx).cpu().numpy().round(2)

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
        return log


@gin.configurable('eval_mig')
class MIGScore(Evaluation):
    def compute(self, model, train_dl) -> dict:
        model.cpu()
        _encoder, _decoder = convert_model(model, device='cpu')
        result = mig.compute_mig(train_dl.dataset, lambda x: _encoder(x)[0], np.random.RandomState(), )
        model.cuda()
        return result


@gin.configurable('eval_visualization')
class Visualization(Evaluation):
    def compute(self, model, train_dl) -> dict:
        _encoder, _decoder = convert_model(model)
        num_latent = model.num_latent
        mu = torch.zeros(1, num_latent)
        fig = plt_sample_traversal(mu, _decoder, 8, range(num_latent), 2)
        return {'traversal': wandb.Image(fig)}


@gin.configurable('eval_H_xCz')
class H_xCz(Evaluation):
    @torch.no_grad()
    def compute(self, model, train_dl):
        num_samples = 10000
        decoder = model.decode
        res = {}
        test_samples = int(np.sqrt(num_samples))
        assert test_samples * test_samples == num_samples
        for i in range(model.num_latent + 1):
            if i == 0:
                z = torch.randn(num_samples, model.num_latent, device=device)
                recons = torch.sigmoid(decoder(z))
                img_size = np.prod(recons.shape[1:])
                recons = recons.reshape(1, num_samples, img_size)
                mean_recons = torch.mean(recons, 1, keepdim=True).repeat([1, num_samples, 1])

            elif i == model.num_latent:
                z = torch.randn(num_samples, model.num_latent, device=device)
                recons = torch.sigmoid(decoder(z))
                img_size = np.prod(recons.shape[1:])
                recons = recons.reshape(num_samples, img_size)
                mean_recons = recons

            else:
                z1 = torch.randn(test_samples, 1, i, device=device).repeat([1, test_samples, 1]).view(
                    test_samples * num_samples, -1)
                z2 = torch.randn(1, test_samples, model.num_latent - i, device=device).repeat(
                    [test_samples, 1, 1]).view(test_samples * num_samples, -1)
                z = torch.cat([z1, z2], 1)
                recons = torch.sigmoid(decoder(z))
                img_size = np.prod(recons.shape[1:])
                recons = recons.reshape(test_samples, test_samples, img_size)
                mean_recons = torch.mean(recons, 1, keepdim=True).repeat([1, test_samples, 1])

            loss = F.binary_cross_entropy(recons.reshape(num_samples, img_size),
                                          mean_recons.reshape(num_samples, img_size),
                                          reduction="none")
            res[f"H_xCz/{i}"] = loss.mean(0).cpu()
        return res


@gin.configurable('eval_cmi')
class FactorMI(Evaluation):
    def compute(self, model, train_dl=None):
        """
        reference: https://github.com/rtqichen/beta-tcvae/blob/master/elbo_decomposition.py
        :param model:
        :param dataset_loader:
        :return: dict(): TC, MI, DWKL
        """
        dataset_loader = train_dl if train_dl else self.dl
        N = len(dataset_loader.dataset)  # number of data samples
        K = model.num_latent  # number of latent variables
        S = 1  # number of latent variable samples
        nparams = 2
        qz_params = torch.Tensor(N, K, nparams)
        y = np.zeros((N, train_dl.dataset.num_factors), dtype=np.int)
        n = 0
        for samples in dataset_loader:
            xs, labels = samples
            batch_size = xs.size(0)
            xs = xs.view(batch_size, -1, 64, 64).cuda()
            mu, logvar = model.encode(xs)
            qz_params[n:n + batch_size, :, 0] = mu.data.cpu()
            qz_params[n:n + batch_size, :, 1] = logvar.data.cpu()
            y[n:n + batch_size] = labels
            n += batch_size

        from disentanglement_lib.evaluation.metrics import RMIG
        log = RMIG.estimate_JEMMIG_cupy(qz_params[..., 0].numpy(), qz_params[..., 1].numpy(), y)
        return log
