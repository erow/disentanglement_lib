import os
from tkinter import Image
import gin, wandb
import numpy as np
import torch
from torch import nn

from disentanglement_lib.methods.shared import losses
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from disentanglement_lib.methods.unsupervised.model import gaussian_log_density, sample_from_latent_distribution

from disentanglement_lib.utils.mi_estimators import estimate_entropies
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

class EarlyStop(Callback):
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch_output, batch, batch_idx) -> None:
        if pl_module.summary['reconstruction_loss']<1:
            trainer.should_stop=True
            
        
class Evaluation(Callback):
    def __init__(self, every_n_step, prefix=""):
        self.every_n_step = every_n_step
        self.log={}
        self.prefix = prefix

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,batch_output, batch, batch_idx):
        if trainer.global_rank == 0 and (trainer.global_step+1) % self.every_n_step == 0:
            logger = trainer.logger
            log = self.compute(pl_module, trainer.train_dataloader)
            try:
                logger.log_metrics({os.path.join(self.prefix,k):v for k,v in log.items()})
            except Exception as e:
                print(e)
                # logger.log_metrics(log)


    @torch.no_grad()
    def compute(self, model, train_dl=None):
        raise NotImplementedError()

class Decomposition(Evaluation):
    def __init__(self, every_n_step,ds,prefix=""):
        super().__init__(every_n_step,prefix)
        self.ds = ds
    @torch.no_grad()
    def compute(self, model, train_dl=None):
        """Compute the decomposition of the KL: TC, ML, and DWKL.
        reference: https://github.com/rtqichen/beta-tcvae/blob/master/elbo_decomposition.py
        
        :param model:
        :param dataset_loader:
        :return: dict(): TC, MI, DWKL
        """
        device = model.device
        model.eval()

        N = len(self.ds)  # number of data samples
        K = model.num_latent  # number of latent variables
        S = 1  # number of latent variable samples
        nparams = 2

        print('Computing q(z|x) distributions.')
        # compute the marginal q(z_j|x_n) distributions
        qz_params = torch.Tensor(N, K, nparams).to(device)
        n = 0
        for samples in DataLoader(self.ds,64,num_workers=4):
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

class ShowSamples(Evaluation):
    def __init__(self, every_n_step,ds,number=16,prefix="viz"):
        """Show samples

        Args:
            every_n_step (_type_): _description_
            ds (_type_): _description_
            number (int, optional): _description_. Defaults to 16, must be 8x.
            prefix (str, optional): _description_. Defaults to "viz".
        """
        super().__init__(every_n_step,prefix)
        self.ds = ds
        self.number = number

    @torch.no_grad()
    def compute(self, model, train_dl=None):
        device = model.device
        model.eval()
        for samples in train_dl:
            xs, _ = samples
            batch_size = xs.size(0)
            xs = xs[:min(batch_size,self.number)].to(device)
            mu, logvar = model.encode(xs)
            z =sample_from_latent_distribution(mu, logvar)
            recons = model.decode(z).data.sigmoid()
            break

        pic = make_grid(torch.cat([recons,xs]).cpu(), 8,pad_value=1)
        fig: Image = to_pil_image(pic) 

        model.train()

        trainer = model.trainer
        logger = model.trainer.logger
        # dirpath = os.path.join(trainer.log_dir, str(logger.name), logger.version)
        try:
            logger.experiment.log({os.path.join(self.prefix,'samples'):wandb.Image(fig)})   
        except Exception as e:
            dirpath = logger.log_dir
            print(e)
            os.makedirs(os.path.join(dirpath,self.prefix),exist_ok=True)
            fig.save(os.path.join(dirpath,self.prefix,f"samples_{trainer.global_step}.png"))
        return {}

class ComputeMetric(Evaluation):
    def __init__(self, every_n_step, metric_fn, dataset=None,prefix=""):
        super().__init__(every_n_step,prefix)
        self.metric_fn = metric_fn
        self.dataset = dataset
        self.representation_fun='mean'

    @torch.no_grad()
    def compute(self, model, train_dl) -> dict:
        device = model.device
        model.eval()
        model.cpu()
        _encoder, _decoder = model.convert()
        def sample_latent(x):
            mu, logvar = _encoder(x)
            e = np.random.randn(*mu.shape)
            if self.representation_fun=='mean':
                return mu
            else:
                return mu + np.exp(logvar/2) *e
        if self.dataset is None:
            dataset = train_dl.dataset.datasets
        else:
            dataset = self.dataset
        result = self.metric_fn(dataset, 
                                sample_latent,
                                np.random.RandomState(),)
        model.to(device)
        model.train()
        return result

class Traversal(Evaluation):
    def __init__(self, every_n_step,prefix="viz"):
        super().__init__(every_n_step,prefix)

    @torch.no_grad()
    def compute(self, model, train_dl) -> dict:                
        from disentanglement_lib.visualize.visualize_util import plt_sample_traversal
        device = model.device
        trainer = model.trainer
        model.eval()
        model.cpu()
        _encoder, _decoder = model.convert()
        num_latent = model.num_latent
        mu = torch.zeros(1, num_latent)
        fig = plt_sample_traversal(mu, _decoder, 8, range(num_latent), 2)
        model.to(device)
        model.train()
        
        trainer = model.trainer
        logger = model.trainer.logger
        # dirpath = os.path.join(trainer.log_dir, str(logger.name), logger.version)
        
        try:
            logger.experiment.log({os.path.join(self.prefix,'traversal'):wandb.Image(fig)})
        except Exception as e:
            dirpath = logger.log_dir
            print(e)
            os.makedirs(os.path.join(dirpath,self.prefix),exist_ok=True)
            fig.savefig(os.path.join(dirpath,self.prefix,f"traversal_{trainer.global_step}.png"))
        return {}

class Projection(Evaluation):
    def __init__(self, every_n_step,dataset, factor_list,latent_list,
        title="", prefix="viz"):
        super().__init__(every_n_step,prefix)
        self.latent_list = latent_list
        self.factor_list = factor_list
        self.title = str(factor_list)+"->"+str(latent_list)

        assert len(factor_list)==2 and len(latent_list) ==2

        factor_sizes = dataset.factors_num_values
        c1 = np.unique(np.linspace(0,factor_sizes[factor_list[0]]-1,7).astype(int))
        c2 = np.unique(np.linspace(0,factor_sizes[factor_list[1]]-1,7).astype(int))
        self.c1 = c1
        self.c2 = c2

        c = dataset.sample_factors(1,np.random.RandomState())
        c = np.repeat(c,len(c1)*len(c2),axis=0)
        c1,c2 = np.meshgrid(c1,c2,indexing='ij')
        
        c[:,factor_list[0]] = c1.flatten()
        c[:,factor_list[1]] = c2.flatten()

        self.x = torch.tensor(dataset.sample_observations_from_factors(c, np.random.RandomState())).permute(0,3,1,2)



    @torch.no_grad()
    def compute(self, model, train_dl) -> dict:
        from matplotlib.patches import Ellipse
        import matplotlib.pyplot as plt

        model.eval()
        with torch.no_grad():
            mu, logvar = model.encode(self.x.to(model.device).to(model.dtype))
        model.train()
        sigma = (logvar/2).exp().cpu().data
        mu = mu[:,self.latent_list].cpu().data.numpy()
        sigma2 = sigma[:,self.latent_list].numpy()*2

        fig, ax = plt.subplots()
        for i in range(len(mu)):
            ax.add_patch(Ellipse(mu[i],sigma2[i,0]*2,2*sigma2[i,1],alpha=0.5,color='g'))
            # 类似直径
        
        z = mu.reshape(len(self.c1),len(self.c2),2)
        for i in range(z.shape[0]):
            plt.plot(*z[i].T)
        plt.grid()
        plt.title(self.title)
        
        plt.xlabel("z="+str(self.latent_list[0]))
        plt.ylabel("z="+str(self.latent_list[1]))

        
        trainer = model.trainer
        logger = model.trainer.logger
        # dirpath = os.path.join(trainer.log_dir, str(logger.name), logger.version)
        try:
            logger.experiment.log({os.path.join(self.prefix,f'traversal_{self.title}'):wandb.Image(fig,caption=self.title)})
        except Exception as e:
            dirpath = logger.log_dir
            print(e)
            os.makedirs(os.path.join(dirpath,self.prefix),exist_ok=True)
            fig.savefig(os.path.join(dirpath,self.prefix,f"traversal_{self.title}_{trainer.global_step}.png"))
        return {}
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
