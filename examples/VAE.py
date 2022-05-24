#%% VAE example
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from disentanglement_lib.methods.unsupervised.model import (
    compute_gaussian_kl, sample_from_latent_distribution)
from torch import nn
from torch.utils import *


class VAE(pl.LightningModule):
    def __init__(self,d_in=1, num_latent=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in,10),nn.ReLU(),
            nn.Linear(10,10),nn.ReLU(),
            nn.Linear(10,num_latent*2),
        )

        self.decode = nn.Sequential(
            nn.Linear(num_latent,10),nn.ReLU(),
            nn.Linear(10,10),nn.ReLU(),
            nn.Linear(10,d_in),
        )

    def encode(self,x):
        return torch.split(self.encoder(x),[1,1],1)

    def forward(self, x):
        mu, logvar =  self.encode(x)
        z = sample_from_latent_distribution(mu, logvar)
        kl = compute_gaussian_kl(mu,logvar).sum()
        recon = self.decode(z)
        recon_loss = ((recon-x)**2/2).mean()

        t = self.global_step/10000
        beta = np.exp(-t*4)+0.001
        loss =  recon_loss + kl*beta
        self.log('beta',beta)
        self.log('elbo',-(recon_loss+kl))
        self.log('recon_loss', recon_loss)
        return loss, recon_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, x,batch_idx):
        loss, _ = self(x)
        return loss

from torch.utils.data import DataLoader, IterableDataset


class DiscreteData(IterableDataset):
    def __init__(self, num, with_noise=False) -> None:
        super().__init__()
        self.num = num
        self.with_noise = with_noise
        self.data = torch.linspace(-2,2,num)
    
    def __iter__(self):
        return self

    def __next__(self):
        data = self.data[np.random.randint(self.num)].reshape(-1)
        if self.with_noise:
            return torch.cat([data,torch.randn(1)])
        else:
            return data

class ContinuesData(IterableDataset):
    def __init__(self, ) -> None:
        super().__init__()
    
    def __iter__(self):
        return self

    def __next__(self):
        data = torch.rand(1)*4-2
        return data

dl = DataLoader(DiscreteData(10,True),64)
for x in dl:break
x.shape
#%%
dl = DataLoader(DiscreteData(10,True),64)
model = VAE(2)
trainer = pl.Trainer(
    max_steps=20000
)
trainer.fit(model,dl)

# %%
m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

x = torch.linspace(-2,2,10).reshape(-1,1)
mu,logvar = model.encode(x)

plt.plot(x.flatten(),m
.log_prob(x).exp().flatten())
p = m.log_prob(mu.data).exp().flatten()
for i in range(len(x)):
    plt.arrow(x[i],0,(mu[i].data-x[i]),p[i])

# %%
zs = sample_from_latent_distribution(mu.repeat(1,1000),logvar.repeat(1,1000)).data

x = torch.linspace(-3,3,20).reshape(-1,1)
p = m.log_prob(x.data).exp().flatten()
plt.plot(x.flatten(),m.log_prob(x).exp().flatten())
plt.hist(zs.flatten().numpy(),bins=40,density=True)
plt.title('z vs nz')
# %%

for i in range(zs.shape[0]):
    sns.distplot(zs[i].numpy(),label=str(i))
# %% 

zs = []
for i,x in enumerate(dl):
    if i>1000: break
    mu,logvar = model.encode(x)
    z = sample_from_latent_distribution(mu,logvar).data
    zs.append(z.flatten())
zs = torch.stack(zs).numpy()
# %%
x = torch.linspace(-2,2,20).reshape(-1,1)
plt.plot(x.flatten(),m.log_prob(x).exp().flatten())
plt.hist(zs.flatten(),bins=40,density=True)
plt.title('z vs nz')

# %%
plt.xlim(-3,3)
plt.ylim(0,5)
for t in np.linspace(-2,2,10)[::4]:
    x = torch.randn(100,2)
    x[:,0] = t
    mu,logvar = model.encode(x)
    zs = sample_from_latent_distribution(mu.repeat(1,1000),logvar.repeat(1,1000)).data
    sns.distplot(zs.data.flatten().numpy())

# %%
# %%

