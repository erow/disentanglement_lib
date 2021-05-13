import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
import wandb

from disentanglement_lib.methods.shared.architectures import *

opt_g_class = gin.configurable('opt_g')(torch.optim.Adam)
opt_d_class = gin.configurable('opt_d')(torch.optim.Adam)


class GAN(pl.LightningModule):
	def __init__(
			self,
			img_shape,
			num_latent: int = 100,
			activate=torch.tanh,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		self.activate = activate
		# networks
		self.generator = DeepConvDecoder(num_latent, img_shape)
		self.discriminator = Discriminator(img_shape=img_shape)

	def forward(self, z):
		return self.activate(self.generator(z))

	def adversarial_loss(self, y_hat, y):
		return F.binary_cross_entropy(y_hat, y)

	def generating_step(self, batch, sampled_z):
		imgs, _ = batch

		self.generated_imgs = self(sampled_z)
		# ground truth result (ie: all fake)
		# put on GPU because we created this tensor inside training_loop
		valid = torch.ones(imgs.size(0), 1)
		valid = valid.type_as(imgs)

		# adversarial loss is binary cross-entropy
		g_loss = self.adversarial_loss(self.discriminator(self(sampled_z)), valid)
		tqdm_dict = {'g_loss': g_loss}
		output = ({
			'loss': g_loss,
			'progress_bar': tqdm_dict,
			'log': tqdm_dict
		})
		return output

	def discriminating_step(self, batch, sampled_z):
		imgs, _ = batch
		# how well can it label as real?
		valid = torch.ones(imgs.size(0), 1)
		valid = valid.type_as(imgs)

		real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

		# how well can it label as fake?
		fake = torch.zeros(imgs.size(0), 1)
		fake = fake.type_as(imgs)

		fake_loss = self.adversarial_loss(
			self.discriminator(self(sampled_z).detach()), fake)

		# discriminator loss is the average of these
		d_loss = (real_loss + fake_loss) / 2
		tqdm_dict = {'d_loss': d_loss}
		output = ({
			'loss': d_loss,
			'progress_bar': tqdm_dict,
			'log': tqdm_dict
		})
		return output

	def training_step(self, batch, batch_idx, optimizer_idx):
		imgs, _ = batch

		# sample noise
		z = torch.randn(imgs.shape[0], self.hparams.num_latent)
		z = z.type_as(imgs)

		# train generator
		if optimizer_idx == 0:
			return self.generating_step(batch, z)

		# train discriminator
		if optimizer_idx == 1:
			return self.discriminating_step(batch, z)

	def configure_optimizers(self):
		opt_g = opt_g_class(self.generator.parameters(), )
		opt_d = opt_d_class(self.discriminator.parameters(), )
		return [opt_g, opt_d], []

	def on_epoch_end(self):
		z = torch.randn(8, self.hparams.num_latent, device=self.device)

		# log sampled images
		sample_imgs = self(z)
		grid = torchvision.utils.make_grid(sample_imgs)
		self.logger.experiment.log({'generated_images': wandb.Image(grid)})
