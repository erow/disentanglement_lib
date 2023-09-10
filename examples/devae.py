from PIL import Image
import numpy as np
from functools import partial
from random import shuffle
import gin
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import wandb
from disentanglement_lib.data.named_data import get_named_ground_truth_data
from disentanglement_lib.methods.shared.architectures import *
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import callbacks
from disentanglement_lib.methods.unsupervised.model import *
from disentanglement_lib.evaluation.metrics.mig import compute_mig
import os
import pytorch_lightning as pl
import torchvision
from torch import nn
from disentanglement_lib.methods.unsupervised.model import Regularizer
from disentanglement_lib.utils.results import save_gin
from disentanglement_lib.visualize.visualize_util import plt_sample_traversal

from pytorch_lightning.callbacks import ModelCheckpoint

    
class ConvEncoder(architectures.conv_encoder):
    def __init__(self,input_shape, num_latent, base_channel=32,num_stages=10) -> None:
        super().__init__(input_shape, num_latent, base_channel)
        self.projs = torch.nn.ModuleList(
            [Projection(num_latent) for i in range(num_stages)]
            )
    
    def forward(self, x, stage=0,training=False):
        means, log_var = super().forward(x)
        save_rep=[(means, log_var)]
        for i in range(stage):
            means, log_var = self.projs[i](means, log_var)
            if training:
                save_rep.append((means, log_var))
        if training:
            return save_rep
        else:
            return means, log_var
    
class ConvDecoder(architectures.deconv_decoder):
    def __init__(self, num_latent, output_shape,num_stages=10):
        super().__init__(num_latent * 2, output_shape)
        self.stage_embed = nn.Embedding(num_stages,num_latent) # 管够
        
    def forward(self, z, stage = 0):
        bs = len(z)
        device = z.device
        stage_embedding = self.stage_embed(torch.LongTensor([stage]*bs).to(device)).to(device)
        z = torch.cat([z,stage_embedding],1)
        return super().forward(z)
    
            
@gin.configurable('devae')
class DeVAE(train.PLModel):
    def __init__(self,
                 scale = 1,
                 betas=[1,10,40],
                 **kwargs):
        super().__init__(
            encoder_fn=partial(ConvEncoder,num_stages=len(betas)),
            decoder_fn=partial(ConvDecoder,num_stages=len(betas)),
            **kwargs)
        self.scale = float(scale)
        self.betas = betas
    
    
    def model_fn(self, features, labels, global_step):
        """Training compatible model function."""
        self.summary = {}
        # z_mean, z_logvar = self.encode(features)
        bs = len(features)
       
        K = len(self.betas)
        
        a = 1
        loss = 0
        save_rep = self.encode(features,K-1,training=True)
        for stage in range(K):
            mu, logvar = save_rep[stage]
            z = sample_from_latent_distribution(mu, logvar)
            reconstructions = self.decode(z, stage)
            
            reconstruction_loss = losses.make_reconstruction_loss(features,reconstructions).mean()
            kl_loss = compute_gaussian_kl(mu, logvar).sum()
            lossi = reconstruction_loss + kl_loss*self.betas[stage]
            loss = loss + a*lossi
            a*=self.scale

            self.summary[f'reconstruction_loss/{stage}'] = reconstruction_loss.item()
            self.summary[f'kl_loss/{stage}'] = kl_loss.item()

    
        self.summary['loss'] = loss

        return loss, self.summary

    def convert(self, device='cpu',stage=0):
        def _decoder(latent_vectors):
            with torch.no_grad():
                z = torch.FloatTensor(latent_vectors).to(device)
                imgs = self.decode(z,stage).cpu().sigmoid().numpy()
                return imgs.transpose((0, 2, 3, 1))

        def _encoder(obs):
            with torch.no_grad():
                # if isinstance(obs,torch.Tensor):
                obs = torch.FloatTensor(obs.transpose((0, 3, 1, 2))).to(device)  # convert tf format to torch's
                mu, logvar = self.encode(obs,stage)
                mu, logvar = mu.cpu().numpy(), logvar.cpu().numpy()
                return mu, logvar

        return _encoder, _decoder

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=99)
    parser.add_argument('--output_dir',type=str,default='outputs')
    parser.add_argument('-c', '--configs', default=["model.gin"],nargs='+')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--ngpus',type=int,default=1)
    
    args, unknown = parser.parse_known_args()
    pl.seed_everything(args.seed)
    for gin_file in args.configs:
        print('load gin config', gin_file)
        gin.parse_config_file(gin_file)
        
    
    if unknown:
        unknown = [i.strip('--') for i in unknown]
        gin.parse_config(unknown)
        
    print(gin.config_str())
    
    CALLBACK_STEPS = args.max_steps//10
    pl_model = DeVAE(seed=args.seed)
    dataset = get_named_ground_truth_data()    
    dl = torch.utils.data.DataLoader(dataset,args.batch_size,num_workers=4,pin_memory=True, shuffle=True )
    
    
    callbacks_fn = [
        callbacks.Traversal(CALLBACK_STEPS),
        callbacks.ShowSamples(CALLBACK_STEPS,dataset),
        ModelCheckpoint(args.output_dir,'checkpoint'),
    ]
    if dataset.supervision: 
        callbacks_fn.append(callbacks.ComputeMetric(CALLBACK_STEPS,compute_mig,dataset))

    os.makedirs(args.output_dir,exist_ok=True)
    # save all config
    save_gin(os.path.join(args.output_dir, "model.gin"))
    
    name = os.path.basename(args.output_dir)
    if args.wandb:
        logger = WandbLogger(save_dir=args.output_dir,name=name,resume=True)
    else:
        logger = CSVLogger(args.output_dir,name=name)

    ckpt_path = os.path.join(args.output_dir,'checkpoint.ckpt')
    trainer = pl.Trainer(
        logger,
        default_root_dir=args.output_dir,
        resume_from_checkpoint=ckpt_path if os.path.exists(ckpt_path) else None,
        accelerator='gpu', devices=args.ngpus,
        max_steps=args.max_steps,
        callbacks=callbacks_fn,
    )
    trainer.fit(pl_model, dl)
