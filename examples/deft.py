from argparse import ArgumentParser
import gin
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data
from disentanglement_lib.methods.shared.architectures import *
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import callbacks
from disentanglement_lib.evaluation.metrics.mig import compute_mig
import os
import pytorch_lightning as pl

from disentanglement_lib.methods.unsupervised.model import Regularizer
os.environ['WANDB_NOTES']="""reproduce experiments on dsprites_full for DEFT"""
os.environ['WANDB_TAGS']='DEFT'
os.environ['WANDB_PROJECT']='dsprites_test'

example_text="""python examples/deft.py --model.seed=0 --total_steps=200000 --deft1.betas=[70,30,15,1]  --dataset.name="'dsprites_noshape'"""""
parser = ArgumentParser(epilog=example_text)

parser.add_argument('--total_steps',default=200000,type=int)
parser.add_argument('--callback_steps',default=10000,type=int)
args, unknown = parser.parse_known_args()
TOTAL_STEPS=args.total_steps
CALLBACK_STEPS = args.callback_steps


@gin.configurable('deft1')
class DEFT(Regularizer):
    def __init__(self,
                 beta_max = 150,
                 gamma = 20,
                 stage_steps=gin.REQUIRED,
                 betas=gin.REQUIRED,):
        super().__init__()
        self.betas = betas
        self.stage_steps = stage_steps
        self.beta_max = beta_max
        self.gamma = gamma

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        """Training compatible model function."""
        global_step = model.global_step
        t = global_step/self.stage_steps
        beta = max(np.exp(-self.gamma*t)*self.beta_max,1)
        stage = model.encode.stage
        if stage < len(self.betas) and beta<= self.betas[stage]:
            stage = stage + 1
            model.encode.set_stage(stage)
            print('stage', stage)

        model.summary['beta'] = beta
        return beta * (kl.sum())
if unknown:
    unknown = [i.strip('--') for i in unknown]
    bindings= [
        "model.encoder_fn = @frac_encoder",
        "model.regularizers = [@deft1()]",
        f"deft1.stage_steps={TOTAL_STEPS}",
    ]
    gin.parse_config(bindings+unknown)
if __name__ == "__main__":
    dataset = get_named_ground_truth_data()
    pl_model = train.PLModel()
    dl = torch.utils.data.DataLoader(train.Iterate(dataset),256,num_workers=10,pin_memory=True)
    logger = WandbLogger()
    trainer = pl.Trainer(
        logger,
        # resume_from_checkpoint='/home/erow/disentanglement_lib/dsprites_test/35yjk1pz/checkpoints/N-Step-Checkpoint_epoch=0_global_step=60000.ckpt',
        max_steps=TOTAL_STEPS,
        # checkpoint_callback=False,
        callbacks=[
            callbacks.ComputeMetric(CALLBACK_STEPS,compute_mig),
            callbacks.Decomposition(CALLBACK_STEPS,dataset),
            callbacks.Traversal(CALLBACK_STEPS),
            callbacks.Projection(CALLBACK_STEPS,dataset, [2,3],[0,1],"posX, posY",key="viz/XY"),
            callbacks.Projection(CALLBACK_STEPS,dataset, [0,0],[2,3],"scale, scale",key='viz/S'),
            callbacks.Projection(CALLBACK_STEPS,dataset, [1,1],[4,5],"rotation, rotation",key='viz/R')
            ],
        gpus=1,)
    trainer.fit(pl_model, dl)
    wandb.join()
