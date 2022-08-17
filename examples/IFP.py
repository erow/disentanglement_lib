from argparse import ArgumentParser
import gin
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from disentanglement_lib.data.ground_truth.ground_truth_data import RandomAction
from disentanglement_lib.data.ground_truth import get_named_ground_truth_data
from disentanglement_lib.methods.shared.architectures import *
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import callbacks
import os
import pytorch_lightning as pl
from disentanglement_lib.methods.unsupervised.model import Regularizer
os.environ['WANDB_TAGS']='annealed_test'
os.environ['WANDB_PROJECT']='IFP'

@gin.configurable('annealed_test')
class AnnealedTest(Regularizer):
    def __init__(self,
                 gamma = 10,
                 beta_max = 70,
                 stage_steps=gin.REQUIRED):
        super().__init__()
        self.stage_steps = stage_steps
        self.gamma = gamma
        self.beta_max = beta_max

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        """Training compatible model function."""
        global_step = model.global_step
        k = global_step/self.stage_steps
        beta = (np.exp(-self.gamma*k)*self.beta_max + 1)
        model.summary['beta'] = beta
        return beta * (kl.sum())

@gin.configurable('action')
def get_action(dataset,index=gin.REQUIRED):
    return RandomAction(dataset,index)

parser = ArgumentParser()
parser.add_argument('--seed',type=int,default=99)
parser.add_argument('--dataset',type=str,default='dsprites_full')
parser.add_argument('--num_latent',type=int,default=1)
parser.add_argument('-s', '--steps', type=int, default=8000)
parser.add_argument('-g','--gamma',type=float,default=15)
if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    if len(unknown)==0:
        seed= args.seed
        steps = args.steps
        gamma = args.gamma

        bindings= [
            "model.regularizers = [@annealed_test()]",
            f"annealed_test.stage_steps={steps}",
            f"annealed_test.gamma={gamma}",
            f"model.seed={seed}",
            f"dataset.name='{args.dataset}'",
            f"model.num_latent={args.num_latent}"
        ]
        gin.parse_config(bindings)
    else:
        unknown = [i.strip('--') for i in unknown] + ["model.regularizers = [@annealed_test()]"]
        print(unknown)
        gin.parse_config(unknown)

    dataset = get_named_ground_truth_data()
    action = get_action(dataset)
    
    rs = np.random.RandomState(0)
    w,h,c = dataset.observation_shape
    pl_model = train.PLModel(input_shape=[c,w,h])
    dl = torch.utils.data.DataLoader(train.Iterate(action), 64,num_workers=2,pin_memory=True)
    logger = WandbLogger()
    trainer = pl.Trainer(
        logger,
        # progress_bar_refresh_rate=500,  # disable progress bar
        max_steps=steps,
        checkpoint_callback=False,
        callbacks=[
            callbacks.EarlyStop(),
            callbacks.Traversal(2000),
        ],
        gpus=1,)
    trainer.fit(pl_model, dl)
    wandb.join()
