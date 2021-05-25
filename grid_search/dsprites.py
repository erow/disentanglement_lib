import os
import disentanglement_lib.utils.hyperparams as h
import argparse

os.environ['WANDB_PROJECT'] = 'disentanglement_lib-bin'
os.environ['WANDB_TAGS'] = 'MI_reg'

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
parser.add_argument('e', type=int, default=0)
parser.add_argument('--skip', type=int, default=0)
args = parser.parse_args()

seed = h.sweep('seed', h.discrete(range(args.s, args.e)))
a = h.sweep('model.alpha', h.discrete([0.1, 0.2, 0.5, 0.8]))
lam = h.sweep('model.lam', h.discrete([1]))

model = [
    {'train.model': '@vae'},
    # {'train.model': '@cascade_vae_c'},
    # {'train.model': '@beta_tc_vae'},
]

runs = h.product([seed, model, lam, a])
general_config = {
    "train.dataset": "\"'dsprites_tiny'\"",
    "evaluate.dataset": "\"'dsprites_tiny'\"",
    'train.eval_numbers': 3,
    'train.training_steps': 6000,
    'model.num_latent': 5,
    "vae.beta": 1,
}

print(len(runs))
for i, run_args in enumerate(runs):
    if i < args.skip: continue
    run_args.update(general_config)
    args_str = " ".join([f"--{k}={v}" for k, v in run_args.items()])
    print(args_str, f"{100 * i // len(runs)}%")

    ret = os.system("dlib_run " + args_str)
    if ret != 0:
        exit(ret)
