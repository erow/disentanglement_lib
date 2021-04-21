import os
import disentanglement_lib.utils.hyperparams as h
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
parser.add_argument('e', type=int, default=0)
parser.add_argument('--skip', type=int, default=0)
args = parser.parse_args()

seed = h.sweep('train.random_seed', h.discrete(range(args.s, args.e)))
model = h.sweep('train.model', h.discrete(['@vae', '@beta_tc_vae', '@cascade_vae_c']))
model += [{'train.model': '@deft', 'model.stage_steps': 5000, 'deft.betas': '[60, 40, 1]'}]
model += [{'train.model': '@annealed_vae'}]

dataset = [
    {'dataset.name': "\"'chairs'\"",
     'train.training_steps': 30000,
     'model.stage_steps': 3000},
]

runs = h.product([seed, model, dataset])

general_config = {
    'train.eval_numbers': 1
}
metrics = " "
print(len(runs))
for i, run_args in enumerate(runs):
    if i < args.skip: continue
    run_args.update(general_config)
    args_str = " ".join([f"--{k}={v}" for k, v in run_args.items()])
    args_str += metrics
    print(args_str, f"{100 * i // len(runs)}%")
    # print(args_str)
    ret = os.system("dlib_run " + args_str)
    if ret != 0:
        exit(ret)
