import os
import disentanglement_lib.utils.hyperparams as h
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
args = parser.parse_args()

seed = h.sweep('seed', h.discrete(range(args.s, args.s + 2)))
a = h.sweep('a', h.discrete([2]))
lam = h.sweep('lam', h.discrete([0, 1e-3, 1]))

model = [
    {'model': 'cascade_vae_c'},
    {'model': 'beta_tc_vae'},
    {'model': 'beta_vae'}
]

runs = h.product([seed, a, lam, model])
general_config = {
    "dataset": "dsprites_tiny"
}

print(len(runs))
for i, args in enumerate(runs):
    args.update(general_config)
    args_str = " ".join([f"--{k}={v}" for k, v in args.items()])
    print(args_str, f"{100 * i // len(runs)}%")

    ret = os.system("dlib_run.py " + args_str)
    if ret != 0:
        exit(ret)
