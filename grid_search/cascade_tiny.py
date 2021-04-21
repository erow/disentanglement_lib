import os
import disentanglement_lib.utils.hyperparams as h
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
parser.add_argument('e', type=int, default=0)
parser.add_argument('--skip', type=int, default=0)
args = parser.parse_args()

seed = h.sweep('train.random_seed', h.discrete(range(args.s, args.e)))
a = h.sweep('model.alpha', h.discrete([1, 2]))
lam = h.sweep('model.lam', h.discrete([1e-3, 1]))
model = h.sweep('train.model', h.discrete(['@cascade_vae_c_reg', '@cascade_vae_c_reg1']))

runs = h.product([seed, a, lam, model])
general_config = {
    "dataset": "dsprites_tiny"
}

print(len(runs))
for i, run_args in enumerate(runs):
    if i < args.skip: continue
    run_args.update(general_config)
    args_str = " ".join([f"--{k}={v}" for k, v in run_args.items()])
    print(args_str, f"{100 * i // len(runs)}%")

    ret = os.system("dlib_run.py " + args_str)
    if ret != 0:
        exit(ret)
