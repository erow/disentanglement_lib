import os
import disentanglement_lib.utils.hyperparams as h
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
parser.add_argument('e', type=int, default=0)
parser.add_argument('--skip', type=int, default=0)
# parser.add_argument('--no_error', default=True)
args = parser.parse_args()

seed = h.sweep('train.random_seed', h.discrete(range(args.s, args.e)))
a = h.sweep('model.alpha', h.discrete([2]))
shared = h.sweep('model.shared', h.discrete([True, False]))
lam = h.sweep('model.lam', h.discrete([1e-2, 1]))
model = h.sweep('train.model', h.discrete(['@beta_tc_vae', '@vae', '@cascade_vae_c']))

runs = h.product([seed, shared, a, lam, model])
general_config = {
    "dataset": "dsprites_full"
}

print(len(runs))
for i, run_args in enumerate(runs):
    if i < args.skip: continue
    run_args.update(general_config)
    args_str = " ".join([f"--{k}={v}" for k, v in run_args.items()])
    print(args_str, f"{100 * i // len(runs)}%")

    # ret = os.system("dlib_run.py " + args_str)
    # if ret != 0:
    #     exit(ret)
