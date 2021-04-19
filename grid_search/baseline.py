import os
import disentanglement_lib.utils.hyperparams as h
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
parser.add_argument('e', type=int, default=0)
args = parser.parse_args()

seed = h.sweep('train.random_seed', h.discrete(range(args.s, args.e)))
model = h.sweep('train.model', h.discrete(['@vae', '@beta_tc_vae', '@cascade_vae_c']))
dataset = h.sweep('dataset', h.discrete(['dsprites_full', 'smallnorb', 'dsprites_noshape', 'color_dsprites']))
runs = h.product([seed, model, dataset])

general_config = {
}

print(len(runs))
for i, args in enumerate(runs):
    args.update(general_config)
    args_str = " ".join([f"--{k}={v}" for k, v in args.items()])
    print(args_str, f"{100 * i // len(runs)}%")

    ret = os.system("dlib_run.py " + args_str)
    if ret != 0:
        exit(ret)
