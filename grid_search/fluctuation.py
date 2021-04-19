import os
import disentanglement_lib.utils.hyperparams as h
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
parser.add_argument('e', type=int, default=0)
parser.add_argument('--skip', type=int, default=0)
args = parser.parse_args()

seed = h.sweep('train.random_seed', h.discrete(range(args.s, args.e)))
model = h.sweep('train.model', h.discrete(['@annealed_vae', '@cascade_vae_c', '@vae', '@beta_tc_vae']))

dataset = h.sweep('dataset.name', h.discrete([f"\\'{i}\\'"
                                              for i in ['dsprites_full', 'scream_dsprites', 'smallnorb']]))

runs = h.product([seed, dataset, model])
general_config = {
}

print(len(runs))
for i, run_args in enumerate(runs):
    if i < args.skip: continue
    run_args.update(general_config)
    args_str = " ".join([f"--{k}={v}" for k, v in run_args.items()])
    print(args_str, f"{100 * i // len(runs)}%")

    ret = os.system("python fluctuation.py " + args_str)
    if ret != 0:
        exit(ret)
