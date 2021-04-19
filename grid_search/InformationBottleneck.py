import os
import disentanglement_lib.utils.hyperparams as h
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
parser.add_argument('e', type=int, default=0)
args = parser.parse_args()

seed = h.sweep('train.random_seed', h.discrete(range(args.s, args.e)))
anneal_model = h.sweep('train.model', h.discrete(['@anneal', '@annealed_vae']))
anneal_model = h.zipit([[{'train.training_steps': 70000,
                          'annealed_vae.iteration_threshold': 70000}], anneal_model])

var_model = h.zipit([[{'train.model': '@vae',
                       'train.training_steps': 20000,
                       'annealed_vae.iteration_threshold': 20000}],
                     h.sweep('model.num_latent', h.discrete([1, 2, 4, 10]))])

model = h.chainit([anneal_model, var_model])
dataset = [{'dataset.name': "\\'dsprites_full\\'", 'train.training_steps': 100000},
           {'dataset.name': "\\'smallnorb\\'",
            'train.training_steps': 50000,
            'annealed_vae.iteration_threshold': 50000}]

runs = h.product([seed, dataset, model])
general_config = {
}

print(len(runs))
for i, args in enumerate(runs):
    args.update(general_config)
    args_str = " ".join([f"--{k}={v}" for k, v in args.items()])
    print(args_str, f"{100 * i // len(runs)}%")

    ret = os.system("python InformationBottleneck.py " + args_str)
    if ret != 0:
        exit(ret)
