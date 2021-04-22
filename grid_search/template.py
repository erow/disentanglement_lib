import os
import disentanglement_lib.utils.hyperparams as h
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
parser.add_argument('e', type=int, default=0)
parser.add_argument('--skip', type=int, default=0)
args = parser.parse_args()

seed = h.sweep('train.random_seed', h.discrete(range(args.s, args.e)))
model = h.sweep('train.model', h.discrete(['@vae', '@beta_tc_vae']))
model = model + [{'train.model': '@deft', 'deft.betas': "'[20, 2]'", 'model.stage_steps': 3000},
                 {'train.model': '@cascade_vae_c', 'model.stage_steps': 2000},
                 {'train.model': '@annealed_vae', 'annealed_vae.c_max': 10,
                  'annealed_vae.iteration_threshold': 7000,
                  'annealed_vae.gamma': 100}
                 ]
runs = h.product([seed, model])

general_config = {
    'train.eval_numbers': 1,
    'train.training_steps': 10000,
    'model.num_latent': 5
}
metrics = " --metrics dci factor_vae_metric"
print(len(runs))
for i, run_args in enumerate(runs):
    if i < args.skip: continue
    run_args.update(general_config)
    args_str = " ".join([f"--{k}={v}" for k, v in run_args.items()])
    args_str += metrics
    print(args_str, f"{100 * i // len(runs)}%")

    ret = os.system("python experiments/Correlation.py " + args_str)
    # if ret != 0:
    #     exit(ret)
