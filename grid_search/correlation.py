import os
import disentanglement_lib.utils.hyperparams as h
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
parser.add_argument('e', type=int, default=0)
parser.add_argument('--skip', type=int, default=0)
args = parser.parse_args()

seed = h.sweep('train.random_seed', h.discrete(range(args.s, args.e)))
model = h.sweep('train.model', h.discrete(['@beta_tc_vae']))
model += [{'train.model': '@deft', 'model.stage_steps': 3000, 'deft.betas': "'[20, 1]'"}]
model += [{'train.model': '@cascade_vae_c', 'model.stage_steps': 2000, }]
runs = h.product([seed, model])

general_config = {
    'dataset.name': "\"'correlation'\"",
    'train.training_steps': '10000',
    'model.num_latent': '5',
    'train.eval_numbers': 3
}
metrics = " --metrics dci factor_vae_metric"
print(len(runs))
for i, run_args in enumerate(runs):
    if i < args.skip: continue
    run_args.update(general_config)
    args_str = " ".join([f"--{k}={v}" for k, v in run_args.items()])
    args_str += metrics
    print(args_str, f"{100 * i // len(runs)}%")

    ret = os.system("dlib_run " + args_str)
    if ret != 0:
        exit(ret)
