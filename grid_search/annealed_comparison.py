import os
import disentanglement_lib.utils.hyperparams as h
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
parser.add_argument('e', type=int, default=0)
parser.add_argument('--skip', type=int, default=0)
args = parser.parse_args()

seed = h.sweep('train.random_seed', h.discrete(range(args.s, args.e)))
# model = [{'train.model':'@annealed'},
#          {'train.model':'@cascade','model.stage_steps':12000},
#          {'train.model':'@annealed_vae','annealed_vae.c_max':25, 'annealed_vae.iteration_threshold':50000,'annealed_vae.gamma':100},
#          {'train.model':'@vae'},
#          {'train.model':'@cascade_vae_c', 'model.stage_steps':6000}]
model = [{'train.model': '@annealed_vae', 'annealed_vae.c_max': 25, 'annealed_vae.iteration_threshold': 50000,
          'annealed_vae.gamma': 100},
         {'train.model': '@annealed_vae', 'annealed_vae.c_max': 15, 'annealed_vae.iteration_threshold': 50000,
          'annealed_vae.gamma': 100}]
runs = h.product([seed, model])
general_config = {
    # 'train.training_steps':1000,
    'train.eval_numbers': 10
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
