import os
import disentanglement_lib.utils.hyperparams as h
import argparse

os.environ['WANDB_TAGS'] = 'baseline'
parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
parser.add_argument('e', type=int, default=0)
parser.add_argument('--skip', type=int, default=0)
args = parser.parse_args()

seed = h.sweep('train.random_seed', h.discrete(range(args.s, args.e)))
model = h.sweep('model.regularizers', h.discrete(['[@vae]', '[@beta_tc_vae]', '[@cascade_vae_c]']))
dataset = h.sweep('model.dataset',
                  h.discrete([f"\"'{ds}'\"" for ds in
                              ["dsprites_full", "color_dsprites", "smallnorb", "cars3d"]]))

runs = h.product([seed, model, dataset])

general_config = {

}
metrics = " --metrics dci factor_vae_metric"
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
