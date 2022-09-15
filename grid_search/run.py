#!/usr/bin/env python
import argparse
import disentanglement_lib.utils.hyperparams as h
import os 
import logging
training_steps = int(3e5)
program = "dlib_run"

seeds = h.sweep("model.seed",h.categorical(list(range(10,15))))

datasets = h.sweep("configs", ["disentanglement_lib/config/data/shapes3d.gin"])

model_name = h.fixed("model.regularizers", "'[@vae()]'")
betas = h.sweep("vae.beta", h.discrete([1., 6. ]))
config_beta_vae = h.zipit([model_name, betas])

model_name = h.fixed("model.regularizers", "'[@beta_tc_vae()]'")
betas = h.sweep("beta_tc_vae.beta", h.discrete([ 12.]))
config_beta_tc_vae = h.zipit([model_name, betas])

# model_name = h.fixed("model.regularizers", "'[@control()]'")
# model_setting1 = [{"control.training_steps":training_steps}]
# model_setting2 = h.sweep("control.C", h.discrete([13.5, 16.]))
# config_control_vae = h.zipit([model_name, model_setting1,model_setting2])

all_models = h.chainit([config_beta_tc_vae,config_beta_vae])

all_experiemts = h.product([seeds, datasets, all_models])

parser = argparse.ArgumentParser()
parser.add_argument('-s','--start',default=0,type=int)
parser.add_argument('-e','--end',default=None,type=int)
parser.add_argument('--extra_args', default="", type=str)

program_args = parser.parse_args()
logging.basicConfig(filename='log.txt',filemode='a',level=logging.INFO,
)
logging.getLogger().addHandler(logging.StreamHandler())
for i,args in enumerate(all_experiemts):
    if i<program_args.start: continue
    if program_args.end and i>= program_args.end: break
    
    args = " ".join(map(lambda x:f"--{x[0]}={x[1]}",args.items()))
    cmd = f"{program} {args} --max_steps {training_steps} --output_dir outputs/saved_results"
    print("Run: ", cmd)
    ret = os.system(cmd)
    logging.info(f"[{__file__}:{i+program_args.start}-{program_args.end}] {cmd} -> {ret}")
    if ret!=0:
        print('error! Stop at ', i)
        break
