#!/usr/bin/env python
import disentanglement_lib.utils.hyperparams as h
import os 
import argparse

os.environ['WANDB_ENTITY']='dlib'
os.environ['WANDB_TAG']='hyperparameter'

training_steps = int(2e5)
program = "python exps/decrement.py"

seeds = h.sweep("model.seed",h.categorical(list(range(10))))

datasets = h.sweep("configs", ["disentanglement_lib/config/data/dsprites.gin","disentanglement_lib/config/data/shapes3d.gin"])


model_setting1 = h.sweep("decrement.betas", h.discrete(["[1.0,10.0]", "[1.0,10.0,40.0]","[1.0,10.0,20.0,40.0,80.0]"]))
model_setting2 = h.sweep("decrement.scale", h.discrete([1.0,0.5,0.3]))

all_experiemts = h.product([model_setting1,model_setting2,seeds, datasets])

parser = argparse.ArgumentParser()
parser.add_argument('-s','--start',default=0,type=int)
parser.add_argument('-e','--end',default=None,type=int)
parser.add_argument('--extra_args', default="", type=str)

program_args = parser.parse_args()

for i,args in enumerate(all_experiemts):
    if i<program_args.start: continue
    if program_args.end and i>= program_args.end: break
    
    args = " ".join(map(lambda x:f"--{x[0]}={x[1]}",args.items()))
    cmd = f"{program} {args} --max_steps {training_steps} " + program_args.extra_args
    # print("Run: ", cmd)
    ret = os.system(cmd)
    if ret!=0:
        print('error! Stop at ', i, ret)
        break