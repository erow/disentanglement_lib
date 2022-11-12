#!/usr/bin/env python
import disentanglement_lib.utils.hyperparams as h
import os 
import argparse
import logging
logging.basicConfig(filename='log.txt',filemode='a',level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

os.environ['WANDB_ENTITY']='dlib'
os.environ['WANDB_TAGS']='stability'

training_steps = int(2e5)
program = "python exps/decrement.py"

seeds = h.sweep("model.seed",h.categorical(list(range(5))))

datasets = h.sweep("configs", ["disentanglement_lib/config/data/dsprites.gin","disentanglement_lib/config/data/shapes3d.gin"])

model_setting1 = h.sweep("decrement.betas", h.discrete(["[1.0,10.0,20,30,40,50,60,70,80]"]))
model_setting2 = h.sweep("decrement.scale", h.discrete([1.0]))

all_experiemts = h.product([seeds, datasets, model_setting1,model_setting2[:1]])

parser = argparse.ArgumentParser()
parser.add_argument('-s','--start',default=0,type=int)
parser.add_argument('-e','--end',default=None,type=int)
parser.add_argument('--print', default=False, action='store_true')
parser.add_argument('--extra_args', default="", type=str)

program_args = parser.parse_args()
experiments = all_experiemts[program_args.start:program_args.end]
for i,args in enumerate(experiments):
    
    args = " ".join(map(lambda x:f"--{x[0]}={x[1]}",args.items()))
    cmd = f"{program} {args} --max_steps {training_steps} --output_dir=outputs/decrement/h{i}" + program_args.extra_args

    if program_args.print:
        print(f"[{__file__}:{i+program_args.start}-{program_args.end}] {cmd} ")
        continue
    ret = os.system(cmd)
    logging.info(f"[{__file__}:{i+program_args.start}-{program_args.end}] {cmd} -> {ret}")
    if ret!=0:
        print('error! Stop at ', i + program_args.start)
        break
        