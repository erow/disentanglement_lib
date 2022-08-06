#!/usr/bin/env python
import disentanglement_lib.utils.hyperparams as h
import os 
training_steps = int(1e6)
program = "dlib_run"

seeds = h.sweep("model.seed",h.categorical(list(range(10))))

datasets = h.sweep("configs", ["disentanglement_lib/config/data/dsprites.gin","disentanglement_lib/config/data/shapes3d.gin"])

model_name = h.fixed("model.regularizers", "'[@vae()]'")
betas = h.sweep("vae.beta", h.discrete([1., 6. , 10.]))
config_beta_vae = h.zipit([model_name, betas])

model_name = h.fixed("model.regularizers", "'[@beta_tc_vae()]'")
betas = h.sweep("beta_tc_vae.beta", h.discrete([6. , 12.]))
config_beta_tc_vae = h.zipit([model_name, betas])

model_name = h.fixed("model.regularizers", "'[@control()]'")
model_setting1 = [{"control.training_steps":training_steps}]
model_setting2 = h.sweep("control.C", h.discrete([13.5, 16.]))
config_control_vae = h.zipit([model_name, model_setting1,model_setting2])

all_models = h.chainit([config_beta_tc_vae,config_beta_vae,config_control_vae])

all_experiemts = h.product([all_models,seeds, datasets])
for i,args in enumerate(all_experiemts):
    args = " ".join(map(lambda x:f"--{x[0]}={x[1]}",args.items()))
    cmd = f"{program} {args} --max_steps {training_steps}"
    print("Run: ", cmd)
    ret = os.system(cmd)
    if ret!=-1:
        print('error! Stop at ', i)
        break