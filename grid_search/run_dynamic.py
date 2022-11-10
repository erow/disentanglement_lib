
#!/usr/bin/env python
import argparse
import disentanglement_lib.utils.hyperparams as h
import os
import logging
os.environ['WANDB_PROJECT']='disentanglement_lib-bin'
os.environ['WANDB_ENTITY']="dlib"


seeds = h.sweep("model.seed",h.categorical(list(range(1,10))))


datasets = h.sweep("configs", ["disentanglement_lib/config/data/dsprites.gin","disentanglement_lib/config/data/shapes3d.gin"])

model_setting1 = [{
    'model.regularizers':"'[@dynamic()]'",
    'dynamic.C':18,
    'dynamic.K_p':0.04,
    'dynamic.K_i':0.004,
    'dynamic.total_steps':int(2e5),
    'dynamic.K':5000}]

all_experiemts = h.product([datasets, seeds, model_setting1])


training_steps = int(3e5)

program = "bin/dlib_run"
logging.basicConfig(filename='log.txt',filemode='a',level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

for i,args in enumerate(all_experiemts):

    args = " ".join(map(lambda x:f"--{x[0]}={x[1]}",args.items()))
    cmd = f"{program} {args} --max_steps {training_steps} --output_dir outputs/run"
    print(cmd)
    ret = os.system(cmd)
    logging.info(f"[{__file__}:{i}] {cmd} -> {ret}")
    if ret!=0:
        print('error! Stop at ', i )
        break
