import glob
import os
from xml.dom import NotFoundErr
import numpy as np
import torch
import gin

from disentanglement_lib.methods.unsupervised.train import PLModel


def get_model(model_file, model_fun=PLModel,
              device='cpu',):
    state = torch.load(model_file)['state_dict']
    model = model_fun()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def retrive_model(
        model_path,
        ckpt_step=None,
        model_fn=PLModel,        
    ):
    gin.parse_config_file(model_path+"/model.gin",True)
    ckpt = glob.glob(model_path+"/checkpoints/*.ckpt")
    steps = [int(i.split('step=')[1].split('.ckpt')[0]) for i in ckpt]
    
    if ckpt_step is None:
        idx = np.argmax(steps)
    elif ckpt_step in steps:
        for i,s in enumerate(steps):
            if s == ckpt_step:
                idx = i 
                break
    else:
        raise NotFoundErr()
    state_dict = torch.load(ckpt[idx])['state_dict']
    model = model_fn()
    model.load_state_dict(state_dict)
    model.eval()
    return model