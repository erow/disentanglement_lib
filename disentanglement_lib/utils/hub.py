import os

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
