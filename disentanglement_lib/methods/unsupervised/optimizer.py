import gin.torch
import torch


@gin.configurable("AdamOpt")
def AdamOpt(parameters, args=gin.REQUIRED):
    return torch.optim.Adam(parameters, **args)
