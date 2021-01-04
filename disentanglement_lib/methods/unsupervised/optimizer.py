import gin.torch
import torch


@gin.configurable("AdamOpt")
def AdamOpt(parameters, args=gin.REQUIRED):
    return torch.optim.Adam(parameters, **args)


@gin.configurable("SGDOpt")
def AdamOpt(parameters, args=gin.REQUIRED):
    return torch.optim.SGD(parameters, **args)
