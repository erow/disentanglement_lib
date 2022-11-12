# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of commonly used architectures and reconstruction losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchvision.models.resnet import BasicBlock
import gin

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer('shape', torch.LongTensor(shape))

    def forward(self, x):
        return x.view(*self.shape)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x  

@gin.configurable("fc_encoder", allowlist=[])
class fc_encoder(nn.Module):
    """Fully connected encoder used in beta-VAE paper for the dSprites data.
    Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl).

    Args:
      num_latent: Number of latent variables to output.
    Returns:
      means: Output tensor of shape (batch_size, num_latent) with latent variable
        means.
      log_var: Output tensor of shape (batch_size, num_latent) with latent
        variable log variances.
    """

    def __init__(self, input_shape, num_latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape).item(), 1200), nn.ReLU(),
            nn.Linear(1200, 1200), nn.ReLU(),
            nn.Linear(1200, num_latent * 2)
        )
        self.K = num_latent
        self.num_latent = num_latent
        self.input_shape = input_shape

    def forward(self, input_tensor):
        x = self.net(input_tensor)
        means, log_var = torch.split(x, [self.num_latent] * 2, 1)
        return means, log_var


@gin.configurable("conv_encoder", allowlist=[])
class conv_encoder(nn.Module):
    """Convolutional encoder used in beta-VAE paper for the chairs data.

    Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Args:
      input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
        build encoder on.
      num_latent: Number of latent variables to output.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      means: Output tensor of shape (batch_size, num_latent) with latent variable
        means.
      log_var: Output tensor of shape (batch_size, num_latent) with latent
        variable log variances.
    """

    def __init__(self, input_shape, num_latent, base_channel=32):
        super().__init__()
        self.num_latent = num_latent
        self.input_shape = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], base_channel, (4, 4), stride=2, padding=1), nn.ReLU(True),  # 32x32
            nn.Conv2d(base_channel, base_channel, (4, 4), stride=2, padding=1), nn.ReLU(True),    # 16
            nn.Conv2d(base_channel, base_channel * 2, (4, 4), stride=2, padding=1), nn.ReLU(True),# 8
            nn.Conv2d(base_channel * 2, base_channel * 2, (4, 4), stride=2, padding=1), nn.ReLU(True),  # 4
            nn.Conv2d(base_channel * 2, base_channel * 8, (4, 4)), nn.ReLU(True),           # 1x1
            nn.Flatten(),
            nn.Linear(1 * 1 * base_channel * 8, num_latent * 2)
        )
        self.K = num_latent

    def forward(self, input_tensor):
        x = self.net(input_tensor)
        means, log_var = torch.split(x, [self.num_latent] * 2, 1)
        return means, log_var


@gin.configurable("fc_decoder", allowlist=[])
class fc_decoder(nn.Module):
    def __init__(self, num_latent, output_shape):
        super().__init__()
        self.num_latent = num_latent
        self.output_shape = output_shape
        self.net = nn.Sequential(
            nn.Linear(num_latent, 1200), nn.ReLU(),
            nn.Linear(1200, 1200), nn.ReLU(),
            nn.Linear(1200, 1200), nn.ReLU(),
            nn.Linear(1200, np.prod(output_shape).item())
        )

    def forward(self, latent_tensor):
        x = self.net(latent_tensor)
        return torch.reshape(x, shape=[-1] + self.output_shape)



@gin.configurable("deconv_decoder", allowlist=[])
class deconv_decoder(nn.Module):
    """Convolutional decoder used in beta-VAE paper for the chairs data.

    Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Args:
      latent_tensor: Input tensor of shape (batch_size,) to connect decoder to.
      output_shape: Shape of the data.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      Output tensor of shape (batch_size, 64, 64, num_channels) with the [0,1]
        pixel intensities.
    """

    def __init__(self, num_latent, output_shape):
        super().__init__()
        self.num_latent = num_latent
        self.output_shape = output_shape
        self.net = nn.Sequential(
            nn.Linear(num_latent, 256), nn.ReLU(True),
            View([-1, 256, 1, 1]),
            nn.ConvTranspose2d(256, 64, 4), nn.ReLU(True),  # 4
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(True),  # 8
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(True),  # 16
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(True),  # 32
            nn.ConvTranspose2d(32, output_shape[0], 4, stride=2, padding=1)  # 64
        )

    def forward(self, latent_tensor):
        x = self.net(latent_tensor)
        return torch.reshape(x, shape=[-1] + self.output_shape)


@gin.configurable("fc_discriminator", allowlist=[])
class fc_discriminator(nn.Module):
    """Fully connected discriminator used in FactorVAE paper for all datasets.

    Based on Appendix A page 11 "Disentangling by Factorizing"
    (https://arxiv.org/pdf/1802.05983.pdf)

    Args:
      input_tensor: Input tensor of shape (None, num_latents) to build
        discriminator on.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      logits: Output tensor of shape (batch_size, 2) with logits from
        discriminator.
      probs: Output tensor of shape (batch_size, 2) with probabilities from
        discriminator.
    """

    def __init__(self, num_latent):
        super().__init__()
        self.num_latent = num_latent
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_latent, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 2)
        )

    def forward(self, input_tensor):
        logits = self.net(input_tensor)
        probs = torch.softmax(logits, 1)
        clipped = torch.clamp(probs, 1e-6, 1 - 1e-6)
        return logits, clipped


@gin.configurable("test_encoder", allowlist=[])
class test_encoder(nn.Module):
    """Fully connected encoder used in beta-VAE paper for the dSprites data.
    Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl).

    Args:
      num_latent: Number of latent variables to output.
    Returns:
      means: Output tensor of shape (batch_size, num_latent) with latent variable
        means.
      log_var: Output tensor of shape (batch_size, num_latent) with latent
        variable log variances.
    """

    def __init__(self, input_shape, num_latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape).item(), 1200), nn.ReLU(),
            nn.Linear(1200, num_latent * 2)
        )
        self.num_latent = num_latent

    def forward(self, input_tensor):
        x = self.net(input_tensor)
        means, log_var = torch.split(x, [self.num_latent] * 2, 1)
        return means, log_var


@gin.configurable("test_decoder", allowlist=[])
class test_decoder(nn.Module):
    """Fully connected encoder used in beta-VAE paper for the dSprites data.

    Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Args:
      latent_tensor: Input tensor to connect decoder to.
      output_shape: Shape of the data.
      is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
      Output tensor of shape (None, 64, 64, num_channels) with the [0,1] pixel
      intensities.
    """

    def __init__(self, num_latent, output_shape):
        super().__init__()
        self.num_latent = num_latent
        self.output_shape = output_shape
        self.net = nn.Sequential(
            nn.Linear(num_latent, 1200), nn.ReLU(),
            nn.Linear(1200, np.prod(output_shape).item())
        )

    def forward(self, latent_tensor):
        x = self.net(latent_tensor)
        return torch.reshape(x, shape=[-1] + self.output_shape)


class lite_decoder(nn.Module):
    def __init__(self, num_latent, output_shape):
        super().__init__()
        self.num_latent = num_latent
        self.output_shape = output_shape
        self.net = nn.Sequential(
            nn.Linear(num_latent, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            View([-1, 16, 4, 4]),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1), nn.ReLU(),  # 8
            nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),  # 16
            nn.ConvTranspose2d(32, 4, 4, stride=2, padding=1), nn.ReLU(),  # 32
            nn.ConvTranspose2d(4, output_shape[0], 4, stride=2, padding=1)  # 64
        )

    def forward(self, latent_tensor):
        x = self.net(latent_tensor)
        return torch.reshape(x, shape=[-1] + self.output_shape)

@gin.configurable("deep_decoder", allowlist=[])
class DeepConvDecoder(nn.Module):
    def __init__(self, num_latent, input_shape,
                 depths=[3, 9, 3, 3], dims=[384, 192, 96, 48], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()
        self.input_shape = input_shape
        self.num_latent = num_latent
        self.map_size=input_shape[-1]//16
        
        self.upsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        out_chans = input_shape[0]
        
        self.head = nn.Linear(num_latent,dims[0])
        self.norm = nn.LayerNorm(num_latent, eps=1e-6) # final norm layer
        # self.tail = nn.Conv2d(out_chans,out_chans,kernel_size=1)
        
        dims.append(out_chans)
        for i in range(4):
            upsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.ConvTranspose2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.upsample_layers.append(upsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.stages[i](x)
            x = self.upsample_layers[i](x)
        return x

    def forward(self, x):
        x = self.norm(x)
        x = self.head(x)[:,:,None,None]
        x = x.repeat(1,1,self.map_size,self.map_size) # global repeating, (N, C) -> (N, C, H, W)
        x = self.forward_features(x)
        return x



@gin.configurable("deep_encoder", allowlist=[])
class DeepConvEncoder(nn.Module):
    def __init__(self, input_shape, num_latent, 
                 depths=[3, 3, 9, 3], dims=[48, 96, 192, 384], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()
        self.input_shape = input_shape
        self.num_latent = num_latent
        
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        in_chans = input_shape[0]
        
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=1, stride=1), # do not down-sampling
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_latent*2)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x.chunk(2,1)


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.ReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
    
@gin.configurable("frac_encoder", allowlist=[])
class FracEncoder(nn.Module):
    def __init__(self,
                input_shape,
                num_latent, G=5):
        """Fractional encoder with multi sub-encoders. https://link.springer.com/article/10.1007/s10994-022-06134-7

        Args:
            input_shape (list): The shape of inputs.
            num_latent (int): The number of latent variables.
            G (int, optional): The number of sub-encoders. Defaults to 5.
        """
        super().__init__()
        self.G=G
        self.num_latent = num_latent
        assert num_latent % G==0
        self.K=num_latent//G
        self.sub_encoders = nn.ModuleList(
            [conv_encoder(input_shape, self.K,8) for _ in range(G)])
        self.projs = nn.ModuleList([
            Projection(self.K) for _ in range(G)
        ])
        self.stage = 0
        
    def set_stage(self, i):
        """Stop gradients for [0,i) sub-encoders.

        Args:
            i (int): current stage.
        """
        if i> self.G:
            i = self.G
        self.stage = i
        for j in range(i):
            self.sub_encoders[j].requires_grad_(False)


    def forward(self, x):
        mus, logvars = [], []

        for i in range(self.G):
            f = self.sub_encoders[i]
            if i<self.stage:
                mu, logvar = f(x)
                mu, logvar = self.projs[i](mu.data,logvar.data)
            elif i ==self.stage:
                mu, logvar = f(x)
            else:
                mu = torch.zeros_like(mu)
                logvar = torch.zeros_like(logvar)
            mus.append(mu)
            logvars.append(logvar)
        
        mu = torch.cat(mus,1)
        logvar = torch.cat(logvars,1)
        return mu, logvar

@gin.configurable("discriminator", allowlist=["discriminator_fn"])
def make_discriminator(num_latent,
                       discriminator_fn=fc_discriminator):
    """Gin wrapper to create and apply a discriminator configurable with gin.

    This is a separate function so that several different models (such as
    FactorVAE) can potentially call this function while the gin binding always
    stays 'discriminator.(...)'. This makes it easier to configure models and
    parse the results files.

    Args:
        num_latent: Number of the latent variables.
      discriminator_fn: Function that that takes the arguments
      (input_tensor, is_training) and returns tuple of (logits, clipped_probs).

    Returns:
      Tuple of (logits, clipped_probs) tensors.
    """
    # logits, probs = discriminator_fn(num_latent)
    # clipped = torch.clamp(probs, 1e-6, 1 - 1e-6)
    return discriminator_fn(num_latent)

class Projection(nn.Module):
    def __init__(self,num_latent) -> None:
        super().__init__()
        self.num_latent = num_latent  
        self.W1 = nn.Parameter(torch.zeros(1,num_latent))             
        self.W2 = nn.Parameter(torch.zeros(1,num_latent))

    def forward(self, mu, logvar):
        z_mean1 = mu * self.W1.exp()
        z_logvar1 = logvar + self.W2
        return z_mean1, z_logvar1
    
    def extra_repr(self):
        return "W1: " + str((self.W1.exp()).data.cpu().numpy().round(2))+",\n"+\
        "W2: " + str(self.W2.exp().data.cpu().numpy().round(2))
    
    
  

@gin.configurable("residual_encoder")
class ResidualEncoder(nn.Module):
    """ Reference:
    @inproceedings{
        dittadi2021transfer,
        title={On the Transfer of Disentangled Representations in Realistic Settings},
        author={Andrea Dittadi and Frederik Tr{\"a}uble and Francesco Locatello and Manuel W{\"u}thrich and Vaibhav Agrawal and Ole Winther and Stefan Bauer and Bernhard Sch{\"o}lkopf},
        booktitle={International Conference on Learning Representations},
        year={2021},
    }
    """
    def __init__(self,input_shape, num_latent) -> None:
        super().__init__()
        K,H,W = input_shape
        assert H==128
        self.conv1 = nn.Conv2d(K,64,5,stride=2,padding=2)
        self.ac1 = nn.LeakyReLU(0.02)
        
        self.conv2=nn.Sequential(
                BasicBlock(64,64),
                BasicBlock(64,64),
                nn.Conv2d(64,128,1),
                nn.AvgPool2d(2),
                BasicBlock(128,128),
                BasicBlock(128,128),
                nn.AvgPool2d(2),
                BasicBlock(128,128),
                BasicBlock(128,128),
                nn.Conv2d(128,256,1),
                nn.AvgPool2d(2),
                BasicBlock(256,256),
                BasicBlock(256,256),
                nn.AvgPool2d(2),
                BasicBlock(256,256),
                BasicBlock(256,256),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LeakyReLU(0.02),
            nn.Linear(4096,512),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(512),
            nn.Linear(512,num_latent*2)
        )
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return torch.chunk(x,2,dim=1)

@gin.configurable("residual_decoder")
class ResidualDecoder(nn.Module):
    """ Reference:
    @inproceedings{
        dittadi2021transfer,
        title={On the Transfer of Disentangled Representations in Realistic Settings},
        author={Andrea Dittadi and Frederik Tr{\"a}uble and Francesco Locatello and Manuel W{\"u}thrich and Vaibhav Agrawal and Ole Winther and Stefan Bauer and Bernhard Sch{\"o}lkopf},
        booktitle={International Conference on Learning Representations},
        year={2021},
    }
    """
    def __init__(self, num_latent, output_shape) -> None:
        super().__init__()
        K,H,W = output_shape
        assert H==128
        self.conv1 = nn.Conv2d(64,K,5,padding=2)
        self.ac1 = nn.LeakyReLU(0.02)
        
        self.conv2=nn.Sequential(
                BasicBlock(256,256),
                BasicBlock(256,256),
                nn.UpsamplingBilinear2d(8),
                BasicBlock(256,256),
                BasicBlock(256,256),
                nn.Conv2d(256,128,1),
                 nn.UpsamplingBilinear2d(16),
                BasicBlock(128,128),
                BasicBlock(128,128),
                 nn.UpsamplingBilinear2d(32),
                BasicBlock(128,128),
                BasicBlock(128,128),
                nn.Conv2d(128,64,1),
                 nn.UpsamplingBilinear2d(64),
                BasicBlock(64,64),
                BasicBlock(64,64),
                 nn.UpsamplingBilinear2d(128),
        )
        self.fc = nn.Sequential(
            nn.Linear(num_latent,512),
            nn.LeakyReLU(0.02),
            nn.Linear(512,4096),
            View([-1, 256,4,4]),
        )
    
    def forward(self,x):
        x = self.fc(x)
        x = self.conv2(x)
        x = self.ac1(x)
        x = self.conv1(x)
        return x