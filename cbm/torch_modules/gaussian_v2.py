from typing import List, Tuple, Union
import numpy as np
import abc
import math
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
from torch.distributions import Normal

from cbm.torch_modules.mlp import build_mlp_v2
import cbm.torch_modules.utils as ptu


# A more simple implementation for torch.jit.script. 
# However, it is more difficult to be configured via json file.
# currently does not support to compute log_prob
class Gaussian(metaclass=abc.ABCMeta):
    def __init__(
        self, 
        squashed: bool = False,
        mean_scale: float = 1.0,
        std_scale: float = 1.0,
        min_std: float = 1e-4,
    ) -> None:
        self.squashed = squashed
        self.mean_scale = mean_scale
        self.std_scale = std_scale
        assert min_std >= 0
        self.min_std = min_std

    @abc.abstractmethod
    def get_mean_std(self, x):
        pass

    def forward(self, x):
        mean, std = self.get_mean_std(x)
        mean = mean * self.mean_scale
        std = std * self.std_scale + self.min_std
        out_pred = mean + torch.randn_like(mean) * std
        if self.squashed:
            out_pred - torch.tanh(out_pred)
        #NOTE: the mean is before being squshaed
        return out_pred, mean, std


class FixedGaussian(Gaussian, nn.Module):
    def __init__(
        self, 
        out_pred_size: int,  
        std: float,
        squashed: bool = False,
        mean_scale: float = 1.0,
        std_scale: float = 1.0,
        min_std: float = 1e-3,
    ) -> None:
        Gaussian.__init__(self, squashed, mean_scale, std_scale, min_std)
        nn.Module.__init__(self)
        self.out_pred_size = out_pred_size
        self.std = std

    def get_mean_std(self, x):
        mean = torch.zeros(x.size(0), self.out_pred_size, device=x.device)
        std = torch.ones(x.size(0), self.out_pred_size, device=x.device).mul_(self.std)
        return mean, std


# the following is much faster than the "simple" one......
class MeanLogstdGaussian(Gaussian, nn.Module):
    def __init__(
        self, 
        feature_size: int, 
        out_pred_size: int,
        hidden_layers: List[int], 
        logstd_extra_bias: float = 0,
        squashed: bool = False,
        mean_scale: float = 1.0,
        std_scale: float = 1.0,
        min_std: float = 1e-4,
        **mlp_kwargs
    ):
        Gaussian.__init__(self, squashed, mean_scale, std_scale, min_std)
        nn.Module.__init__(self)
        layer_size = [feature_size] + hidden_layers + [out_pred_size*2]
        self.mlp_net, _ = build_mlp_v2(layer_size, None, **mlp_kwargs)
        self.feature_size = feature_size
        self.out_pred_size = out_pred_size
        self.logstd_extra_bias = logstd_extra_bias

    def get_mean_std(self, x):
        output=self.mlp_net(x)
        mean, log_std = torch.chunk(output,2,-1)
        log_std = log_std + self.logstd_extra_bias
        return mean, torch.exp(log_std)
            
class MeanSoftplusStdGaussian(Gaussian, nn.Module):
    def __init__(
        self, 
        feature_size: int, 
        out_pred_size: int, 
        hidden_layers: List[int], 
        std_extra_bias: float = 1.0,
        squashed: bool = False,
        mean_scale: float = 1.0,
        std_scale: float = 1.0,
        min_std: float = 1e-4,
        **mlp_kwargs
    ):
        Gaussian.__init__(self, squashed, mean_scale, std_scale, min_std)
        nn.Module.__init__(self)
        layer_size = [feature_size] + hidden_layers + [out_pred_size*2]
        self.mlp_net, _ = build_mlp_v2(layer_size, None, **mlp_kwargs)
        self.feature_size = feature_size
        self.out_pred_size = out_pred_size
        self.std_extra_bias = std_extra_bias
        self.min_std = min_std


    def get_mean_std(self, x):
        output=self.mlp_net(x)
        mean, std = torch.chunk(output,2,-1)
        std = F.softplus(std+self.std_extra_bias)
        return mean, std
