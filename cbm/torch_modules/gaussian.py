from typing import Tuple, Union
import numpy as np
import abc
import math
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
from torch.distributions import Normal

from cbm.torch_modules.mlp import MLP
import cbm.torch_modules.utils as ptu

EPS = 1e-6
LOG_STD_MAX = 2
LOG_STD_MIN = -10
def bound_logstd(
    logstd: torch.Tensor, 
    bound_mode: str
) -> torch.Tensor:
    if bound_mode == "clamp":
        logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)
    elif bound_mode == "tanh":
        scale = (LOG_STD_MAX-LOG_STD_MIN) / 2
        logstd = (torch.tanh(logstd)+1) * scale + LOG_STD_MIN
    elif bound_mode == "no":
        logstd = logstd
    else:
        raise NotImplementedError
    return logstd

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
    def get_mean_std(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        reparameterize: bool = True,
        return_log_prob: bool = False,
        return_mean_std: bool = False,
        return_dist: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        :param x: Observation
        :param deterministic: 
        :param return_log_prob: 
        :return: 
        """
        mean, std = self.get_mean_std(x)
        mean = mean * self.mean_scale
        std = std * self.std_scale + self.min_std
        dist = Normal(mean, std)

        if deterministic:
            out_pred = mean
        elif reparameterize:
            out_pred = dist.rsample()
        else:
            out_pred = dist.sample()
        if self.squashed:
            pretanh = out_pred
            out_pred = torch.tanh(out_pred)

        info = {}
        if return_log_prob:
            if self.squashed:
                log_prob = dist.log_prob(pretanh)
                log_prob -= 2*np.log(2)-F.softplus(2*pretanh)-F.softplus(-2*pretanh)
            else:
                log_prob = dist.log_prob(out_pred)
            info['log_prob'] = log_prob.sum(-1, keepdim=True)
        if return_mean_std:
            info['mean'] = mean
            info['std'] = std
        if return_dist:
            info['dist'] = dist
        return out_pred, info

    def log_prob(
        self, 
        x: torch.Tensor, 
        out_pred: torch.Tensor
    ) -> torch.Tensor:
        mean, std = self.get_mean_std(x)
        dist = self.dist_class(mean, std)
        log_prob = dist.log_prob(out_pred)
        return log_prob.sum(-1, keepdim=True)

class FixedGaussian(Gaussian, nn.Module):
    def __init__(
        self, 
        out_pred_size: int, 
        std: Union[int, float], 
        squashed: bool = False,
        mean_scale: float = 1.0,
        std_scale: float = 1.0,
        min_std: float = 1e-4,
    ) -> None:
        Gaussian.__init__(self, squashed, mean_scale, std_scale, min_std)
        nn.Module.__init__(self)
        self.out_pred_size = out_pred_size
        self.std = std
        self.log_std = np.log(std)

    def get_mean_std(self, x):
        mean = ptu.zeros(x.size(0), self.out_pred_size)
        std = ptu.ones(x.size(0), self.out_pred_size) * (self.std)
        return mean, std


class SimpleGaussian(Gaussian, MLP):
    def __init__(
        self, 
        feature_size: int, 
        out_pred_size: int, 
        bound_mode: str = 'tanh',
        module_name: str = 'simple_gaussian',
        squashed: bool = False,
        mean_scale: float = 1.0,
        std_scale: float = 1.0,
        min_std: float = 1e-4,
        **mlp_kwargs
    ) -> None:
        Gaussian.__init__(self, squashed, mean_scale, std_scale, min_std)
        MLP.__init__(
            self,
            feature_size,
            out_pred_size,
            module_name=module_name,
            **mlp_kwargs
        )
        self.feature_size = feature_size
        self.out_pred_size = out_pred_size
        self.bound_mode = bound_mode
        self.log_std = nn.Parameter(ptu.zeros(1,out_pred_size))

    def get_mean_std(self, x):
        mean=MLP.forward(self, x)
        log_std = bound_logstd(self.log_std, self.bound_mode)
        log_std = log_std.expand(mean.shape)
        return mean, torch.exp(log_std)

# the following is much faster than the "simple" one......
class MeanLogstdGaussian(Gaussian, MLP):
    def __init__(
        self, 
        feature_size: int , 
        out_pred_size: int , 
        bound_mode: str = 'tanh',
        logstd_extra_bias: float = 0,
        init_func_name: str = "orthogonal_",
        module_name: str = 'mean_logstd_gaussian',
        squashed: bool = False,
        mean_scale: float = 1.0,
        std_scale: float = 1.0,
        min_std: float = 1e-4,
        **mlp_kwargs
    ):
        Gaussian.__init__(self, squashed, mean_scale, std_scale, min_std)
        MLP.__init__(
            self,
            feature_size,
            out_pred_size*2,
            module_name=module_name,
            init_func_name=init_func_name,
            **mlp_kwargs
        )
        self.feature_size = feature_size
        self.out_pred_size = out_pred_size
        self.bound_mode = bound_mode
        self.logstd_extra_bias = logstd_extra_bias

    def get_mean_std(self, x):
        output=MLP.forward(self, x)
        mean, log_std = torch.chunk(output,2,-1)
        log_std = log_std + self.logstd_extra_bias
        log_std = bound_logstd(log_std, self.bound_mode)
        return mean, torch.exp(log_std)
            
class MeanSoftplusStdGaussian(Gaussian, MLP):
    def __init__(
        self, 
        feature_size: int, 
        out_pred_size: int, 
        std_extra_bias: float = 1.0,
        module_name: str = 'mean_softplus_gaussian',
        squashed: bool = False,
        mean_scale: float = 1.0,
        std_scale: float = 1.0,
        min_std: float = 1e-4,
        **mlp_kwargs
    ):
        Gaussian.__init__(self, squashed, mean_scale, std_scale, min_std)
        MLP.__init__(
            self,
            feature_size,
            out_pred_size*2,
            module_name=module_name,
            **mlp_kwargs
        )
        self.feature_size = feature_size
        self.out_pred_size = out_pred_size
        self.std_extra_bias = std_extra_bias
        self.min_std = min_std

    def get_mean_std(self, x):
        output=MLP.forward(self, x)
        mean, std = torch.chunk(output,2,-1)
        std = F.softplus(std+self.std_extra_bias)
        return mean, std
