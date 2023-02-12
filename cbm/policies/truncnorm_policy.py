from typing import Tuple, Optional
from cbm.policies.base_policy import RandomPolicy
from cbm.torch_modules.mlp import MLP
from torch import distributions as pyd
import cbm.torch_modules.utils as ptu
from torch.distributions.utils import _standard_normal
from torch import nn
import torch
from contextlib import contextmanager


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(
            shape,
            dtype=self.loc.dtype,
            device=self.loc.device
        )
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

class TruncNormPolicy(nn.Module, RandomPolicy):
    def __init__( 
        self, 
        env, 
        deterministic: bool = False,
        noise_scale: float = 0.1,
        noise_clip: float = 0.3,
        pretanh_scale: float = 0.1,
        policy_name: str = 'truncnorm_policy',
        **mlp_kwargs
    ) -> None:
        nn.Module.__init__(self) 
        RandomPolicy.__init__(self, env, deterministic)

        self.module = MLP( 
            self._get_feature_size(), 
            self.action_shape[0],
            module_name=policy_name,
            **mlp_kwargs
        )
        self.noise_scale = noise_scale
        self.noise_clip = noise_clip
        self.pretanh_scale = pretanh_scale
        
    def _get_feature_size(self) -> int:
        return self.observation_shape[0]

    def _get_features(self, o: torch.Tensor) -> torch.Tensor:
        return o
        
    def action( 
        self, 
        obs: torch.Tensor, 
        deterministic: Optional[bool] = None,
        use_noise_clip: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        input_tensor = self._get_features(obs)
        mean = self.module(
            input_tensor, 
            **kwargs
        )
        mean = mean * self.pretanh_scale
        mean = torch.tanh(mean)
        std = ptu.ones_like(mean) * self.noise_scale

        if deterministic is None:
            deterministic = self._deterministic
        if deterministic:
            return mean, {}
        else:
            dist = TruncatedNormal(mean, std)
            if use_noise_clip:
                return dist.sample(self.noise_clip), {}
            else:
                return dist.sample(), {}

    def action_from_feature( 
        self, 
        input_tensor: torch.Tensor, 
        deterministic: Optional[bool] = None,
        use_noise_clip: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        mean = self.module(
            input_tensor, 
            **kwargs
        )
        mean = mean * self.pretanh_scale
        mean = torch.tanh(mean)
        std = ptu.ones_like(mean) * self.noise_scale

        if deterministic is None:
            deterministic = self._deterministic
        if deterministic:
            return mean, {}
        else:
            dist = TruncatedNormal(mean, std)
            if use_noise_clip:
                return dist.sample(self.noise_clip), {}
            else:
                return dist.sample(), {}

    def set_noise_scale(self, noise_scale: float):
        self.noise_scale = noise_scale

    @contextmanager
    def noise_scale_(self, noise_scale):
        original_noise_scale= self.noise_scale
        self.noise_scale = noise_scale
        yield
        self.noise_scale = original_noise_scale