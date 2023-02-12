from typing import Tuple, Optional

from torch.nn.modules.activation import Threshold
from cbm.policies.base_policy import RandomPolicy
from cbm.torch_modules.mlp import MLP
from torch import distributions as pyd
import cbm.torch_modules.utils as ptu
from torch.distributions.utils import _standard_normal
from torch import nn
import torch
from contextlib import contextmanager


class DiscreteAction(nn.Module):
    def __init__(self, action_size, k):
        super().__init__(self)
        self._k_head = nn.Parameter(ptu.rand(k, action_size)*2-1)
        self.action_size = action_size

    @property
    def k_head(self):
        return torch.tanh(self._k_head)

    def rsample(self, prob):
        ind = torch.multinomial(prob,1)
        sample_prob = torch.gather(prob,-1,ind)
        action = torch.gather(self.k_head, 0, ind.repeat(1,self.action_size))
        return action, sample_prob

    def compute_q(self, s, prob, qf):
        a, p = self.rsample(prob)
        q = qf(s, a).mean()
        
        

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

    def set_noise_scale(self, noise_scale: float):
        self.noise_scale = noise_scale

    @contextmanager
    def noise_scale_(self, noise_scale):
        original_noise_scale= self.noise_scale
        self.noise_scale = noise_scale
        yield
        self.noise_scale = original_noise_scale