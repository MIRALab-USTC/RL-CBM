from typing import Optional, Tuple
from cbm.policies.base_policy import RandomPolicy
from cbm.torch_modules.gaussian import MeanLogstdGaussian
from cbm.utils.logger import logger
from torch import nn
import torch

class GaussianPolicy(nn.Module, RandomPolicy):
    def __init__( 
        self, 
        env, 
        deterministic: bool = False,
        squashed: bool = True,
        policy_name: str = 'gaussian_policy',
        **gaussian_kwargs
    ) -> None:
        nn.Module.__init__(self) 
        RandomPolicy.__init__(self, env, deterministic)
        self.squashed = squashed

        self.module = MeanLogstdGaussian( 
            self._get_feature_size(), 
            self.action_shape[0],
            squashed=squashed,
            module_name=policy_name,
            **gaussian_kwargs
        )
        
    def _get_feature_size(self) -> int:
        return self.observation_shape[0]

    def _get_features(self, o: torch.Tensor) -> torch.Tensor:
        return o
        
    def action( 
        self, 
        obs: torch.Tensor, 
        return_log_prob: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        input_tensor = self._get_features(obs)
        return self.module(
            input_tensor, 
            deterministic=self._deterministic, 
            return_log_prob=return_log_prob,
            **kwargs
        )
    
    def log_prob(
        self, 
        obs: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        return self.module.log_prob(obs, action)

    def save(self, save_dir: Optional[str] = None) -> None:
        if save_dir == None:
            save_dir = logger._snapshot_dir
        self.module.save(save_dir)
    
    def load(self, load_dir: Optional[str] = None) -> None:
        if load_dir == None:
            load_dir = logger._snapshot_dir
        self.module.load(load_dir)

    def get_snapshot(self):
        return self.module.get_snapshot()
