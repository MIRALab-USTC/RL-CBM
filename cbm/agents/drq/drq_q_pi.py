from typing import Optional, Tuple
import torch
from cbm.values.ensemble_value import EnsembleQValue
from cbm.policies.gaussian_policy import GaussianPolicy
from cbm.torch_modules.linear import MyLinear
from torch import nn


class DrQQValue(EnsembleQValue):
    def __init__( 
        self, 
        env, 
        processor,
        embedding_size: int = 50,
        trunk_detach: bool = False,
        value_name='drq_value',
        **ensemble_q_kwargs
    ):
        self.embedding_size = embedding_size
        self.feature_size = embedding_size + env.action_space.shape[0]
        super().__init__(env, value_name=value_name, **ensemble_q_kwargs)
        self.trunk = nn.Sequential(
            nn.Linear(processor.output_shape[0], self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.Tanh()
        )
        self.trunk_detach = trunk_detach

    def _get_feature_size(self):
        return self.feature_size

    def _get_features(self, obs, action):
        obs = self.trunk(obs)
        if self.trunk_detach:
            obs = obs.detach()
        return super()._get_features(obs, action)

class DrQPolicy(GaussianPolicy):
    def __init__( 
        self, 
        env, 
        processor,
        embedding_size: int = 50,
        trunk_detach: bool = False,
        policy_name: str = 'gaussian_policy',
        **mlp_kwargs
    ) -> None:
        self.embedding_size = embedding_size
        self.feature_size = embedding_size
        super().__init__(env, False, True, policy_name, **mlp_kwargs)
        self.trunk = nn.Sequential(
            MyLinear(processor.output_shape[0], self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.Tanh()
        )
        self.trunk_detach = trunk_detach

    def reset_parameters(self):
        for m in self.module.net:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        self.trunk[0].reset_parameters()

    def _get_feature_size(self):
        return self.feature_size

    def _get_features(self, obs):
        obs = self.trunk(obs)
        if self.trunk_detach:
            obs = obs.detach()
        return super()._get_features(obs)
    
    def action_from_feature(self,
        feature: torch.Tensor, 
        return_log_prob: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        return self.module(
            feature, 
            deterministic=self._deterministic, 
            return_log_prob=return_log_prob,
            **kwargs
        )


