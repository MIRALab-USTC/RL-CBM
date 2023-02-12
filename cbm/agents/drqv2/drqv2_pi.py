from cbm.values.ensemble_value import EnsembleQValue
from cbm.policies.truncnorm_policy import TruncNormPolicy
from torch import nn
from typing import Optional, Tuple
import torch

class DrQv2Policy(TruncNormPolicy):
    def __init__( 
        self, 
        env, 
        processor,
        embedding_size: int = 50,
        trunk_detach: bool = False,
        noise_clip: float = 0.3,
        policy_name: str = 'drqv2_policy',
        **mlp_kwargs
    ):
        self.embedding_size = embedding_size
        self.feature_size = embedding_size
        super().__init__(
            env, 
            noise_clip=noise_clip, 
            noise_scale=1.0,
            policy_name=policy_name, 
            **mlp_kwargs
        )
        self.trunk = nn.Sequential(
            nn.Linear(processor.output_shape[0], self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.Tanh()
        )
        self.trunk_detach = trunk_detach

    def _get_feature_size(self):
        return self.feature_size

    def _get_features(self, obs):
        obs = self.trunk(obs)
        if self.trunk_detach:
            obs = obs.detach()
        return super()._get_features(obs)

        




