from cbm.policies.base_policy import Policy
import numpy as np
from torch import nn
import torch

def make_determinsitic(random_policy):
    return MakeDeterministic(random_policy)

class MakeDeterministic(nn.Module, Policy):
    def __init__(self, random_policy):
        nn.Module.__init__(self)
        self.random_policy = random_policy

    def action(self, obs, **kwargs):
        with self.random_policy.deterministic_(True):
            return self.random_policy.action(obs, **kwargs)

    def action_np(self, obs, **kwargs):
        with self.random_policy.deterministic_(True):
            return self.random_policy.action_np(obs, **kwargs)

    def reset(self, **kwarg):
        self.random_policy.reset(**kwarg)

    def save(self, **kwarg):
        self.random_policy.save(**kwarg)

    def load(self, **kwarg):
        self.random_policy.load(**kwarg)

class TanhPolicyModule(nn.Module):
    def __init__(self, policy, discrete=False, epsilon=1e-8):
        super(TanhPolicyModule, self).__init__()
        self._inner_policy = policy
        self.discrete = discrete
        self.eps = epsilon
    
    def forward(
        self,
        obs,
        return_pretanh_action=False,
        **kwargs
    ): 
        pre_action, info = self._inner_policy(obs, **kwargs)
        action = torch.tanh(pre_action)
        keys = list(info.keys())
        for k in keys:
            v = info.pop(k)
            info['pretanh_'+k] = v
            if k == 'log_prob':
                if self.discrete:
                    log_prob = v
                else:
                    log_prob = v - torch.log(1 - action*action + self.eps).sum(-1, keepdim=True)
                info['log_prob'] = log_prob

        if return_pretanh_action:
            info['pretanh_action'] = pre_action
        return action, info

    def log_prob(self, obs, action=None, pretanh_action=None, **kwargs):
        if pretanh_action is None:
            assert action is not None
            pretanh_action = 0.5 * torch.log(1+action+self.eps) - 0.5 * torch.log(1-action+self.eps) 
        else:
            action = torch.tanh(pretanh_action)
        pre_log_prob = self._inner_policy.log_prob(obs, pretanh_action, **kwargs)
        log_prob = pre_log_prob - torch.log(1 - action * action + self.eps).sum(-1, keepdim=True)
        return log_prob

    def save(self, save_dir, **kwargs):
        self._inner_policy.save(save_dir, **kwargs)

    def load(self, load_dir, **kwargs):
        self._inner_policy.load(load_dir, **kwargs)

    def get_snapshot(self, **kwargs):
        return self._inner_policy.get_snapshot(**kwargs)

    def load_snapshot(self, **kwargs):
        self._inner_policy.load_snapshot(**kwargs)

