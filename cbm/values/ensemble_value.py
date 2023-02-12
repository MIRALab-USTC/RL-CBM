from sklearn import ensemble
import torch
from torch import nn
import numpy as np
from cbm.torch_modules.mlp import MLP
import cbm.torch_modules.utils as ptu
from cbm.utils.logger import logger
from cbm.values.base_value import StateValue, QValue

def sample_from_ensemble(
    input_tensor,
    sample_number=1,
    replace=True
):
    input_size = input_tensor.shape
    assert len(input_size) == 3
    ensemble_size = input_size[0]
    batch_size = input_size[1]
    shape = input_size[-1]
    if replace is False:
        assert sample_number <= ensemble_size
    if replace or (sample_number == 1):
        indices = np.random.choice(ensemble_size, batch_size*sample_number)
    else:
        indices = [np.random.choice(ensemble_size-i, (batch_size,)) for i in range(sample_number)]
        indices = np.stack(indices)
        for i in range(sample_number-1,0,-1):
            for j in range(i,sample_number):
                indices[j] = (indices[j] + indices[i-1] + 1) % (ensemble_size + 1 -i)
        indices = indices.flatten()
    n_arange = np.tile(np.arange(batch_size), sample_number)
    output_tensor = input_tensor[indices, n_arange]
    output_tensor = output_tensor.reshape(sample_number, batch_size, shape)
    return output_tensor

class EnsembleQValue(nn.Module, QValue):
    def __init__( 
        self, 
        env, 
        ensemble_size=2,
        value_name='ensemble_q_value',
        **mlp_kwargs
    ):
        nn.Module.__init__(self)
        QValue.__init__(self, env)
        self.ensemble_size = ensemble_size
        self.module = MLP( 
            self._get_feature_size(), 
            1,
            ensemble_size=ensemble_size,
            module_name=value_name,
            **mlp_kwargs
        )
    
    def _get_feature_size(self):
        return self.observation_shape[0] + self.action_shape[0]

    def _get_features(self, obs, action):
        if obs.dim() > 2:
            obs = obs.unsqueeze(-3)
            action = action.unsqueeze(-3)
        return torch.cat([obs, action], dim=-1)

    def value(
        self, 
        obs, 
        action, 
        sample_number=2, 
        batchwise_sample=False,
        mode='min', 
        return_ensemble=False,
        only_return_ensemble=False
    ):
        input_tensor = self._get_features(obs, action)
        if self.ensemble_size is not None:
            ensemble_value = self.module(input_tensor)
            if only_return_ensemble:
                info = {'ensemble_value': ensemble_value}
                return None, info
            if sample_number is None:
                sample_number = self.ensemble_size
            if self.ensemble_size != sample_number:
                if batchwise_sample:
                    index = np.random.choice(self.ensemble_size, sample_number, replace=False)
                    sampled_value = ensemble_value[...,index,:,:]
                else:
                    sampled_value = sample_from_ensemble(ensemble_value, sample_number, replace=False)
            else:
                sampled_value = ensemble_value
            if mode == 'min':
                value = torch.min(sampled_value, dim=-3)[0]
            elif mode == 'mean':
                value = torch.mean(sampled_value, dim=-3)
            elif mode == 'max':
                value = torch.max(sampled_value, dim=-3)[0]
            elif mode == 'sample':
                index = np.random.randint(sample_number)
                value = sampled_value[index]
            else:
                raise NotImplementedError
        else:
            value = self.module(input_tensor)
        info = {}
        if return_ensemble and self.ensemble_size is not None:
            info['ensemble_value'] = ensemble_value
        return value, info

    def save(self, save_dir=None):
        if save_dir == None:
            save_dir = logger._snapshot_dir
        self.module.save(save_dir)
    
    def load(self, load_dir=None):
        if load_dir == None:
            load_dir = logger._snapshot_dir
        self.module.load(load_dir)

    def get_snapshot(self):
        return self.module.get_snapshot()