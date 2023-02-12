import abc
from collections import OrderedDict
from typing import Iterable
from torch import nn
import cbm.torch_modules.utils as ptu
import numpy as np 
import torch


class Agent(object):
    def __init__(self, env) -> None:
        self.env = env
        self._num_init_steps = 0
        self._num_explore_steps = 0
        self.cur_epoch = 0
    
    def update_epoch(self, epoch=None):
        if epoch is not None:
            self.cur_epoch = epoch
        else:
            self.cur_epoch += 1

    @property
    def num_total_steps(self):
        return self.num_init_steps + self.num_explore_steps
    
    @property
    def num_init_steps(self):
        return self._num_init_steps 

    @property
    def num_explore_steps(self):
        return self._num_explore_steps

    def train(self, data):
        raise NotImplementedError

    def pretrain(self, pool):
        pass

    def start_new_path(self, o):
        pass

    def end_a_path(self, o):
        pass

    def step(self, o, step_mode='exploit', **kwargs):
        with torch.no_grad():
            if step_mode == 'init':
                a, info = self.step_init(o, **kwargs)
                self._num_init_steps += 1
            elif step_mode == 'explore':
                a, info = self.step_explore(o, **kwargs)
                self._num_explore_steps += 1
            elif step_mode == 'exploit':
                a, info = self.step_exploit(o, **kwargs)
            else:
                raise NotImplementedError
        return a, info
        
    def step_init(self, o, **kwargs):
        n = o.shape[0] if hasattr(o, "shape") else 1
        shape = (n, *self.env.action_space.shape)
        action = np.random.uniform(-1,1,shape)
        return action, {}   
    
    def step_explore(self, o, **kwargs):
        if hasattr(self.policy, 'deterministic_'):
            with self.policy.deterministic_(False):
                return self.policy.action_np(o, **kwargs)

    def step_exploit(self, o, **kwargs):
        if hasattr(self.policy, 'deterministic_'):
            with self.policy.deterministic_(True):
                return self.policy.action_np(o, **kwargs)
        else:
            return self.policy.action_np(o, **kwargs)

    def start_epoch(self, epoch):
        pass

    def end_epoch(self, epoch):
        pass

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return {}

class BatchTorchAgent(Agent, metaclass=abc.ABCMeta):
    def __init__(self, env):
        super().__init__(env)

    def train(self, data, *args, **kwargs):
        return self.train_from_numpy_batch(data, *args, **kwargs)

    def train_from_numpy_batch(self, np_batch, *args, **kwargs):
        batch = ptu.np_to_pytorch_batch(np_batch)
        return self.train_from_torch_batch(batch, *args, **kwargs)

    @abc.abstractmethod
    def train_from_torch_batch(self, batch, *args, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass




