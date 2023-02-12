import numpy as np
import torch
from cbm.utils.misc_untils import combine_item
import cbm.torch_modules.utils as ptu
from collections import OrderedDict
from tqdm import tqdm

def rollout(
        env,
        agent,
        max_time_steps=1000,
        use_tqdm=False,
    ):
    
    pass

# Note the difference between agent_infos/env_infos
# TODO: check 
class Path():
    def __init__(self, data_type="numpy"):
        self.actions = [] # len
        self.agent_infos = [] # len; [{[],[]}, {[],[]}]
        self.observations = [] # len + 1
        self.rewards = [] # len
        self.terminals = [] # len + 1
        self.env_infos = [] # len; [[{}, {}], [{}, {}]]
        self.extra_info = []

        self.path = None
        self.batch = None
        self.step_count = 0

        self.data_type = data_type

    def start_new_paths(self, o , t=None):
        self.observations.append(o)
        self.o = o
        self.n_env = len(o)
        if self.data_type == "numpy":
            self.t = np.ones((self.n_env,1))
        elif self.data_type == "torch":
            self.t = ptu.ones
        self.terminals.append(self.t)
        
        #padding
        if len(self.actions) > 0:
            self.actions.append(self.actions[-1])
            self.agent_infos.append(self.agent_infos[-1])
            self.rewards.append(self.rewards[-1])
            self.env_infos.append(self.env_infos[-1])
        

    def update(self, a, agent_info, next_o, r, d, env_info):
        self.actions.append(a)
        self.agent_infos.append(agent_info)
        
        self.o = next_o
        if self.data_type == "numpy":
            self.step_count = self.step_count + (1-self.t.astype(int))
            self.t = np.logical_or(self.t, d)
        elif self.data_type == "torch":
            self.step_count = self.step_count + (1-self.t.int())
            self.t = torch.logical_or(self.t, d)

        self.observations.append(next_o)
        self.rewards.append(r)
        self.terminals.append(self.t)
        self.env_infos.append(env_info)
        return self.t
    
    def wipe_memory(self):
        self.__init__(self.data_type)
    
    def get_terminal(self):
        return self.t
    
    def get_total_steps(self):
        if self.data_type == "numpy":
            return np.sum(self.step_count)
        elif self.data_type == "torch":
            return torch.sum(self.step_count)

    def get_statistic(
        self,
        discard_uncomplete_paths=True,
    ):
        pass
    
    def get_useful_infos(
        self,
        info=[],
        useful_keys=None,
        return_type="path"
    ):
        raise NotImplementedError

    def get_batch(
        self,
        useful_env_info=None,
        useful_agent_info=None,
        useful_extra_info=None,
        qlearning_form=True,
    ):
        raise NotImplementedError

    def get_path(
        self,
        useful_env_info=None,
        useful_agent_info=None,
        useful_extra_info=None,
        cut_path=False
    ):
        raise NotImplementedError

def path_to_samples(paths):
    path_number = len(paths)
    data = paths[0]
    for i in range(1,path_number):
        data = combine_item(data, paths[i])
    return data
