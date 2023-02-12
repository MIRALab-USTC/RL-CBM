import random
import abc
import numpy as np
from gym import Wrapper
from gym.spaces import Box
from cbm.environments.utils import make_gym_env
from cbm.utils.logger import logger

class Env(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, env_name):
        self.env_name = env_name

    @abc.abstractproperty
    def n_env(self):
        pass

class SimpleEnv(Env, Wrapper):
    def __init__(self,  env_name):
        super().__init__(env_name)
        self.cur_seed = random.randint(0,65535)
        inner_env = make_gym_env(env_name, self.cur_seed)
        Wrapper.__init__(self, inner_env)
    
    @property
    def n_env(self):
        return 1
    
    def reset(self):
        self.cur_step_id = 0
        return np.array([self.env.reset()])

    def step(self, action):
        self.cur_step_id = self.cur_step_id + 1
        o, r, d, info = self.env.step(action[0])
        o, r, d = np.array([o]), np.array([[r]]), np.array([[d]])
        if logger.log_or_not(logger.WARNING):
            for k in info:
                info[k] = np.array([[info[k]]])
        else:
            info = {}
        return o, r, d, info

    