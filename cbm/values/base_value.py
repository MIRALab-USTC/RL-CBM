import abc
import cbm.torch_modules.utils as ptu

class Value(object, metaclass=abc.ABCMeta):
    def __init__(self, env):
        self.action_shape = env.action_space.shape
        self.observation_shape = env.observation_space.shape

    def save(self, save_dir=None):
        raise NotImplementedError
     
    def load(self, load_dir=None):
        raise NotImplementedError

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}


class StateValue(Value, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def value(self, obs):
        pass

    def value_np(self, obs, **kwargs):
        obs = ptu.from_numpy(obs)
        value, info = self.value(obs, **kwargs)
        value = ptu.get_numpy(value)
        info = ptu.torch_to_np_info(info)
        return value, info


class QValue(Value, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def value(self, obs, action):
        pass
    
    def value_np(self, obs, action, **kwargs):
        obs = ptu.from_numpy(obs)
        action = ptu.from_numpy(action)
        value, info = self.value(obs, action, **kwargs)
        value = ptu.get_numpy(value)
        info = ptu.torch_to_np_info(info)
        return value, info

