import abc
import cbm.torch_modules.utils as ptu

class Processor(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def process(self, x):
        pass

    def process_np(self, x, **kwarg):
        x = ptu.from_numpy(x)
        return ptu.get_numpy(self.process(x, **kwarg)) 

    def forward(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def recover(self, *args, **kwargs):
        raise NotImplementedError

    def recover_np(self, x, **kwarg):
        x = ptu.from_numpy(x)
        return ptu.get_numpy(self.recover(x, **kwarg)) 

class PairProcessor(Processor):
    @abc.abstractmethod
    def process(self, o, a):
        pass

    def process_np(self, o, a, **kwarg):
        x = ptu.from_numpy(x)
        return ptu.get_numpy(self.process(o, a, **kwarg)) 

class TrajectorProcessor(Processor):
    @abc.abstractmethod
    def process(self, t, extra_or):
        pass

    def process_np(self, t, extra_or, **kwarg):
        raise NotImplementedError

class Identity(Processor):
    def __init__(self, shape=None):
        self.input_shape = self.output_shape = shape

    def process(self, x):
        return x

    def process_np(self, x):
        return x

    def recover(self, x):
        return x
    
    def recover_np(self, x):
        return x
    
identity = Identity()