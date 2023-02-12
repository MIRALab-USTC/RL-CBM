import abc
from contextlib import contextmanager

class Collector(object, metaclass=abc.ABCMeta):
    def start_epoch(self, epoch=None):
        pass

    def end_epoch(self, epoch=None):
        pass

    def get_diagnostics(self):
        return {}
    
    def set_policy(self, policy, epoch=None):
        self.end_epoch(epoch)
        self._policy = policy

    @contextmanager
    def with_policy(self, policy):
        old_policy = self._policy
        self.set_policy(policy)
        yield
        self.set_policy(old_policy)
 

class PathCollector(Collector, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_paths(
        self,
        max_path_length,
        num_steps,
    ):
        pass


class StepCollector(Collector, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_steps(
        self,
        max_path_length,
        num_steps,
    ):
        pass
