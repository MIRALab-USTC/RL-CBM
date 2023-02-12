import abc

class Analyzer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def analyze(self):
        pass

    def get_diagnostics(self):
        return {}