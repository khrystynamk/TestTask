from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    def __init__(self, train, test):
        self.model = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
