from abc import abstractmethod, ABC


class Network(ABC):
    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def train(self, x, y, valX, valY, batchSize, epochs):
        pass