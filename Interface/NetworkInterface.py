from abc import abstractmethod, ABC


class Network(ABC):
    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def load_network(self, path):
        pass

    @abstractmethod
    def save_network(self, path):
        pass

    @abstractmethod
    def check_mse_accuracy(self):
        pass