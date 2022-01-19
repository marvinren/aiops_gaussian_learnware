import abc


class BaseAnomalyDetector(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, X, y=None):
        pass

    @abc.abstractmethod
    def predict(self, X, return_confidence=False):
        pass

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        pred = self.predict(X)
        return pred



    # @abc.abstractmethod
    # def predict(self, X):
    #     pass

