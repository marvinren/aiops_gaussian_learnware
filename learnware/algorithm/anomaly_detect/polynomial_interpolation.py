import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class PolynomialInterpolation(object):

    def __init__(self, threshold=0.15, degree=4, window_size=10):
        self.degree = degree
        self.threshold = threshold
        self.window_size = window_size

    def predict(self, X):
        x_train = list(range(0, len(X)))
        x_train = np.array(x_train)
        x_train = x_train[:, np.newaxis]
        avg_value = np.mean(X[-(self.window_size + 1):])
        if avg_value > 1:
            y_train = X / avg_value
        else:
            y_train = X
        model = make_pipeline(PolynomialFeatures(self.degree), Ridge())
        model.fit(x_train, y_train)
        return list(np.where(np.abs(y_train - model.predict(x_train)) > self.threshold, 1, -1).ravel())

    def predict_one(self, X):
        result = self.predict(X)
        return result[-1]
