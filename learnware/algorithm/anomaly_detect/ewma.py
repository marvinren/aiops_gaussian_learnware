import numpy as np


class Ewma(object):
    def __init__(self, alpha=0.3, coefficient=3, window_size=10):
        self.alpha = alpha
        self.coefficient = coefficient
        self.window_size = 10

    def predict_one(self, X):
        s = [X[0]]
        for i in range(1, len(X)):
            temp = self.alpha * X[i] + (1 - self.alpha) * s[-1]
            s.append(temp)
        s_avg = np.mean(s)
        sigma = np.sqrt(np.var(X))
        ucl = s_avg + self.coefficient * sigma * np.sqrt(self.alpha / (2 - self.alpha))
        lcl = s_avg - self.coefficient * sigma * np.sqrt(self.alpha / (2 - self.alpha))
        print(s)
        if s[-1] > ucl or s[-1] < lcl:
            return -1
        return 1

    def predict(self, X):
        s = [X[0]]
        for i in range(1, len(X)):
            temp = self.alpha * X[i] + (1 - self.alpha) * s[-1]
            s.append(temp)

        labels = []
        for i in range(0, len(X)):
            if i < self.window_size:
                labels.append(0)
                continue
            s_avg = np.mean(s[i - self.window_size:i])
            sigma = np.sqrt(np.var(X[i - self.window_size:i]))
            ucl = s_avg + self.coefficient * sigma * np.sqrt(self.alpha / (2 - self.alpha))
            lcl = s_avg - self.coefficient * sigma * np.sqrt(self.alpha / (2 - self.alpha))
            if s[i] > ucl or s[i] < lcl:
                labels.append(-1)
            else:
                labels.append(1)
        return labels
