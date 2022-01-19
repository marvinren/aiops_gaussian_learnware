import numpy as np
from scipy import stats

from learnware.algorithm.anomaly_detect.base import BaseAnomalyDetector


class ThreeSigma(BaseAnomalyDetector):
    """
    使用统计分析里常用的3个sigma的异常检测的方法，发现数据的离群性，主要针对最后一个值来判断该值是否异常
    具体思路可以查看正态分布中的概率分布，68.27%, 95.45% and 99.73%
    """

    def __init__(self, index=3):
        """
        :param index: 用于确定是几个sigma, 类型：int 或者 float, 默认值为3
        """
        self.index = index

    def fit(self, X, y=None):
        self.mu_ = np.mean(X)
        self.sigma_ = np.std(X)
        print(self.mu_)
        print(self.sigma_)
        return self

    def predict(self, X, y=None, return_confidence=False):
        pred = np.ones_like(X, dtype=int)
        pred[abs(X - self.mu_) > self.index * self.sigma_] = -1
        if return_confidence:
            conf = self.predict_confidence(X)
            return pred, conf
        return pred

    def predict_confidence(self, X):
        prob = (100 - stats.norm.pdf(X, self.mu_, self.sigma_)) / 100
        return prob
