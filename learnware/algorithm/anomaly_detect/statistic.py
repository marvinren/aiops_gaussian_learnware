import numpy as np
import pandas as pd


class ThreeSigma:
    """
    使用统计分析里常用的3个sigma的异常检测的方法，发现数据的离群性，主要针对最后一个值来判断该值是否异常
    具体思路可以查看正态分布中的概率分布，68.27%, 95.45% and 99.73%
    """

    def __init__(self, index=3):
        """
        :param index: 用于确定是几个sigma, 类型：int 或者 float, 默认值为3
        """
        self.index = index

    def predict(self, X: pd.Series):
        """
        预测时序数据的最后一个值是否异常
        :param X: 需要检测的时序序列
        :return: 返回-1或者1， 其中-1代表异常，1代表正常
        """
        data = X.values
        if abs(data[-1] - np.mean(data[:-1])) > self.index * np.std(data[:-1]):
            return -1
        return 1
