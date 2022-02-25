#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose


def time_series_stationary_test(x: pd.Series) -> bool:
    """
    对数据的平稳性进行检测, 主要采用ADF Test的方式进行检测，还可以考虑使用KPSS Test或者PP Test对时序数据的平稳性进行检测
    ADF检验，零假设是时间序列有一个单位根并且非平稳。所以ADF检验p值小于0.05的显著性水平，你拒绝零
    :param x: 时序数据x，type: Series
    :return: 时序数据是否平稳, True: 时序数据属于平稳数据，反之，不属于平稳数据
    """
    from statsmodels.tsa.stattools import adfuller
    ret = adfuller(x.values)
    p_value = ret[1]
    return p_value < 0.05


def time_series_seasonal_decompose(x: pd.Series):
    """
    对数据进行时序分解，将数据分解成趋势trend, 季节seasonal，残差resid
    可用于提取周期视图，判断周期，或者提取趋势，然后通过计算去趋势，去季节
    PS: 这里利用的乘法模型，使用的seasonal_decompose方法来完成分解
    :param x: 时序数据x，type: Series
    :return: 返回 趋势trend, 季节seasonal，残差resid的三个序列
    """
    result_mul = seasonal_decompose(x, model='multiplicative', extrapolate_trend='freq')
    return {"seasonal": result_mul.seasonal, "trend": result_mul.trend, "resid": result_mul.resid}


def time_series_seasonal_test(x: pd.Series, expected_lags: list):
    """
    通过自相关系数来获取不同lag的相关系数，通过相关系数来判断时序数据的周期值
    PS：需要列出lag的值的列表
    :param x: 时序数据x，type: Series
    :param expected_lags: 可供选择的的滞后值
    :return: 返回滞后值值的自相关性排序序列
    """
    acf_scores = []
    for lag in expected_lags:
        acf_score = acf(x.values, nlags=lag, fft=False)[-1]
        acf_scores.append(abs(acf_score))
    sorted_idx = np.argsort(acf_scores)
    return [expected_lags[i] for i in sorted_idx]



