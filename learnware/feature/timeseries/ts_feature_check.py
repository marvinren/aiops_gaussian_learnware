from pandas import Series
from scipy.fftpack import fft, fftfreq
from statsmodels.tsa.stattools import acf, adfuller
from scipy.stats import norm
import numpy as np


def stationary_check(ts: Series):
    """
    对数据是否是平稳数据进行检测，使用adfuller方法
    :param ts: 时序数据
    :return: 1：平稳数据，0：非平稳数据
    """
    result = adfuller(ts)
    p_value = result[1]
    return 1 if p_value < 0.5 else 0


def seasonal_check(ts):
    """
    针对时序分析周期性，首先采用fft进行时域和频域的拆分，发现最坑的lags，然后再利用acf来计算相关系数，对于超过自相关系统超过0.5的lag
    :param ts: 时序数据
    :return: period, acf_score
    period: 周期值，如果无周期，返回0
    acf_score: 自相关系数值
    """

    fft_series = fft(ts)
    power = np.abs(fft_series)
    sample_freq = fftfreq(fft_series.size)

    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    powers = power[pos_mask]

    top_k_seasons = 3
    # top K=3 index
    top_k_idxs = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
    top_k_power = powers[top_k_idxs]
    fft_periods = (1 / freqs[top_k_idxs]).astype(int)

    acf_scores = []
    for lag in fft_periods:
        acf_score = acf(ts, nlags=lag)[-1]
        acf_scores.append(acf_score)

    p = np.argsort(acf_scores)[-1]
    if p is not None and acf_scores[p] > 0.5:
        return fft_periods[p], acf_scores[p]
    else:
        return 0, 0


def mk_trend_check(x, alpha=0.1):
    """
    对时序数据进行趋势检测，使用MK（Mann-Kendall）检验
    :param x: 时序数据
    :param alpha: 置信度
    :return: 0：没有趋势，1: 趋势增长，-1: 趋势下降
    """
    n = len(x)

    s = 0
    for j in range(n - 1):
        for i in range(j + 1, n):
            s += np.sign(x[i] - x[j])

    unique_x, tp = np.unique(x, return_counts=True)
    g = len(unique_x)

    if n == g:
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:
        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    if n <= 10:
        z = s / (n * (n - 1) / 2)
    else:
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

    p = 2 * (1 - norm.cdf(abs(z)))

    h = abs(z) > norm.ppf(1 - alpha / 2)

    if (z < 0) and h:
        return -1, p
    elif (z > 0) and h:
        return 1, p
    else:
        return 0, p
