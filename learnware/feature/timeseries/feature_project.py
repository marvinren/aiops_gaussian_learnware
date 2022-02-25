import pandas as pd
import numpy as np
import tsfresh as tsf
from pandas import DatetimeIndex
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing


def extracted_features(df, n=10, sesaonal=1440, select_features=None):
    df = ts_time_feature_generate(df, select_features)
    if sesaonal != 0:
        df = ts_exponential_smoothing_feature_generate(df, sesaonal, select_features)
        df = ts_seasonal_decompose(df, sesaonal, select_features)
    df = ts_rolling_statistics_feature_generate(df, n, select_features)
    df = ts_lag_feature_generate(df, n, select_features)
    df = ts_linear_feature_generate(df, n, select_features)
    df = ts_fft_feature_generate(df, n, select_features)
    df = ts_wavelet_feature_generate(df, n, select_features)
    df = ts_wavelet_feature_generate(df, n, select_features)
    return df


def ts_time_feature_generate(data_df, select_features=None):
    # 时间特性，要看训练数据的时间区间这些值是否有价值
    if type(data_df['ts']) is DatetimeIndex:
        data_df['ts'] = data_df['ts'].to_series()
    ts_col = data_df['ts'].dt
    if select_features is None or "quarter" in select_features: data_df['quarter'] = ts_col.quarter
    if select_features is None or "month" in select_features: data_df['month'] = ts_col.month
    if select_features is None or "day" in select_features: data_df['day'] = ts_col.day
    if select_features is None or "dayofweek" in select_features: data_df['dayofweek'] = ts_col.dayofweek
    if select_features is None or "weekofyear" in select_features: data_df['weekofyear'] = ts_col.isocalendar().week
    if select_features is None or "hour" in select_features: data_df['hour'] = ts_col.hour

    # 这些特征一般跟业务属性强关系
    if select_features is None or "is_year_start" in select_features: data_df[
        'is_year_start'] = ts_col.is_year_start.astype(np.int)
    if select_features is None or "is_year_end" in select_features: data_df['is_year_end'] = ts_col.is_year_end.astype(
        np.int)
    if select_features is None or "is_quarter_start" in select_features: data_df[
        'is_quarter_start'] = ts_col.is_quarter_start.astype(np.int)
    if select_features is None or "is_quarter_end" in select_features: data_df[
        'is_quarter_end'] = ts_col.is_quarter_end.astype(np.int)
    if select_features is None or "is_month_start" in select_features: data_df[
        'is_month_start'] = ts_col.is_month_start.astype(np.int)
    if select_features is None or "is_month_end" in select_features: data_df[
        'is_month_end'] = ts_col.is_month_end.astype(np.int)
    if select_features is None or "is_weekend" in select_features: data_df['is_weekend'] = data_df['dayofweek'].apply(
        lambda x: 1 if x == 0 or x == 6 else 0)

    # 是否时一天的高峰时段 8~18
    if select_features is None or "day_high" in select_features: data_df['day_high'] = data_df['hour'].apply(
        lambda x: 1 if 8 < x < 18 else 0)
    # 是否是批跑时段
    if select_features is None or "day_night" in select_features: data_df['day_night'] = data_df['hour'].apply(
        lambda x: 1 if 0 < x < 6 else 0)

    return data_df


def ts_rolling_statistics_feature_generate(data_df, n, select_features=None):
    def mad(x):
        return np.fabs(x - x.mean()).mean()

    if select_features is None or f'rolling_{n}_avg' in select_features: data_df[f'rolling_{n}_avg'] = data_df[
        "value"].rolling(n).mean()
    if select_features is None or f'rolling_{n}_median' in select_features: data_df[f'rolling_{n}_median'] = data_df[
        "value"].rolling(n).median()
    if select_features is None or f'rolling_{n}_max' in select_features: data_df[f'rolling_{n}_max'] = data_df[
        "value"].rolling(n).max()
    if select_features is None or f'rolling_{n}_min' in select_features: data_df[f'rolling_{n}_min'] = data_df[
        "value"].rolling(n).min()
    if select_features is None or f'rolling_{n}_std' in select_features: data_df[f'rolling_{n}_std'] = data_df[
        "value"].rolling(n).std()
    if select_features is None or f'rolling_{n}_var' in select_features: data_df[f'rolling_{n}_var'] = data_df[
        "value"].rolling(n).var()
    if select_features is None or f'rolling_{n}_mad' in select_features: data_df[f'rolling_{n}_mad'] = data_df[
        "value"].rolling(n).apply(mad, raw=True)
    if select_features is None or f'rolling_{n}_skew' in select_features: data_df[f'rolling_{n}_skew'] = data_df[
        "value"].rolling(n).skew()
    if select_features is None or f'rolling_{n}_kurt' in select_features: data_df[f'rolling_{n}_kurt'] = data_df[
        "value"].rolling(n).kurt()
    if select_features is None or f'rolling_{n}_corr' in select_features: data_df[f'rolling_{n}_corr'] = data_df[
        "value"].rolling(n).corr()
    if select_features is None or f'rolling_{n}_cov' in select_features: data_df[f'rolling_{n}_cov'] = data_df[
        "value"].rolling(n).cov()
    if select_features is None or f'rolling_{n}_q1' in select_features: data_df[f'rolling_{n}_q1'] = data_df[
        "value"].rolling(n).quantile(0.25)
    if select_features is None or f'rolling_{n}_q3' in select_features: data_df[f'rolling_{n}_q3'] = data_df[
        "value"].rolling(n).quantile(0.75)
    if select_features is None or f'rolling_{n}_ewma' in select_features: data_df[f'rolling_{n}_ewma'] = data_df[
        "value"].ewm(span=n).mean()

    return data_df


def ts_lag_feature_generate(data_df, *lags, select_features=None):
    if select_features is None or f'diff_1' in select_features:
        data_df[f'diff_1'] = data_df["value"] - data_df["value"].shift(1)
        data_df[f'diff_1'] = data_df[f'diff_1'].fillna(value=0.0)
    if select_features is None or f'diff_2' in select_features:
        data_df[f'diff_2'] = data_df["diff_1"] - data_df["diff_1"].shift(1)
        data_df[f'diff_2'] = data_df[f'diff_2'].fillna(value=0.0)
    # for n in lags:
    #     if select_features is None or f'ago_{n}_diff_1' in select_features:
    #         data_df[f'ago_{n}_diff_1'] = (data_df["value"] - data_df["value"].shift(n)).fillna(value=0)
    return data_df


def ts_linear_feature_generate(data_df, n, select_features=None):
    if select_features is None or f'rolling_{n}_linear_trend_slope' in select_features:
        data_df[f'rolling_{n}_linear_trend_slope'] = data_df["value"].rolling(n) \
            .apply(
            lambda data: list(tsf.feature_extraction.feature_calculators.linear_trend(data, [{'attr': 'slope'}]))[0][1])
    if select_features is None or f'rolling_{n}_linear_trend_intercept' in select_features:
        data_df[f'rolling_{n}_linear_trend_intercept'] = data_df["value"].rolling(n) \
            .apply(
            lambda data:
            list(tsf.feature_extraction.feature_calculators.linear_trend(data, [{'attr': 'intercept'}]))[0][1])

    return data_df


def ts_exponential_smoothing_feature_generate(data_df, seasonal=1440, select_features=None):
    if select_features is None or f'smoothing_1exp' in select_features:
        data_df['smoothing_1exp'] = SimpleExpSmoothing(data_df["value"]).fit(smoothing_level=0.5).fittedvalues
    if select_features is None or f'smoothing_2exp' in select_features:
        data_df['smoothing_2exp'] = ExponentialSmoothing(data_df["value"], trend="add",
                                                         seasonal=None).fit().fittedvalues
    if select_features is None or f'smoothing_3exp' in select_features:
        data_df['smoothing_3exp'] = ExponentialSmoothing(data_df["value"], trend="add", seasonal="add",
                                                         seasonal_periods=seasonal).fit().fittedvalues
    return data_df


def ts_seasonal_decompose(data_df, seasonal=1440, select_features=None):
    result_mul = None
    if select_features is None or 'seasonal_decompose_trend' in select_features:
        result_mul = seasonal_decompose(data_df['value'], model='additive', extrapolate_trend='freq',
                                        period=seasonal) if result_mul is None else result_mul
        data_df['seasonal_decompose_trend'] = result_mul.trend
    if select_features is None or 'seasonal_decompose_seasonal' in select_features:
        result_mul = seasonal_decompose(data_df['value'], model='additive', extrapolate_trend='freq',
                                        period=seasonal) if result_mul is None else result_mul
        data_df['seasonal_decompose_seasonal'] = result_mul.seasonal
    if select_features is None or 'seasonal_decompose_resid' in select_features:
        result_mul = seasonal_decompose(data_df['value'], model='additive', extrapolate_trend='freq',
                                        period=seasonal) if result_mul is None else result_mul
        data_df['seasonal_decompose_resid'] = result_mul.resid
    return data_df


def ts_fft_feature_generate(data_df, n, select_features=None):
    if select_features is None or f'rolling_{n}_fft_agg_centroid' in select_features:
        data_df[f'rolling_{n}_fft_agg_centroid'] = data_df["value"].rolling(n).apply(lambda data: list(
            tsf.feature_extraction.feature_calculators.fft_aggregated(data, [{'aggtype': 'centroid'}]))[0][1])
    if select_features is None or f'rolling_{n}_fft_agg_variance' in select_features:
        data_df[f'rolling_{n}_fft_agg_variance'] = data_df["value"].rolling(n).apply(lambda data: list(
            tsf.feature_extraction.feature_calculators.fft_aggregated(data, [{'aggtype': 'variance'}]))[0][1])
    if select_features is None or f'rolling_{n}_fft_coeff_angle' in select_features:
        data_df[f'rolling_{n}_fft_coeff_angle'] = data_df["value"].rolling(n).apply(lambda data: list(
            tsf.feature_extraction.feature_calculators.fft_coefficient(data, [{'coeff': 2, 'attr': 'angle'}]))[0][1])
    if select_features is None or f'rolling_{n}_fft_coeff_real' in select_features:
        data_df[f'rolling_{n}_fft_coeff_real'] = data_df["value"].rolling(n).apply(lambda data: list(
            tsf.feature_extraction.feature_calculators.fft_coefficient(data, [{'coeff': 2, 'attr': 'real'}]))[0][1])
    if select_features is None or f'rolling_{n}_fft_coeff_abs' in select_features:
        data_df[f'rolling_{n}_fft_coeff_abs'] = data_df["value"].rolling(n).apply(lambda data: list(
            tsf.feature_extraction.feature_calculators.fft_coefficient(data, [{'coeff': 2, 'attr': 'abs'}]))[0][1])
    if select_features is None or f'rolling_{n}_fft_coeff_imag' in select_features:
        data_df[f'rolling_{n}_fft_coeff_imag'] = data_df["value"].rolling(n) \
            .apply(lambda data: list(
            tsf.feature_extraction.feature_calculators.fft_coefficient(data, [{'coeff': 2, 'attr': 'imag'}]))[0][1])

    return data_df


def ts_wavelet_feature_generate(data_df, n, select_features=None):
    if select_features is None or f'rolling_{n}_wavelet_cwt_coeff' in select_features:
        data_df[f'rolling_{n}_wavelet_cwt_coeff'] = data_df["value"].rolling(n).apply(lambda data: list(
            tsf.feature_extraction.feature_calculators.cwt_coefficients(data, [
                {'widths': tuple([2, 2, 2]), 'coeff': 2, 'w': 2}]))[0][1])
    return data_df
