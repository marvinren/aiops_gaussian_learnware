import pandas as pd
from pandas import Series, DataFrame

from learnware.algorithm.anomaly_detect.arima import ARIMA
from learnware.algorithm.anomaly_detect.ewma import Ewma
from learnware.algorithm.anomaly_detect.iforest import iForest
from learnware.algorithm.anomaly_detect.polynomial_interpolation import PolynomialInterpolation
from learnware.algorithm.anomaly_detect.statistic import ThreeSigma
from learnware.detector.base import BaseDetector
from learnware.feature.timeseries.feature_project import extracted_features
from learnware.feature.timeseries.ts_feature_check import *

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor


class TimeSeriesAnomalyDetector(BaseDetector):

    def __init__(self, config={}):
        self.ts_feature = config["ts_feature"] if "ts_feature" in config else None
        self.selected_features = config["selected_features"] if "selected_features" in config else None

    def _ts_type_check(self, X: Series, resample_time="20T"):
        compressed_X = X.resample(resample_time).mean().fillna(
            method="bfill") if resample_time is not None else X.fillna(method="bfill")
        is_stationary = stationary_check(compressed_X)
        is_trend, p_value = mk_trend_check(compressed_X)
        period, acf_scores = seasonal_check(compressed_X.values)
        diff_compressed_X = compressed_X.diff(1).dropna()
        diff_period, diff_acf_scores = seasonal_check(diff_compressed_X.values)

        self.ts_feature["is_stationary"] = is_stationary == 1
        self.ts_feature["period"] = period if acf_scores > diff_acf_scores else diff_period
        self.ts_feature["is_diff"] = diff_acf_scores is not None and diff_acf_scores > acf_scores
        self.ts_feature["trend"] = is_trend
        self.ts_feature["check_time"] = resample_time

    def _ts_feature_select(self, feature_train_X: DataFrame, window_size=10):

        feature_train_X = extracted_features(feature_train_X, window_size)
        feature_train_X = feature_train_X[window_size:]

        f_df = feature_train_X.fillna(method='ffill')
        f_df_adnormal = f_df[f_df["label"] == 1]
        adnormal_num = len(f_df_adnormal)
        f_df_normal = f_df[f_df["label"] == 0].sample(n=adnormal_num, random_state=42)

        train_df = pd.concat([f_df_adnormal, f_df_normal])
        train_columns = [c for c in train_df.columns if c != "label" and c != "ts"]
        if "label" in train_df.columns:
            rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=42), n_features_to_select=5, step=1)
            fit = rfe.fit(train_df[train_columns].values, train_df["label"].values)

            self.selected_features = [train_columns[i] for i, v in enumerate(fit.support_) if v]
        else:
            self.selected_features = train_columns
        return feature_train_X[self.selected_features]

    def fit(self, X: Series, y: Series = None):
        feature_train_X = pd.DataFrame()
        feature_train_X["value"] = X.resample("T").mean()
        feature_train_X["value"] = feature_train_X["value"].fillna(method="bfill")
        feature_train_X["ts"] = X.index.to_series()
        if y is not None: feature_train_X["label"] = y
        feature_train_X.index = feature_train_X["ts"]

        if self.ts_feature is None:
            self.ts_feature = {}
            self._ts_type_check(X)
        print(self.ts_feature)
        window_size = 10
        if self.selected_features is None:
            feature_train_X = self._ts_feature_select(feature_train_X, window_size)
        print(self.selected_features)

        if self.ts_feature["is_stationary"]:
            self.model_func = "_stationary_ts_anomaly_detect_predict"
        else:
            self.model_func = "_not_stationary_ts_anomaly_detect_predict"

        return None

    def predict(self, X: Series):
        if self.model_func == "_not_stationary_ts_anomaly_detect_predict":
            return self._not_stationary_ts_anomaly_detect_predict(X)
        else:
            return self._stationary_ts_anomaly_detect_predict(X)

    def _not_stationary_ts_anomaly_detect_predict(self, X: Series, y: Series = None):
        self._statistic_model = ThreeSigma()
        self._ewma_model = Ewma()
        self._polynomial_interpolation = PolynomialInterpolation()
        self._iforest_model = iForest()

        window_size = 10
        feature_train_X = pd.DataFrame()
        feature_train_X["value"] = X.resample("T").mean()
        feature_train_X["value"] = feature_train_X["value"].fillna(method="bfill")
        feature_train_X["label"] = y
        feature_train_X["ts"] = feature_train_X.index

        self._statistic_model = self._statistic_model.fit(feature_train_X["value"].values)
        result1 = self._statistic_model.predict(feature_train_X["value"].values)
        result2 = self._ewma_model.predict(feature_train_X["value"].values)
        # result3 = self._polynomial_interpolation.predict(feature_train_X["value"].values)

        feature_train_X = extracted_features(feature_train_X, window_size, self.selected_features)
        feature_train_X = feature_train_X.fillna(method="bfill")

        self._iforest_model.fit(feature_train_X[self.selected_features].values)
        result4, conf4 = self._iforest_model.predict(feature_train_X[self.selected_features].values, True)

        ret = []
        prob = []
        for i in range(len(result1)):
            if int(result1[i]) == -1 or int(result2[i]) == -1:
                ret.append(-1)
                prob.append(1)
            else:
                ret.append(result4[i])
                prob.append(conf4[i])

        return ret, prob

    def _stationary_ts_anomaly_detect_predict(self, X: Series):
        model = ARIMA(seasonal_order=(1, 1, 1, 1440))
        pred = model.predict(X)
        return pred, None

    def predict_one(self, value):
        pass
