from typing import List

import numpy as np
from pandas import Series
import pandas as pd
from learnware.algorithm.anomaly_detect.base import BaseAnomalyDetect
import statsmodels.api as sm


class ARIMA(BaseAnomalyDetect):

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1440)):
        self.order = order
        self.seasonal_order = seasonal_order
        self._model = None
        self._results = None

    def fit(self, X: Series, y: Series = None):
        trainX = X
        self._model = sm.tsa.SARIMAX(trainX, order=self.order, seasonal_order=self.seasonal_order,
                                     enforce_stationarity=False, enforce_invertibility=False)
        results = self._model.fit()
        self._results = results
        return self

    def predict_one(self, X: float) -> int:
        pred = self._results.get_forecast()
        pred_ci = pred.conf_int()
        #pred_value = pred.predicted_mean.item()
        if X > pred_ci.iloc[0, 1] or X < pred_ci.iloc[0, 0]:
            return -1
        else:
            return 1

    def predict(self, X: Series) -> List:
        trainX = X
        self._model = sm.tsa.SARIMAX(trainX, order=self.order, seasonal_order=self.seasonal_order,
                                     enforce_stationarity=False, enforce_invertibility=False)
        results = self._model.fit()

        pred = results.get_prediction(start=0)
        forecast_ci = pred.conf_int()
        forecast_ci["value"] = X
        labels = []
        for idx, row in forecast_ci.iterrows():
            if row["value"] > row["upper value"] or row["value"] < row["lower value"]:
                labels.append(-1)
            else:
                labels.append(1)
        return labels

