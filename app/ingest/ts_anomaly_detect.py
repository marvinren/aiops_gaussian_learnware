from datetime import timedelta

import pandas as pd

from app.timeseries.query import load_metric_history_data
from learnware.detector.ts_anomaly_detector import TimeSeriesAnomalyDetector


class TimeSeriesIngestAnomalyStreamDetector:

    def __init__(self):
        self.metric_model_config = {}

    def detect(self, fqm, ts, value):

        # 获取历史数据
        history_data = self.load_history_data(fqm, ts)

        # 获取探测器数据
        if fqm not in self.metric_model_config:
            self.metric_model_config[fqm] = {}
            detector = TimeSeriesAnomalyDetector(self.metric_model_config[fqm])
            detector.fit(history_data)

            detector.predict()



    def load_history_data(self, fqm, ts) -> pd.Series:
        return load_metric_history_data(fqm, ts, ts - timedelta(days=1))
