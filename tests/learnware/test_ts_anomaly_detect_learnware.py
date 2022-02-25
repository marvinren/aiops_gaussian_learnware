import pandas as pd
import numpy as np
from learnware.ts_anomaly_detect_learnware import TimeSeriesAnomalyDetectLearnware


class TestTimeSeriesAnomalyDetectLearnware:

    def test_train_data_without_label(self):
        # 读取测试数据
        df = pd.read_csv("../../data/kpi.csv")
        df['ts'] = pd.to_datetime(df['timestamp'], unit='s')
        df.index = df['ts']
        df = df[['ts', 'value', 'label']]

        df = df[:3000]

        learnware = TimeSeriesAnomalyDetectLearnware()
        l = learnware.fit(df['value'])
        print(l.selected_features)

    def test_train_data_select_feature_without_label(self):
        df = self.generate_ts_data()

        learnware = TimeSeriesAnomalyDetectLearnware()
        learnware = learnware.fit(df)

        assert "value" in learnware.selected_features
        assert len(learnware.selected_features) > 10

    def test_train_predict_one_without_label(self):
        ts = self.generate_ts_data()
        learnware = TimeSeriesAnomalyDetectLearnware()
        learnware = learnware.fit(ts)
        ret, conf = learnware.predict_one(3.0)
        assert ret == -1
        ret, conf = learnware.predict_one(0.1, history_data=ts)
        assert ret == 1

        print(ret, conf)

    def generate_ts_data(self):
        n_steps = 1000
        freq1, freq2, offsets1, offsets2 = np.random.rand(4, 1)
        time = np.linspace(0, 6, n_steps)
        series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave 1
        series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + wave 2
        series += 0.1 * (np.random.rand(n_steps) - 0.5)  # + noise
        ts = pd.date_range(start='1/1/2022', periods=n_steps, freq='T')
        df = pd.Series(series)
        df.index = ts
        return df
