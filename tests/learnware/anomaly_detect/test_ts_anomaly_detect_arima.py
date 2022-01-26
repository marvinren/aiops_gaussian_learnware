from learnware.algorithm.anomaly_detect.arima import ARIMA
import pandas as pd


class TestTimeSeriesAnomalyDetectARIMA:

    def test_arima_anomaly_detect_one_value(self):
        df = pd.read_csv("../../../data/kpi.csv")
        df['ts'] = pd.to_datetime(df['timestamp'], unit='s')
        df.index = df['ts']
        df = df[['ts', 'value', 'label']]
        df.head()
        train = df["value"].resample("1H").mean().fillna(method="ffill")[-200:]
        model = ARIMA(seasonal_order=(1, 1, 1, 24))
        model.fit(train)
        result = model.predict_one(100.0)
        print(result)
        assert result == -1

    def test_arima_anomaly_detect_values(self):
        df = pd.read_csv("../../../data/kpi.csv")
        df['ts'] = pd.to_datetime(df['timestamp'], unit='s')
        df.index = df['ts']
        df = df[['ts', 'value', 'label']]
        df.head()
        train = df["value"].resample("1H").mean().fillna(method="ffill")[-200:]
        model = ARIMA(seasonal_order=(1, 1, 1, 24))
        pred = model.predict(train)
        print(pred)

        assert len(pred) == len(train)
