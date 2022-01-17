import pandas as pd

from learnware.algorithm.anomaly_detect.statistic import ThreeSigma


class TestTimeSeriesAnomalyDetect:

    def test_ts_statistic_anomaly_detect(self):
        ts = pd.Series([1,3,5,6,2,2,3,5,6,100])
        model = ThreeSigma()
        ret = model.predict(ts)
        assert ret == -1