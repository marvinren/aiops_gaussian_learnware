import numpy as np
from learnware.algorithm.anomaly_detect.statistic import ThreeSigma


class TestTimeSeriesAnomalyDetect:

    def test_ts_statistic_anomaly_detect(self):
        rng = np.random.RandomState(42)

        X_train = 0.3 * rng.randn(100)
        X_outliers = rng.uniform(low=-24, high=24, size=(20,))

        model = ThreeSigma()
        ret = model.fit_predict(X_train)
        print(ret)

        ret = model.predict(X_outliers)
        assert len(ret[ret < 0]) > 0

    def test_ts_statistic_anomaly_detect_conf(self):
        rng = np.random.RandomState(42)
        X_train = 0.3 * rng.randn(100)
        X_outliers = rng.uniform(low=-4, high=4, size=(20,))
        model = ThreeSigma()
        model.fit(X_train)

        ret, conf = model.predict(X_outliers, return_confidence=True)

        assert len(ret) == len(conf)
