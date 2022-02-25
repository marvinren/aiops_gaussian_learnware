import numpy as np

from learnware.algorithm.anomaly_detect.iforest import iForest


class TestTimeSeriesAnomalyDetectIForest:

    def test_iforest_detect_normal(self):
        rng = np.random.RandomState(42)

        X = 0.3 * rng.randn(100, 2)
        X_train = np.r_[X + 2, X - 2]
        X = 0.3 * rng.randn(20, 2)
        X_test = np.r_[X + 2, X - 2]
        X_outliers = rng.uniform(low=-6, high=6, size=(20, 2))

        model = iForest(contamination=0.1)
        labels = model.fit_predict(X_train)

        assert len(labels) == len(X_train)
        assert -1 in labels or 0 in labels

        test_labels = model.predict(X_test)
        assert len(test_labels) * 0.2 >= len(test_labels[np.where(test_labels < 0)])

        test_outliers_labels = model.predict(X_outliers)
        assert len(test_outliers_labels) * 0.2 <= len(test_outliers_labels[np.where(test_outliers_labels < 0)])

    def test_iforest_detect_conf(self):
        rng = np.random.RandomState(42)

        X = 0.3 * rng.randn(100, 2)
        X_train = np.r_[X + 2, X - 2]
        X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

        model = iForest(contamination=0.05)
        model.fit(X_train)
        ret, conf = model.predict(X_outliers, return_confidence=True)
        assert len(ret) == len(conf)
        assert len(np.where((conf <= 1) & (conf >= 0))) > 0
