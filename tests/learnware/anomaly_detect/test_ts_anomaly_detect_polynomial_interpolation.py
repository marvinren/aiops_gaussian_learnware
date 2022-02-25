import numpy as np

from learnware.algorithm.anomaly_detect.polynomial_interpolation import PolynomialInterpolation


class TestTimeSeriesAnomalyDetectPolynomailInterpolation:

    def test_pi_anomaly_detect_one_predict(self):
        rng = np.random.RandomState(42)

        X_train = rng.randn(100, 1)
        model = PolynomialInterpolation()
        result = model.predict_one(X_train)
        print(result)
        assert result == -1

    def test_pi_anomaly_detect_predict(self):
        rng = np.random.RandomState(42)

        X_train = rng.randn(100, 1)
        model = PolynomialInterpolation()
        result = model.predict(X_train)
        print(result)

        assert len(result) == len(X_train)
