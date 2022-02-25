import numpy as np
from scipy.stats import binom
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from scipy.special import erf

from learnware.algorithm.anomaly_detect.base import BaseAnomalyDetect


class iForest(BaseAnomalyDetect):

    def __init__(self, n_estimators=100,
                 max_samples="auto",
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 behaviour='old',
                 random_state=None,
                 verbose=0):
        super(iForest, self).__init__()
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.behaviour = behaviour
        self.random_state = random_state
        self.verbose = verbose
        # 内部算法的检测器
        self.detector_ = None
        self.decision_scores_ = None
        self.threshold_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        self.detector_ = IsolationForest(n_estimators=self.n_estimators,
                                         max_samples=self.max_samples,
                                         contamination=self.contamination,
                                         max_features=self.max_features,
                                         bootstrap=self.bootstrap,
                                         n_jobs=self.n_jobs,
                                         random_state=self.random_state,
                                         verbose=self.verbose)
        X = self._data_type_transform(X)
        self.detector_.fit(X, y=None, sample_weight=None)
        self.decision_function(X)
        self._decision_threshold_process()
        return self

    def predict(self, X, return_confidence=False):
        X = self._data_type_transform(X)
        if self.detector_ is None:
            raise EOFError("detector not found, please fit the train data.")
        pred_score = self.decision_function(X)
        prediction = np.ones_like(pred_score, dtype=int)
        prediction[pred_score < self.threshold_] = -1

        if return_confidence:
            confidence = self.predict_confidence(X)
            return prediction, confidence

        return prediction

    def decision_function(self, X):
        if self.detector_ is None:
            raise EOFError("detector not found, please fit the train data.")
        self.decision_scores_ = self.detector_.decision_function(X)
        return self.decision_scores_

    def _decision_threshold_process(self):
        self.threshold_ = np.percentile(self.decision_scores_,
                                        100 * self.contamination)
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(
            'int').ravel()

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return self

    def predict_confidence(self, X):
        n = len(self.decision_scores_)

        test_scores = self.decision_function(X)

        count_instances = np.vectorize(
            lambda x: np.count_nonzero(self.decision_scores_ <= x))
        n_instances = count_instances(test_scores)

        # Derive the outlier probability using Bayesian approach
        posterior_prob = np.vectorize(lambda x: (1 + x) / (2 + n))(n_instances)

        # Transform the outlier probability into a confidence value
        confidence = np.vectorize(
            lambda p: 1 - binom.cdf(n - np.int(n * self.contamination), n, p))(
            posterior_prob)
        prediction = (test_scores > self.threshold_).astype('int').ravel()
        np.place(confidence, prediction == 0, 1 - confidence[prediction == 0])

        return confidence

    def _data_type_transform(self, X):
        if type(X) is list:
            return np.array(X).reshape(-1, 1)
        return X
