import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, VarianceThreshold

from learnware.algorithm.anomaly_detect.ewma import Ewma
from learnware.algorithm.anomaly_detect.iforest import iForest
from learnware.algorithm.anomaly_detect.polynomial_interpolation import PolynomialInterpolation
from learnware.algorithm.anomaly_detect.statistic import ThreeSigma
from learnware.feature.timeseries.feature_project import extracted_features
from learnware.feature.timeseries.ts_feature_check import stationary_check, mk_trend_check, seasonal_check


class TimeSeriesAnomalyDetectLearnware:

    def __init__(self, **kwargs):
        self.config = kwargs
        if 'granularity' not in self.config:
            self.config['granularity'] = "T"
        if 'feature_gen_window_size' not in self.config:
            self.config['feature_gen_window_size'] = 10
        if 'feature_selected_num' not in self.config:
            self.config['feature_selected_num'] = 5
        if 'ts_period' not in self.config:
            self.config["ts_period"] = 0

        self.selected_features = None

    def fit(self, X: pd.Series, y: pd.Series = None):
        '''
        针对学件进行训练，可以在线进行实时训练
        :param X: 训练的用的序列
        :param y: 如果有标注数据，标注的异常
        :return: 返回学件本身
        '''
        # 时序数据整理、特征生成
        # 数据会按照分钟进行聚合，对于缺失的数据进行bfill补充
        # 默认情况，如数据粒度本身小于分钟，依然会按照分钟进行填充, 否则就会使用granularity的值
        # 训练中会根据Series生成Dataframe，ts：时间列、value：值列、lablel：标记列
        feature_train_X = self._preprocess_data(X, y)

        # 检测数据特征，分别数据类型
        # 主要判断数据是否有趋势和季节性，选择不同的模型进行训练
        # 发现是否存在周期，会设置时序周期
        self._metric_ts_feature(feature_train_X)

        # 特征选择和特征生成对会判断是否已经完成特征工程
        window_size = self.config['feature_gen_window_size']
        seasonal_size = self.config["ts_period"]
        if self.selected_features is None:
            # 时序数据特征生成&时序数据特征选择
            # 特征的生成需要设定feature_gen_window_size来确定特征计算的窗口大小
            feature_train_X = self._ts_feature_select(feature_train_X, window_size, seasonal_size)
        else:
            feature_train_X = extracted_features(feature_train_X, window_size,
                                                 sesaonal=seasonal_size, select_features=self.selected_features)
        # 时序数据模型训练生成model
        self._ts_outlier_anomaly_detect_fit(feature_train_X["value"], feature_train_X[self.selected_features])
        return self

    def _preprocess_data(self, X, y=None):
        feature_train_X = pd.DataFrame()
        feature_train_X["value"] = X.resample(self.config['granularity']).mean()
        feature_train_X["value"] = feature_train_X["value"].fillna(method="bfill")
        feature_train_X["ts"] = X.index.to_series()
        if y is not None: feature_train_X["label"] = y
        feature_train_X.index = feature_train_X["ts"]
        return feature_train_X

    def predict_one(self, X, history_data=None):
        r1 = self._statistic_model.predict_one(X)
        r2 = self._ewma_model.predict_one(X)
        if r1 < 0 or r2 < 0:
            return -1, 1
        else:
            window_size = self.config['feature_gen_window_size']
            seasonal_size = self.config["ts_period"]

            train_X = self._preprocess_data(history_data)
            ts_datetime = pd.date_range(start=train_X['ts'][0], periods=len(train_X) + 1, freq=self.config["granularity"])
            ts = train_X["value"]
            ts = ts.append(pd.Series([X]), ignore_index=True)
            df = pd.DataFrame({"ts": ts_datetime, "value": ts})
            df.index = df["ts"]
            feature_train_x = extracted_features(df, window_size,
                                                 sesaonal=seasonal_size, select_features=self.selected_features)
            feature_train_x[self.selected_features]
            print(feature_train_x[self.selected_features].head())
            print(feature_train_x[self.selected_features].iloc[-1, :])

            self._iforest_model.fit(feature_train_x[self.selected_features].iloc[-200:, :].values)
            r3, conf3 = self._iforest_model.predict(feature_train_x[self.selected_features].iloc[-2:, :].values)
            return r3[-1], conf3[-1]

    def _metric_ts_feature(self, X):
        compressed_X = X.resample(self.config['granularity']).mean().fillna(method="bfill")
        is_stationary = stationary_check(compressed_X.values)
        is_trend, p_value = mk_trend_check(compressed_X.values)
        period, acf_scores = seasonal_check(compressed_X.values)
        diff_compressed_X = compressed_X.diff(1).dropna()
        diff_period, diff_acf_scores = seasonal_check(diff_compressed_X.values)
        self.config["ts_period"] = period
        print(is_stationary, is_trend, period, acf_scores, diff_period, diff_acf_scores)

    def _ts_feature_select(self, feature_train_X, window_size, seasonal=0):
        """
        提取单指标数据的特征，针对有标注的数据选择RFE方法进行递归的特征选择；针对无标注数据利用区分度过滤特征。
        PS：需要关注指标选择过程会修改私有边龙selected_features，保存选择的特征名称
        :param feature_train_X: 指标数据
        :param window_size: 生成指标的窗口大小
        :return: 返回选择特征的数据
        """
        # 首先生成时序特征，具体可详看extracted_features
        feature_train_X = extracted_features(feature_train_X, window_size, sesaonal=seasonal)
        feature_train_X = feature_train_X[window_size:]

        # 判断是否存在标注数据，选择不同的方法进行特征选择
        if "label" in feature_train_X.columns:
            # 针对不平衡的数据采用下采样，做到异常数量和正常的数量相等
            # PS：如果数据量比较少，这里需要使用SMOTE方法
            # 建议针对有标注的数据，固定训练模型的数据，并保障数据量足够
            f_df = feature_train_X.fillna(method='ffill')
            f_df_adnormal = f_df[f_df["label"] == 1]
            adnormal_num = len(f_df_adnormal)
            f_df_normal = f_df[f_df["label"] == 0].sample(n=adnormal_num, random_state=42)

            train_df = pd.concat([f_df_adnormal, f_df_normal])
            train_columns = [c for c in train_df.columns if c != "label" and c != "ts"]

            # 通过RFE进行特征选择，速度比较慢，TODO：需要尝试更快的方法，提升效能
            rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=42),
                      n_features_to_select=self.config['feature_selected_num'], step=1)
            fit = rfe.fit(train_df[train_columns].values, train_df["label"].values)

            self.selected_features = [train_columns[i] for i, v in enumerate(fit.support_) if v]
        else:
            # 完成针对数据区分度(方差）来选择指标的方法，目前未做选择
            f_df = feature_train_X.fillna(method='ffill')
            train_columns = [c for c in f_df.columns if c != "label" and c != "ts"]
            varianceSelect = VarianceThreshold().fit(f_df[train_columns])
            self.selected_features = [train_columns[i] for i, v in enumerate(varianceSelect.get_support()) if v]

        return feature_train_X[self.selected_features]

    def _ts_outlier_anomaly_detect_fit(self, X, featured_X, y=None):
        self._statistic_model = ThreeSigma().fit(X)
        self._ewma_model = Ewma().fit(X)
        # self._polynomial_interpolation = PolynomialInterpolation().fit(X)
        self._iforest_model = iForest().fit(featured_X.values)
