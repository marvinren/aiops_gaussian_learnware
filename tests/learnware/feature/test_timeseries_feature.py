from learnware.feature.timeseries.ts_feature import *
import pandas as pd
import numpy as np


class TestTimeSeriesFeature:
    def test_ts_feature_stationary_test(self):
        df1 = pd.DataFrame(np.random.randint(0, 200, size=(100, 1)), columns=['x'])
        df2 = pd.util.testing.makeTimeDataFrame(50)
        df3 = pd.DataFrame([1, 2, 3, 2, 3, 1, 1, 1, 1, 5, 5, 5, 8, 9, 9, 10, 11, 12], columns=['x'])

        assert time_series_stationary_test(df1['x'])
        assert time_series_stationary_test(df2['A'])
        assert time_series_stationary_test(df3['x']) == False

    def test_ts_feature_seasonal_decompose(self):
        df = pd.DataFrame(np.random.randint(1, 10, size=(365, 1)), columns=['value'],
                          index=pd.date_range('2021-01-01', periods=365, freq='D'))
        ret = time_series_seasonal_decompose(df['value'])

        assert "seasonal" in ret and len(ret["seasonal"]) == len(df)
        assert "resid" in ret and len(ret["resid"]) == len(df)
        assert "trend" in ret and len(ret["trend"]) == len(df)

    def test_ts_feature_get_seasonal_value(self):
        df = pd.DataFrame(np.random.randint(1, 10, size=(365, 1)), columns=['value'],
                          index=pd.date_range('2021-01-01', periods=365, freq='D'))

        ret = time_series_seasonal_test(df['value'], [1, 30, 60, 120])
        assert (type(ret) is list and len(ret) == 4)
