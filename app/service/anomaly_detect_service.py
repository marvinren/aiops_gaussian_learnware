from learnware.algorithm.anomaly_detect.iforest import iForest


def metric_simple_model_anomaly_detect(detector, history_data, data, confidence_return=True):
    model = iForest()
    model.fit(history_data)
    return model.predict(data, confidence_return)