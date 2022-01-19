from flask import request, jsonify

import numpy as np
from app.api import api
from app.service.anomaly_detect_service import metric_simple_model_anomaly_detect
from app.service.metric_service import load_single_metric_data_from_clickhouse


@api.route("/metric/anomaly_metric", methods=["POST"])
def metric_anomaly_detect():
    anomaly_detect_request = request.get_json()
    # 异常检测的数据
    history_data = anomaly_detect_request["history_data"] if "history_data" in anomaly_detect_request else None
    data = anomaly_detect_request["data"] if "data" in anomaly_detect_request else None
    # 异常检测配置
    config = anomaly_detect_request["config"]
    detector = config["detector"] if "detector" in anomaly_detect_request else "default"
    # 从指标库中获取检测数据
    if history_data is None:
        fqm = config["fqm"] if "fqm" in config else None
        starttime = config["starttime"] if "starttime" in config else None
        endtime = config["endtime"] if "endtime" in config else None
        ts = load_single_metric_data_from_clickhouse(fqm, starttime, endtime)
        history_data = ts.values
        if data is None:
            data = history_data
    history_data = np.array(history_data).reshape(-1, 1)
    data = np.array(data).reshape(-1, 1)
    labels, conf = metric_simple_model_anomaly_detect(detector, history_data, data)
    return {
        "label": labels.tolist(),
        "conf": conf.tolist()
    }
