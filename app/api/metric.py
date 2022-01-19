from flask import request

from app.api import api
from app.service import metric_service


@api.route("/integration/metric", methods=["POST"])
def metric_datum_integration():

    datas = request.get_json()
    metric_service.save_metric_datum_to_clickhouse(datas)
    return {
        "ret": 0,
        "message": "指标数据已保存成功"
    }