from app.ingest.base import BaseIngest
import json


class PrometheusIngest(BaseIngest):

    def __init__(self, ingest_model, **kwargs):
        super().__init__(ingest_model)

    def parse(self, message):
        data = json.loads(message)
        app = data["labels"]["app"] if "labels" in data and "app" in data["labels"] else "unknown"
        domain = data["host_code"] if "host_code" in data else "all"
        metric = {
            "fqm": f'{data["__name__"]}.{domain}.{data["instance"]}',
            "ts": data["Timestamp"],
            "metric": data["__name__"],
            "source": data["instance"],
            "app": app,
            "domain": domain,
            "labels": data["labels"] if "labels" in data else None,
            "value": data["value"]
        }
        return metric
