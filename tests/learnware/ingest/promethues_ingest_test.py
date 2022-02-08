from app.ingest.prometheus import PrometheusIngest
from app.model.ingest import Ingest


class TestPrometheusIngest:
    def test_prometheus_data_parse(self):
        message = """
        {
            "Timestamp":1644290320527,
            "__name__":"scrape_series_added",
            "host_code":"host05",
            "instance":"192.168.48.15",
            "labels":{"group":"docker","job":"docker_cadvisor"},
            "value":0
        }
        """
        ingest_model = Ingest()
        ingest = PrometheusIngest(ingest_model)
        metric = ingest.parse(message)

        print(metric)

        assert metric["fqm"] == "scrape_series_added.host05.192.168.48.15"
        assert metric["ts"] == 1644290320527
        assert metric["source"] == '192.168.48.15'
        assert metric['domain'] == 'host05'
        assert metric['value'] == 0
        assert metric['app'] == 'unknown'
        assert metric["labels"] is not None

    # 需要kafka和clickhouse，会进入持续执行，暂时作废
    # def test_ingest_kafka_data(self):
    #     ingest_model = Ingest(name="test_kafka_ingest", type="prometheus",
    #                           servers="aiops01:6667, aiops02:6667, aiops03:6667",
    #                           topic_name="metrics")
    #     ingest = PrometheusIngest(ingest_model)
    #     ingest.run()
