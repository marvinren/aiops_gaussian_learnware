import traceback
from datetime import datetime

from flask import current_app
from kafka import KafkaConsumer
import json

from app.log import get_logger
from app.model.ingest import Ingest
from clickhouse_driver import Client

from config import app_config
import logging


class BaseIngest:

    def __init__(self, ingest_model: Ingest):
        self.logger = get_logger()
        self.ingest_model = ingest_model

        try:
            self.app_config = current_app.config
        except:
            self.app_config = {}
            obj = app_config.config["default"]
            for key in dir(obj):
                if key.isupper():
                    self.app_config[key] = getattr(obj, key)

    def create_ch_saver(self):
        self.logger.info("获取clickhouse的存储", self.app_config["METRIC_CLICKHOUSE_HOST"])
        return Client(host=self.app_config["METRIC_CLICKHOUSE_HOST"],
                      port=self.app_config["METRIC_CLICKHOUSE_PORT"],
                      database='default',
                      user=self.app_config["METRIC_CLICKHOUSE_USERNAME"],
                      password=self.app_config["METRIC_CLICKHOUSE_PASSWORD"],
                      send_receive_timeout=5)

    def create_kafka_consumer(self):
        bootstrap_servers = [s.strip() for s in self.ingest_model.servers.split(",")]
        group_id = self.ingest_model.name
        topic_name = self.ingest_model.topic_name
        self.logger.info("获取kafka的service", bootstrap_servers)

        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id
        )
        return consumer

    def save_metric(self, saver, metrics):
        insert_datas = [{"m_fqm": metric["fqm"],
                         "m_metric": metric["metric"],
                         "m_source": metric["source"],
                         "m_tags": json.dumps(metric["labels"]) if "labels" in metrics and metrics[
                             "labels"] is not None else "",
                         "m_datetime": datetime.fromtimestamp(metric["ts"] / 1000),
                         "m_value": metric["value"]} for metric in metrics]
        saver.execute("INSERT INTO aiops_metric_ods(m_fqm, m_metric, m_source, m_tags, m_datetime, m_value) VALUES",
                      insert_datas)

    def run(self):
        consumer = self.create_kafka_consumer()
        saver = self.create_ch_saver()
        i = 1
        for message in consumer:
            i += 1
            if i % 1000 == 0:
                self.logger.info("已经读取数据", i)
            try:
                data = self.parse(message.value.decode())
                if data is not None:
                    self.save_metric(saver, [data])
            except Exception as e:
                print(e)
                traceback.print_exc()

    def parse(self, message):
        pass
