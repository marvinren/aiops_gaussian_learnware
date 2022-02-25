from datetime import datetime

import pandas as pd
from flask import current_app
from clickhouse_driver import Client


def save_metric_datum_to_clickhouse(datas):
    client = Client(host=current_app.config["METRIC_CLICKHOUSE_HOST"],
                    port=current_app.config["METRIC_CLICKHOUSE_PORT"],
                    database='default',
                    user=current_app.config["METRIC_CLICKHOUSE_USERNAME"],
                    password=current_app.config["METRIC_CLICKHOUSE_PASSWORD"],
                    send_receive_timeout=5)
    insert_datas = [{"m_fqm": d["fqm"],
                     "m_metric": d["metric"],
                     "m_source": d["source"],
                     "m_datetime": datetime.strptime(d["timestamp"], "%Y-%m-%d %H:%M:%S"),
                     "m_value": d["value"]} for d in datas]
    client.execute("INSERT INTO aiops_metric_ods(m_fqm, m_metric, m_source, m_datetime, m_value) VALUES",
                   insert_datas)


def load_single_metric_data_from_clickhouse(fqm, starttime, endtime):
    client = Client(host=current_app.config["METRIC_CLICKHOUSE_HOST"],
                    port=current_app.config["METRIC_CLICKHOUSE_PORT"],
                    database='default',
                    user=current_app.config["METRIC_CLICKHOUSE_USERNAME"],
                    password=current_app.config["METRIC_CLICKHOUSE_PASSWORD"],
                    send_receive_timeout=5)
    datum = client.execute(
        "select m_datetime, m_value from aiops_metric_ods where m_fqm = %(fqm)s and m_datetime>=toDateTime(%(starttime)s) and m_datetime<=toDateTime(%(endtime)s)",
        {
            "fqm": fqm,
            "starttime": starttime,
            "endtime": endtime
        })
    values = [d[1] for d in datum]
    ts = [d[0] for d in datum]
    return pd.Series(data=values, index=ts)
