from clickhouse_driver import Client
from flask import current_app
import pandas as pd

def load_metric_history_data(fqm, starttime, endtime)->pd.Series:
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