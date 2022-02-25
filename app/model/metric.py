from app import db


# class MetricDatum(db.Model):
#     __tablename__ = "aiops_metric_datum"
#     id = db.Column(db.Integer, primary_key=True, nullable=False, autoincrement=True, comment="ID")
#
#     fqm = db.Column(db.String, comment="指标唯一标识")
#     ts = db.Column(db.DateTime, comment="指标采集时间")
#     metric = db.Column(db.String, comment="指标名称")
#     source = db.Column(db.String, comment="指标来源")
#     app = db.Column(db.String, comment="指标所属业务")
#     domain = db.Column(db.String, comment="指标所属域")
#     labels = db.Column(db.JSON, comment="标签")
#     value = db.Column(db.Float, comment="指标的值")


class Metric(db.Model):
    __tablename__ = "aiops_metric"

    id = db.Column(db.Integer, primary_key=True, nullable=False, autoincrement=True, comment="ID")
    code = db.Column(db.String, nullable=False, comment="指标的编码")
    name = db.Column(db.String, comment="指标的名称")
    source = db.Column(db.String, comment="指标的来源")
    tags = db.Column(db.String, comment="标签")
    notes = db.Column(db.String, comment="指标的描述")
    datum_site = db.Column(db.String, comment="指标数据的位置")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return '%r:%r' % (self.code, self.name)
