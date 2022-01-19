from app import db


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