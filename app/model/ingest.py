from app import db
import enum


class IngestStatus(enum.Enum):
    created = 1
    run = 5
    stop = 10
    error = -1
    unknown = 0


class Ingest(db.Model):
    __tablename__ = "aiops_ingest"

    ingest_id = db.Column(db.Integer, primary_key=True, nullable=False, autoincrement=True, comment="ID")
    name = db.Column(db.String(255), nullable=False, comment="数据接收器配置")
    status = db.Column(db.Enum(IngestStatus), default=IngestStatus.created, comment="数据接收器状态")
    type = db.Column(db.String(255), comment="数据接收器类型")
    servers = db.Column(db.String(255), comment="kafka的服务集群")
    topic_name = db.Column(db.String(255), nullable=False, comment="数据接受的topic名称")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return '<Ingest Configuration %r>' % self.name

