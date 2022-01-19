from app import db
from sqlalchemy import Column, String, ForeignKey


class Tag(db.Model):
    __tablename__ = "tag"

    t_id = db.Column(db.Integer, primary_key=True, nullable=False, autoincrement=True, comment="ID")
    name = db.Column(db.String(255), comment="标签名称")
    value = db.Column(db.String(1024), comment="标签值")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return '%r:%r' % (self.t_name, self.t_value)


class Event(db.Model):
    __tablename__ = "event"

    e_id = db.Column(db.Integer, primary_key=True, nullable=False, autoincrement=True, comment="ID")
    description = db.Column(db.String(1024), nullable=False, comment="事件描述")
    severity = db.Column(db.String(128), comment="重要程度")
    source = db.Column(db.String(255), comment="事件来源")
    check = db.Column(db.String(255), nullable=False, comment="事件对象")
    eclass = db.Column(db.String(128), comment="事件分类")
    type = db.Column(db.String(128), comment="事件大分类")
    time = db.Column(db.DateTime, comment="事件发生事件")
    tags = db.Column(db.String(1024), comment="事件标签")
    location = db.Column(db.String(1024), comment="事件位置")
    services = db.Column(db.String(1024), comment="事件服务")
    manager = db.Column(db.String(128), comment="管理方")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return '<Event %r>' % self.description


class Alarm(db.Model):
    __tablename__ = "alarm"
    a_id = db.Column(db.Integer, primary_key=True, nullable=False, autoincrement=True, comment="ID")
    dedupe_key = db.Column(db.String(1024), comment="重复合并的Key")
    description = db.Column(db.String(1024), comment="描述")
    #check = db.Column(db.String(128), comment="告警发起组件")
    event_count = db.Column(db.Integer, comment="事件的数量")
    event_type = db.Column(db.String, comment="事件的类型")
    first_event_time = db.Column(db.DateTime, comment="第一次事件发生的时间")
    last_event_time = db.Column(db.DateTime, comment="最后一次事件发生的时间")
    incidents = db.Column(db.String(1024), comment="相关的故障")
    service = db.Column(db.ARRAY(db.String(1024)), comment="服务列表")
    service_count = db.Column(db.Integer, comment="服务的数量")
    source = db.Column(db.String(1024), comment="来源")
    status = db.Column(db.String(1024), comment="状态")
    last_status_change_time = db.Column(db.DateTime, comment="状态的最后时间")
    severity = db.Column(db.String(255), comment="严重程度")

    tags = db.relationship("Tag", backref="alarms", secondary="alarm_tag_rel")


class AlarmTag(db.Model):
    __tablename__ = "alarm_tag_rel"

    a_t_id = db.Column(db.Integer, primary_key=True, nullable=False, autoincrement=True, comment="ID")
    a_id = db.Column(db.Integer, ForeignKey("alarm.a_id"))
    t_id = db.Column(db.Integer, ForeignKey("tag.t_id"))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return '<AlarmTag %r - %r>' % (self.a_id, self.t_id)