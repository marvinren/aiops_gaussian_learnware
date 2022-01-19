# coding:utf-8
import os
import logging
import time
from logging.handlers import TimedRotatingFileHandler

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# 数据库
from werkzeug.utils import import_string

from config import app_config

db = SQLAlchemy()

# 日志文件配置
if not os.path.exists('logs'):
    os.mkdir('logs')
log_handler = TimedRotatingFileHandler('logs/learnwares.log', when='D')
logging_format = logging.Formatter(
    '[%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - line:%(lineno)s]: %(message)s')
log_handler.setFormatter(logging_format)


def create_app(env='development'):
    """app程序工厂"""

    # 应用启动
    app = Flask(__name__)
    app.uptime = time.time()

    # 加载配置文件
    app.config.from_object(app_config.config[env])

    # 配置应用对应的日志
    app.logger.addHandler(log_handler)

    # 初始化插件
    db.init_app(app)

    # 加载蓝图
    blueprints = ['app.api:api']
    for bp_name in blueprints:
        app.register_blueprint(import_string(bp_name))

    app.logger.info("application started on %s, found %s app", env, len(list(app.url_map.iter_rules())) - 1)
    return app