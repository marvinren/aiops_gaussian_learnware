# coding: utf-8
from urllib import parse


class Config:
    APP_NAME = "aiops_gaussian_learnware"
    APP_VERSION = "0.1"
    APP_UPDATE_TIME = "2021-10-12 13:37:00"

    # 暂时不用
    SECRET_KEY = "marvinrencn@gmail.com"

    SQLALCHEMY_POOL_SIZE = 100  # 连接池个数
    SQLALCHEMY_POOL_TIMEOUT = 30  # 超时时间，秒
    SQLALCHEMY_POOL_RECYCLE = 3600  # 空连接回收时间，秒
    SQLALCHEMY_TRACK_MODIFICATIONS = True


class DevelopmentConfig(Config):
    ENV = "development"
    DEBUG = True
    LOG_LEVEL = 10
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:mysql%40123@192.168.48.10:3306/yusys_aiops_learnware?charset=utf8mb4"
    HDFS_WAREHOUSE_URI = "aiops02:50070"
    HDFS_USERNAME = "hdfs"
    HDFS_ROOT = "/aiops/learnware"
    TEMP_FILE_PATH = "./temp"
    METRIC_CLICKHOUSE_HOST = "192.168.48.12"
    METRIC_CLICKHOUSE_PORT = 9000
    METRIC_CLICKHOUSE_USERNAME = "aiops"
    METRIC_CLICKHOUSE_PASSWORD = "aiops_2021"


class TestingConfig(Config):
    TESTING = True
    LOG_LEVEL = 10
    SQLALCHEMY_DATABASE_URI = "mysql://pyapi_test_user:pyapi_test_pwd_123@127.0.0.1:3306/yusys_aiops_learnware?charset=utf8mb4"
    HDFS_WAREHOUSE_URI = "192.168.48.12:9000"
    HDFS_USERNAME = "hdfs"


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = "mysql://user_name:user_pwd@127.0.0.1:3306/yusys_aiops_learnware?charset=utf8mb4"
    HDFS_WAREHOUSE_URI = "192.168.48.12:9000"
    HDFS_USERNAME = "hdfs"


config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}
