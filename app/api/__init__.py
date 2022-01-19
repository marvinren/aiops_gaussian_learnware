from flask import Blueprint

api = Blueprint("api", __name__, url_prefix="/api")

from . import anomaly_detect
from . import metric
