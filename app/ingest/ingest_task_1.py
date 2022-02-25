from app.ingest.prometheus import PrometheusIngest
from app.model.ingest import Ingest
from multiprocessing import Process


def task_start():
    ingest_model = Ingest(name="test_kafka_ingest", type="prometheus",
                          servers="aiops01:6667, aiops02:6667, aiops03:6667",
                          topic_name="metrics")
    ingest = PrometheusIngest(ingest_model)
    ingest.run()


if __name__ == "__main__":
    p = Process(target=task_start)
    p.start()
    print("================start================")
    p.join()
