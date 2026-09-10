from kfp import compiler, dsl

TRAIN_IMAGE = "europe-west4-docker.pkg.dev/invoice-mlops/invoice-mlops/train:week4"


@dsl.container_component
def train(data_uri: str, model_dir: str, n_estimators: int, max_depth: int):
    return dsl.ContainerSpec(
        image=TRAIN_IMAGE,
        args=[
            "--data-uri",
            data_uri,
            "--model-dir",
            model_dir,
            "--n-estimators",
            n_estimators,
            "--max-depth",
            max_depth,
        ],
    )


@dsl.component(base_image="python:3.12-slim", packages_to_install=["google-cloud-storage"])
def evaluate(model_dir: str, min_f1: float) -> str:
    import json
    from pathlib import Path

    from google.cloud import storage

    prefix = model_dir.rstrip("/")
    rest = prefix[len("gs://") :]
    bucket_name, _, blob_prefix = rest.partition("/")
    blob_name = f"{blob_prefix}/metrics.json" if blob_prefix else "metrics.json"

    local_path = Path("/tmp/metrics.json")
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(str(local_path))
    metrics = json.loads(local_path.read_text(encoding="utf-8"))
    f1 = float(metrics["f1"])
    if f1 < min_f1:
        raise RuntimeError(f"f1={f1} below min_f1={min_f1}")
    return f"f1={f1}"


@dsl.component(base_image="python:3.12-slim", packages_to_install=["google-cloud-pubsub"])
def notify_if_failed(status: dsl.PipelineTaskFinalStatus, topic: str, min_f1: float):
    import json
    from google.cloud import pubsub_v1

    if status.state == "SUCCEEDED":
        return

    publisher = pubsub_v1.PublisherClient()
    payload = {
        "state": status.state,
        "pipeline_job_resource_name": status.pipeline_job_resource_name,
        "error_code": status.error_code,
        "error_message": status.error_message,
        "min_f1": min_f1,
    }
    publisher.publish(topic, json.dumps(payload).encode("utf-8")).result()



@dsl.pipeline(name="risk-train-eval")
def risk_train_eval(
    data_uri: str = "gs://invoice-mlops-raw/raw/invoices.csv",
    model_dir: str = "gs://invoice-mlops-artifacts/models/pipeline-week7",
    n_estimators: int = 100,
    max_depth: int = 4,
    min_f1: float = 0.5,
    topic: str = "projects/invoice-mlops/topics/invoice-mlops-pipeline-alerts",
):
    exit_task = notify_if_failed(topic=topic, min_f1=min_f1)
    with dsl.ExitHandler(exit_task):
        t = train(
            data_uri=data_uri,
            model_dir=model_dir,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        t.set_display_name("train")
        evaluate(
            model_dir=model_dir,
            min_f1=min_f1,
        ).after(t)


if __name__ == "__main__":
    compiler.Compiler().compile(risk_train_eval, "pipelines/risk_train_eval.yaml")
