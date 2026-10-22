"""Risk model training pipeline (KFP v2, runs on Vertex AI Pipelines).

features (BigQuery) -> train (XGBoost) -> evaluate on the gold set vs. the champion
-> if it passes: new version in Vertex AI Model Registry + alias move
   (alias "production" also publishes model-promoted -> canary deploy on Cloud Run)
-> if it does not: Pub/Sub "rejected" message, nothing is registered.
Any task failure publishes a "failed" message from the exit handler.

Every component runs on the same image (Dockerfile.train), so the logic lives in
src/ and is unit-tested there; the components are thin wrappers.
"""

import os
from typing import NamedTuple

from kfp import compiler, dsl

PROJECT = "invoice-mlops"
REGION = "europe-west4"
PIPELINE_IMAGE = os.getenv(
    "PIPELINE_IMAGE", f"{REGION}-docker.pkg.dev/{PROJECT}/invoice-mlops/train:latest"
)
OUTPUT = os.getenv("PIPELINE_YAML", "pipelines/risk_train_eval.yaml")


def component(fn):
    return dsl.component(fn, base_image=PIPELINE_IMAGE, install_kfp_package=False)


@component
def build_features(project: str, source_table: str, features: dsl.Output[dsl.Dataset]):
    from src.features.build_features import query_features

    df = query_features(project, source_table)
    df.to_csv(features.path, index=False)
    features.metadata.update({"rows": len(df), "source_table": source_table, "risk_rate": float(df["risk"].mean())})


@component
def train(
    features: dsl.Input[dsl.Dataset],
    n_estimators: int,
    max_depth: int,
    model: dsl.Output[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
):
    from pathlib import Path

    import pandas as pd

    from src.train.train import fit, save

    clf, holdout = fit(pd.read_csv(features.path), n_estimators, max_depth)
    save(clf, holdout, Path(model.path))
    model.metadata.update({"framework": "xgboost", "n_estimators": n_estimators, "max_depth": max_depth})
    for key in ("accuracy", "precision", "recall", "f1"):
        metrics.log_metric(f"holdout_{key}", round(holdout[key], 4))


@component
def evaluate(
    model: dsl.Input[dsl.Model],
    gold: dsl.Input[dsl.Dataset],
    registry_model: str,
    alias: str,
    min_f1: float,
    max_regression: float,
    metrics: dsl.Output[dsl.Metrics],
) -> NamedTuple("EvalOutputs", [("decision", str), ("reason", str), ("gold_f1", float)]):
    from collections import namedtuple
    from pathlib import Path

    import joblib
    import pandas as pd

    from src import registry
    from src.eval.gate import decide, gold_metrics

    gold_df = pd.read_csv(gold.path)
    candidate = gold_metrics(joblib.load(Path(model.path) / "model.joblib"), gold_df)
    for key, value in candidate.items():
        metrics.log_metric(f"gold_{key}", round(value, 4) if isinstance(value, float) else value)

    champion_f1 = None
    champion = registry.resolve(registry_model, alias)
    if champion is not None:
        champion_path = champion.model_uri.replace("gs://", "/gcs/", 1)
        if Path(champion_path).is_file():
            champion_f1 = gold_metrics(joblib.load(champion_path), gold_df)["f1"]
            metrics.log_metric("champion_gold_f1", round(champion_f1, 4))
            metrics.log_metric("champion_version", int(champion.version_id))

    d = decide(candidate["f1"], champion_f1, min_f1, max_regression)
    print(d.reason)
    out = namedtuple("EvalOutputs", ["decision", "reason", "gold_f1"])
    return out("promote" if d.promote else "reject", d.reason, candidate["f1"])


@component
def register(
    model: dsl.Input[dsl.Model],
    registry_model: str,
    alias: str,
    serving_image: str,
    gold_f1: float,
    reason: str,
    pipeline_job: str,
    promoted_topic: str,
    registered: dsl.Output[dsl.Artifact],
) -> str:
    import json

    from google.cloud import pubsub_v1

    from src.registry import register_version

    version = register_version(
        model=registry_model,
        artifact_uri=model.uri,
        alias=alias,
        serving_image=serving_image,
        description=f"gold_f1={gold_f1:.4f}; {reason}",
        labels={"pipeline_job": pipeline_job},
    )
    registered.uri = f"https://{registry_model.split('/')[3]}-aiplatform.googleapis.com/v1/{registry_model}@{version.version_id}"
    registered.metadata.update({"version_id": version.version_id, "alias": alias, "gold_f1": gold_f1})

    # A new production model is a release: this message triggers cloudbuild-model-deploy.yaml,
    # which deploys a Cloud Run revision pinned to this version as a 20% canary.
    if alias == "production":
        payload = {"model": registry_model, "version_id": version.version_id, "gold_f1": gold_f1}
        pubsub_v1.PublisherClient().publish(
            promoted_topic, json.dumps(payload).encode("utf-8"), version_id=version.version_id
        ).result()
    return version.version_id


@component
def notify_rejected(topic: str, reason: str, gold_f1: float, pipeline_job: str):
    import json

    from google.cloud import pubsub_v1

    payload = {"state": "REJECTED", "reason": reason, "gold_f1": gold_f1, "pipeline_job": pipeline_job}
    pubsub_v1.PublisherClient().publish(topic, json.dumps(payload).encode("utf-8"), kind="gate").result()


@component
def notify_if_failed(status: dsl.PipelineTaskFinalStatus, topic: str):
    import json

    from google.cloud import pubsub_v1

    if status.state == "SUCCEEDED":
        return
    payload = {
        "state": status.state,
        "pipeline_job_resource_name": status.pipeline_job_resource_name,
        "error_code": status.error_code,
        "error_message": status.error_message,
    }
    pubsub_v1.PublisherClient().publish(topic, json.dumps(payload).encode("utf-8"), kind="failure").result()


@dsl.pipeline(name="risk-train-eval")
def risk_train_eval(
    project: str = PROJECT,
    source_table: str = f"{PROJECT}.invoice_mlops.invoices",
    gold_uri: str = "gs://invoice-mlops-raw/gold/risk_gold.csv",
    n_estimators: int = 100,
    max_depth: int = 4,
    min_f1: float = 0.75,
    max_regression: float = 0.02,
    registry_model: str = f"projects/{PROJECT}/locations/{REGION}/models/429981614227849216",
    alias: str = "production",
    serving_image: str = "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest",
    topic: str = f"projects/{PROJECT}/topics/invoice-mlops-pipeline-alerts",
    promoted_topic: str = f"projects/{PROJECT}/topics/invoice-mlops-model-promoted",
):
    exit_task = notify_if_failed(topic=topic)
    with dsl.ExitHandler(exit_task, name="train-eval-register"):
        feats = build_features(project=project, source_table=source_table).set_display_name("features-bq")
        gold = dsl.importer(artifact_uri=gold_uri, artifact_class=dsl.Dataset, reimport=False).set_display_name(
            "gold-set"
        )
        trained = train(features=feats.outputs["features"], n_estimators=n_estimators, max_depth=max_depth)
        trained.set_display_name("train-xgboost")
        evaluated = evaluate(
            model=trained.outputs["model"],
            gold=gold.output,
            registry_model=registry_model,
            alias=alias,
            min_f1=min_f1,
            max_regression=max_regression,
        ).set_display_name("evaluate-gold")
        # The champion lookup must see the Registry as it is now, never a cached answer.
        evaluated.set_caching_options(False)

        with dsl.If(evaluated.outputs["decision"] == "promote", name="gate-passed"):
            register(
                model=trained.outputs["model"],
                registry_model=registry_model,
                alias=alias,
                serving_image=serving_image,
                gold_f1=evaluated.outputs["gold_f1"],
                reason=evaluated.outputs["reason"],
                pipeline_job=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
                promoted_topic=promoted_topic,
            ).set_display_name("register-model").set_caching_options(False)
        with dsl.Else(name="gate-failed"):
            notify_rejected(
                topic=topic,
                reason=evaluated.outputs["reason"],
                gold_f1=evaluated.outputs["gold_f1"],
                pipeline_job=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
            ).set_display_name("notify-rejected")


if __name__ == "__main__":
    compiler.Compiler().compile(risk_train_eval, OUTPUT)
    print(f"compiled -> {OUTPUT} (image {PIPELINE_IMAGE})")
