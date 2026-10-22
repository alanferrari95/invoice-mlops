"""Extraction evaluation pipeline (KFP v2, Vertex AI Pipelines): the prompt-version gate.

gold set (GCS) -> evaluate_extract: run Gemini with prompt `prompt_version` on every gold
document, per-field precision/recall -> gate (micro precision, micro recall, no regression
vs. the prompt version currently holding the alias)
-> pass: new version of `invoice-extractor` in Model Registry (artifact = prompt + report,
   label prompt_version), alias moved; the API reads its prompt version from there
-> fail: Pub/Sub "rejected" message, nothing registered.

    PYTHONPATH=. python pipelines/extract_eval_pipeline.py      # compile
"""

import os
from typing import NamedTuple

from kfp import compiler, dsl

from src import settings

PROJECT = settings.PROJECT
REGION = settings.REGION
PIPELINE_IMAGE = os.getenv("PIPELINE_IMAGE", f"{settings.IMAGE_REPO}/train:latest")
OUTPUT = os.getenv("EXTRACT_PIPELINE_YAML", "pipelines/extract_eval.yaml")


def component(fn):
    return dsl.component(fn, base_image=PIPELINE_IMAGE, install_kfp_package=False)


@component
def evaluate_extract(
    gold: dsl.Input[dsl.Dataset],
    prompt_version: str,
    gemini_model: str,
    registry_model: str,
    alias: str,
    min_precision: float,
    min_recall: float,
    max_regression: float,
    metrics: dsl.Output[dsl.Metrics],
    report: dsl.Output[dsl.Artifact],
) -> NamedTuple("ExtractEvalOutputs", [("decision", str), ("reason", str), ("micro_f1", float)]):
    import json
    from collections import namedtuple
    from pathlib import Path

    from src import registry
    from src.eval.evaluate_extract import evaluate
    from src.eval.extract_metrics import decide_prompt
    from src.extract.extractor import Extractor, make_backend

    extractor = Extractor(make_backend("vertex", model=gemini_model), prompt_version=prompt_version)
    result = evaluate(extractor, Path(gold.path), price_per_token=True)
    Path(report.path).mkdir(parents=True, exist_ok=True)
    (Path(report.path) / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    micro = result["report"]["micro"]
    for name, values in result["report"]["fields"].items():
        metrics.log_metric(f"{name}_precision", values["precision"])
        metrics.log_metric(f"{name}_recall", values["recall"])
    for key in ("precision", "recall", "f1"):
        metrics.log_metric(f"micro_{key}", micro[key])
    for key in ("avg_prompt_tokens", "avg_output_tokens", "latency_p95_ms", "usd_per_1000_docs_launch"):
        metrics.log_metric(key, result["usage"][key])
    metrics.log_metric("docs_failed", result["usage"]["docs_failed"])

    champion_f1 = None
    champion = registry.resolve(registry_model, alias) if registry.model_exists(registry_model) else None
    if champion is not None:
        champion_report = Path(champion.artifact_uri.replace("gs://", "/gcs/", 1)) / "report.json"
        if champion_report.is_file():
            champion_f1 = json.loads(champion_report.read_text())["report"]["micro"]["f1"]
            metrics.log_metric("champion_micro_f1", champion_f1)
            metrics.log_metric("champion_version", int(champion.version_id))

    d = decide_prompt(micro, champion_f1, min_precision, min_recall, max_regression)
    print(d.reason)
    out = namedtuple("ExtractEvalOutputs", ["decision", "reason", "micro_f1"])
    return out("promote" if d.promote else "reject", d.reason, micro["f1"])


@component
def register_prompt(
    report: dsl.Input[dsl.Artifact],
    prompt_version: str,
    gemini_model: str,
    registry_model: str,
    alias: str,
    serving_image: str,
    micro_f1: float,
    reason: str,
    pipeline_job: str,
    bundle: dsl.Output[dsl.Artifact],
) -> str:
    import shutil
    from pathlib import Path

    from src.extract.extractor import PROMPTS_DIR
    from src.registry import register_version

    out = Path(bundle.path)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROMPTS_DIR / f"{prompt_version}.txt", out / "prompt.txt")
    shutil.copy(Path(report.path) / "report.json", out / "report.json")

    version = register_version(
        model=registry_model,
        artifact_uri=bundle.uri,
        alias=alias,
        serving_image=serving_image,
        description=f"prompt {prompt_version} on {gemini_model}: {reason}",
        labels={"prompt_version": prompt_version, "gemini_model": gemini_model, "pipeline_job": pipeline_job},
        display_name="invoice-extractor",
    )
    bundle.metadata.update({"version_id": version.version_id, "prompt_version": prompt_version, "micro_f1": micro_f1})
    return version.version_id


@component
def notify_prompt_rejected(topic: str, prompt_version: str, reason: str, micro_f1: float, pipeline_job: str):
    import json

    from google.cloud import pubsub_v1

    payload = {"state": "REJECTED", "prompt_version": prompt_version, "reason": reason,
               "micro_f1": micro_f1, "pipeline_job": pipeline_job}
    pubsub_v1.PublisherClient().publish(topic, json.dumps(payload).encode("utf-8"), kind="extract-gate").result()


@dsl.pipeline(name="extract-eval")
def extract_eval(
    prompt_version: str = "v2",
    gemini_model: str = "gemini-3.8-flash",
    gold_uri: str = f"gs://{settings.RAW_BUCKET}/gold/extract",
    min_precision: float = 0.85,
    min_recall: float = 0.80,
    max_regression: float = 0.02,
    registry_model: str = settings.EXTRACTOR_MODEL,
    alias: str = "production",
    serving_image: str = f"{settings.IMAGE_REPO}/api:latest",
    topic: str = settings.PIPELINE_ALERTS_TOPIC,
):
    gold = dsl.importer(artifact_uri=gold_uri, artifact_class=dsl.Dataset, reimport=False).set_display_name("gold-set")
    evaluated = evaluate_extract(
        gold=gold.output,
        prompt_version=prompt_version,
        gemini_model=gemini_model,
        registry_model=registry_model,
        alias=alias,
        min_precision=min_precision,
        min_recall=min_recall,
        max_regression=max_regression,
    ).set_display_name("evaluate-extract")
    evaluated.set_caching_options(False)

    with dsl.If(evaluated.outputs["decision"] == "promote", name="gate-passed"):
        register_prompt(
            report=evaluated.outputs["report"],
            prompt_version=prompt_version,
            gemini_model=gemini_model,
            registry_model=registry_model,
            alias=alias,
            serving_image=serving_image,
            micro_f1=evaluated.outputs["micro_f1"],
            reason=evaluated.outputs["reason"],
            pipeline_job=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        ).set_display_name("register-prompt").set_caching_options(False)
    with dsl.Else(name="gate-failed"):
        notify_prompt_rejected(
            topic=topic,
            prompt_version=prompt_version,
            reason=evaluated.outputs["reason"],
            micro_f1=evaluated.outputs["micro_f1"],
            pipeline_job=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        ).set_display_name("notify-rejected")


if __name__ == "__main__":
    compiler.Compiler().compile(extract_eval, OUTPUT)
    print(f"compiled -> {OUTPUT} (image {PIPELINE_IMAGE})")
