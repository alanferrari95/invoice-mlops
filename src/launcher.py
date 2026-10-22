"""Submit the risk training pipeline from the compiled template in GCS.

Runs as the `risk-pipeline-launcher` Cloud Run Job (weekly, from Cloud Scheduler) and is
called by the drift job when drift is detected. Two guards keep retraining sane:
- never submit while another risk-train-eval run is queued or running;
- after a drift-triggered run, wait `cooldown_hours` before the next drift-triggered one,
  so one bad day of traffic does not start a retraining storm.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

PROJECT = "invoice-mlops"
LOCATION = "europe-west4"
PIPELINE = "risk-train-eval"
TEMPLATE_URI = "gs://invoice-mlops-artifacts/pipelines/risk_train_eval.yaml"
PIPELINE_ROOT = "gs://invoice-mlops-artifacts/pipeline-root"
SERVICE_ACCOUNT = f"invoice-mlops-work@{PROJECT}.iam.gserviceaccount.com"
ACTIVE_STATES = {"PIPELINE_STATE_QUEUED", "PIPELINE_STATE_PENDING", "PIPELINE_STATE_RUNNING"}
TRIGGERS = ("schedule", "drift", "manual")


@dataclass
class JobInfo:
    name: str
    state: str
    created: datetime
    labels: dict = field(default_factory=dict)


@dataclass
class LaunchResult:
    submitted: bool
    reason: str
    job: Optional[str] = None


class PipelineApi(Protocol):
    def list_jobs(self) -> list[JobInfo]: ...

    def submit(self, display_name: str, parameters: dict, labels: dict) -> str: ...


class VertexPipelineApi:
    def __init__(self, project: str = PROJECT, location: str = LOCATION,
                 template_uri: str = TEMPLATE_URI, pipeline_root: str = PIPELINE_ROOT,
                 service_account: str = SERVICE_ACCOUNT) -> None:
        from google.cloud import aiplatform

        aiplatform.init(project=project, location=location)
        self._aiplatform = aiplatform
        self.template_uri = template_uri
        self.pipeline_root = pipeline_root
        self.service_account = service_account

    def list_jobs(self) -> list[JobInfo]:
        jobs = self._aiplatform.PipelineJob.list(filter=f'labels.pipeline="{PIPELINE}"')
        return [JobInfo(j.resource_name, j.state.name, j.create_time, dict(j.labels)) for j in jobs]

    def submit(self, display_name: str, parameters: dict, labels: dict) -> str:
        job = self._aiplatform.PipelineJob(
            display_name=display_name,
            template_path=self.template_uri,
            pipeline_root=self.pipeline_root,
            parameter_values=parameters,
            labels=labels,
            enable_caching=False,
        )
        job.submit(service_account=self.service_account)
        return job.resource_name


def should_launch(jobs: list[JobInfo], trigger: str, now: datetime, cooldown_hours: float) -> tuple[bool, str]:
    active = [j for j in jobs if j.state in ACTIVE_STATES]
    if active:
        return False, f"{len(active)} {PIPELINE} run(s) already active: {active[0].name}"
    if trigger == "drift":
        since = now - timedelta(hours=cooldown_hours)
        recent = [j for j in jobs if j.labels.get("trigger") == "drift" and j.created >= since]
        if recent:
            return False, f"drift-triggered run within the last {cooldown_hours:g}h: {recent[0].name}"
    return True, "ok"


def launch(trigger: str, api: PipelineApi, alias: str = "production", cooldown_hours: float = 24.0,
           extra_parameters: Optional[dict] = None, now: Optional[datetime] = None) -> LaunchResult:
    if trigger not in TRIGGERS:
        raise ValueError(f"trigger must be one of {TRIGGERS}, got {trigger!r}")
    now = now or datetime.now(timezone.utc)
    ok, reason = should_launch(api.list_jobs(), trigger, now, cooldown_hours)
    if not ok:
        return LaunchResult(False, reason)
    job = api.submit(
        display_name=f"{PIPELINE}-{trigger}",
        parameters={"alias": alias, **(extra_parameters or {})},
        labels={"pipeline": PIPELINE, "trigger": trigger},
    )
    return LaunchResult(True, f"submitted ({trigger})", job)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger", choices=TRIGGERS, default=os.getenv("LAUNCH_TRIGGER", "schedule"))
    parser.add_argument("--alias", default=os.getenv("MODEL_ALIAS", "production"))
    parser.add_argument("--template-uri", default=os.getenv("PIPELINE_TEMPLATE", TEMPLATE_URI))
    parser.add_argument("--cooldown-hours", type=float, default=24.0)
    args = parser.parse_args(argv)

    result = launch(args.trigger, VertexPipelineApi(template_uri=args.template_uri), args.alias, args.cooldown_hours)
    print(f"submitted={result.submitted} reason={result.reason} job={result.job}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
