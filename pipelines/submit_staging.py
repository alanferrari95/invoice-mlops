"""Submit the risk pipeline after a merge to main (called from cloudbuild.yaml).

Staging runs exercise the new code end to end and register the result under the
`staging` alias; only scheduled / drift-triggered runs move `production`.
"""

import os
import sys
from pathlib import Path

from google.cloud import aiplatform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import settings  # noqa: E402

TEMPLATE = os.getenv("PIPELINE_YAML", "pipelines/risk_train_eval.yaml")

aiplatform.init(project=settings.PROJECT, location=settings.REGION)

job = aiplatform.PipelineJob(
    display_name="risk-train-eval-staging",
    template_path=TEMPLATE,
    pipeline_root=settings.PIPELINE_ROOT,
    parameter_values={"alias": "staging"},
    labels={"pipeline": "risk-train-eval", "trigger": "merge", "commit": os.getenv("SHORT_SHA", "local").lower()},
    enable_caching=True,
)
job.submit(service_account=settings.WORK_SA)
print(job.resource_name)
