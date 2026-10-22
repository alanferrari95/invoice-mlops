"""Submit the risk pipeline after a merge to main (called from cloudbuild.yaml).

Staging runs exercise the new code end to end and register the result under the
`staging` alias; only scheduled / drift-triggered runs move `production`.
"""

import os

from google.cloud import aiplatform

PROJECT = "invoice-mlops"
LOCATION = "europe-west4"
PIPELINE_ROOT = "gs://invoice-mlops-artifacts/pipeline-root"
TEMPLATE = os.getenv("PIPELINE_YAML", "pipelines/risk_train_eval.yaml")
RUNTIME_SA = "invoice-mlops-work@invoice-mlops.iam.gserviceaccount.com"

aiplatform.init(project=PROJECT, location=LOCATION)

job = aiplatform.PipelineJob(
    display_name="risk-train-eval-staging",
    template_path=TEMPLATE,
    pipeline_root=PIPELINE_ROOT,
    parameter_values={"alias": "staging"},
    labels={"trigger": "merge", "commit": os.getenv("SHORT_SHA", "local").lower()},
    enable_caching=True,
)
job.submit(service_account=RUNTIME_SA)
print(job.resource_name)
