"""Submit the extraction gate for one prompt version.

    python pipelines/submit_extract_eval.py --prompt-version v2
    python pipelines/submit_extract_eval.py --prompt-version v1     # expected to be rejected
"""

import argparse

from google.cloud import aiplatform

PROJECT = "invoice-mlops"
LOCATION = "europe-west4"

parser = argparse.ArgumentParser()
parser.add_argument("--prompt-version", required=True)
parser.add_argument("--template", default="pipelines/extract_eval.yaml")
args = parser.parse_args()

aiplatform.init(project=PROJECT, location=LOCATION)
job = aiplatform.PipelineJob(
    display_name=f"extract-eval-{args.prompt_version}",
    template_path=args.template,
    pipeline_root="gs://invoice-mlops-artifacts/pipeline-root",
    parameter_values={"prompt_version": args.prompt_version},
    labels={"pipeline": "extract-eval", "prompt_version": args.prompt_version},
    enable_caching=False,
)
job.submit(service_account=f"invoice-mlops-work@{PROJECT}.iam.gserviceaccount.com")
print(job.resource_name)
