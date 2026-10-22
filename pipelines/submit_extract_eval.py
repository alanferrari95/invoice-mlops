"""Submit the extraction gate for one prompt version.

    python pipelines/submit_extract_eval.py --prompt-version v2
    python pipelines/submit_extract_eval.py --prompt-version v1     # expected to be rejected
"""

import argparse
import sys
from pathlib import Path

from google.cloud import aiplatform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import settings  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--prompt-version", required=True)
parser.add_argument("--template", default="pipelines/extract_eval.yaml")
args = parser.parse_args()

aiplatform.init(project=settings.PROJECT, location=settings.REGION)
job = aiplatform.PipelineJob(
    display_name=f"extract-eval-{args.prompt_version}",
    template_path=args.template,
    pipeline_root=settings.PIPELINE_ROOT,
    parameter_values={"prompt_version": args.prompt_version},
    labels={"pipeline": "extract-eval", "prompt_version": args.prompt_version},
    enable_caching=False,
)
job.submit(service_account=settings.WORK_SA)
print(job.resource_name)
