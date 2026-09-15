from google.cloud import aiplatform

PROJECT = "invoice-mlops"
LOCATION = "europe-west4"
PIPELINE_ROOT = "gs://invoice-mlops-artifacts/pipeline-root"
TEMPLATE = "pipelines/risk_train_eval.yaml"
RUNTIME_SA = "invoice-mlops-work@invoice-mlops.iam.gserviceaccount.com"

aiplatform.init(project=PROJECT, location=LOCATION)

job = aiplatform.PipelineJob(
    display_name="risk-train-eval-staging",
    template_path=TEMPLATE,
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "data_uri": "gs://invoice-mlops-raw/raw/invoices.csv",
        "model_dir": "gs://invoice-mlops-artifacts/models/pipeline-staging",
        "n_estimators": 100,
        "max_depth": 4,
        "min_f1": 0.5,
        "topic": "projects/invoice-mlops/topics/invoice-mlops-pipeline-alerts",
    },
    enable_caching=True,
)
job.submit(service_account=RUNTIME_SA)
print(job.resource_name)