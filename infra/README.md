# infra/

Terraform for the whole project. State in GCS (`<project>-tfstate`, versioned).

| File | What |
|---|---|
| `versions.tf` | Terraform/provider versions, GCS backend |
| `variables.tf` | project, region, GitHub repo, Registry model id, API min instances |
| `main.tf` | APIs, buckets, Artifact Registry, work service account |
| `data.tf` | BigQuery dataset, `invoices`, `predictions`, `extractions` |
| `serving.tf` | Cloud Run service (identity, scaling, public access) |
| `monitoring.tf` | Pub/Sub topics, drift + launcher Cloud Run Jobs, Cloud Scheduler |
| `iam.tf` | least-privilege grants for the four service accounts |
| `cicd.tf` | Cloud Build triggers (PR, main, model promotion) |
| `imports.tf` | resources created by hand before Terraform covered them |

## Existing project (`invoice-mlops`)

```bash
terraform -chdir=infra init -backend-config=bucket=invoice-mlops-tfstate
terraform -chdir=infra plan      # imports + the new resources; nothing is destroyed
terraform -chdir=infra apply
```

## Empty project

```bash
./infra/bootstrap.sh <project_id>                 # state bucket
rm infra/imports.tf                               # nothing to import
terraform -chdir=infra init -backend-config=bucket=<project_id>-tfstate
terraform -chdir=infra apply -var=project_id=<project_id>
```

Then connect the GitHub repo to Cloud Build (GitHub App, one click in the console; it cannot be done from Terraform for 1st-gen triggers), push to `main` once so images exist, run the risk pipeline once with alias `production` and the extract pipeline once, and the service is complete. The Registry model id variable must point at the `risk-classifier` model the first upload creates.

Not in Terraform on purpose: the state bucket (chicken-and-egg), the billing budget (billing-account scope, created in the console in week 0), and what Cloud Build owns at runtime (images, traffic split, Registry versions).
