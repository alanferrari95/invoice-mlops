#!/usr/bin/env bash
# One-time setup before the first `terraform init` in a project: the state bucket.
# It is the only resource not managed by Terraform (it holds Terraform's own state).
#   ./infra/bootstrap.sh <project_id> [region]
set -euo pipefail
PROJECT="${1:?project id}"
REGION="${2:-europe-west4}"
BUCKET="gs://${PROJECT}-tfstate"

gcloud storage buckets describe "$BUCKET" --project="$PROJECT" >/dev/null 2>&1 || \
  gcloud storage buckets create "$BUCKET" --project="$PROJECT" --location="$REGION" \
    --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets update "$BUCKET" --versioning
echo "terraform -chdir=infra init -backend-config=bucket=${PROJECT}-tfstate"
