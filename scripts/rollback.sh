#!/usr/bin/env bash
set -euo pipefail
# set -e: exit si un comando falla
# -u: exit si usás una variable no definida
# -o pipefail: el pipeline falla si falla cualquier eslabón

PROJECT=invoice-mlops
REGION=europe-west4
SERVICE=invoice-mlops-api
STABLE=invoice-mlops-api-00007-7c4

gcloud run services update-traffic "$SERVICE" \
  --project="$PROJECT" \
  --region="$REGION" \
  --to-revisions="${STABLE}=100"

gcloud run services describe "$SERVICE" \
  --project="$PROJECT" \
  --region="$REGION" \
  --format='yaml(status.traffic)'
