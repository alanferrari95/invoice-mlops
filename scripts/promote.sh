#!/usr/bin/env bash
# Move 100% of traffic to the canary (newest revision, or the one given).
set -euo pipefail
source "$(dirname "$0")/_run_env.sh"

CANDIDATE="${1:-$(run_describe --format='value(status.latestCreatedRevisionName)')}"
gcloud run services update-traffic "$SERVICE" --project="$PROJECT" --region="$REGION" \
  --to-revisions="${CANDIDATE}=100"
show_traffic
