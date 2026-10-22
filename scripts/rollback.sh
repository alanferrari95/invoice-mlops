#!/usr/bin/env bash
# One-command rollback, no rebuild: put 100% of traffic back on the stable revision.
#   ./scripts/rollback.sh          # revision with the most traffic right now (stable in 80/20)
#   ./scripts/rollback.sh REV      # an explicit revision
set -euo pipefail
source "$(dirname "$0")/_run_env.sh"

TARGET="${1:-$(stable_revision)}"
echo "rolling back to $TARGET"
gcloud run services update-traffic "$SERVICE" --project="$PROJECT" --region="$REGION" \
  --to-revisions="${TARGET}=100"
show_traffic
