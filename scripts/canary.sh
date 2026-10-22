#!/usr/bin/env bash
# Send PERCENT (default 20) of traffic to the newest revision, the rest to the stable one.
# No rebuild: it only moves traffic between revisions that already exist.
#   ./scripts/canary.sh            # newest revision at 20%
#   ./scripts/canary.sh REV 50     # a given revision at 50%
set -euo pipefail
source "$(dirname "$0")/_run_env.sh"

CANDIDATE="${1:-$(run_describe --format='value(status.latestCreatedRevisionName)')}"
PERCENT="${2:-20}"
STABLE="$(stable_revision)"

if [[ "$CANDIDATE" == "$STABLE" ]]; then
  echo "Candidate $CANDIDATE already is the stable revision; nothing to do."
  exit 0
fi

echo "stable=$STABLE canary=$CANDIDATE at ${PERCENT}%"
gcloud run services update-traffic "$SERVICE" --project="$PROJECT" --region="$REGION" \
  --to-revisions="${STABLE}=$((100 - PERCENT)),${CANDIDATE}=${PERCENT}"
show_traffic
