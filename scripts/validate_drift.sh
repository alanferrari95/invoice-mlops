#!/usr/bin/env bash
# Week 11 validation, run by hand: normal traffic must NOT alert, deliberately skewed
# traffic MUST alert and re-trigger the training pipeline. Output goes to docs/evidence.md.
#   ./scripts/validate_drift.sh
set -euo pipefail
source "$(dirname "$0")/_run_env.sh"
URL="${URL:-$(run_describe --format='value(status.url)')}"
N="${N:-150}"

check() {  # $1 = window in minutes
  gcloud run jobs execute drift-check --project="$PROJECT" --region="$REGION" --wait \
    --args="--window-minutes=$1"
  echo "--- drift-check report (last execution):"
  gcloud logging read \
    "resource.type=cloud_run_job AND resource.labels.job_name=drift-check AND jsonPayload.status:*" \
    --project="$PROJECT" --limit=1 --freshness=10m --format=json
}

echo "== 1. normal traffic ($N invoices sampled from the training data)"
python3 scripts/send_traffic.py --url "$URL" --mode normal --n "$N"
check 10

echo "== waiting 11 minutes so the next window only holds skewed traffic"
sleep 660

echo "== 2. skewed traffic (amount x8, night hours, only new vendors)"
python3 scripts/send_traffic.py --url "$URL" --mode skewed --n "$N" --seed 1
check 10

echo "== 3. alert on Pub/Sub"
gcloud pubsub subscriptions pull drift-alerts-debug --project="$PROJECT" --auto-ack --limit=5 \
  --format='value(message.messageId,message.publishTime,message.attributes.kind)'

echo "== 4. pipeline run started by the drift job"
gcloud ai pipeline-jobs list --project="$PROJECT" --region="$REGION" \
  --filter='labels.trigger=drift' --sort-by=~createTime --limit=3 2>/dev/null \
  || echo "(list it in the console: Vertex AI > Pipelines, label trigger=drift)"
