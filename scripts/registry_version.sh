#!/usr/bin/env bash
# Print the Model Registry version id that holds an alias (default: production).
#   ./scripts/registry_version.sh            -> e.g. 4
#   ./scripts/registry_version.sh staging
set -euo pipefail
ALIAS="${1:-production}"
MODEL_ID="${MODEL_ID:-risk-classifier}"
REGION="${REGION:-europe-west4}"
PROJECT="${PROJECT:-${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}}"

gcloud ai models list-version "$MODEL_ID" --project="$PROJECT" --region="$REGION" --format=json 2>/dev/null \
  | python3 -c '
import json, sys
alias = sys.argv[1]
matches = [v["versionId"] for v in json.load(sys.stdin) if alias in v.get("versionAliases", [])]
if not matches:
    sys.exit(f"no version holds alias {alias!r}")
print(matches[0])
' "$ALIAS"
