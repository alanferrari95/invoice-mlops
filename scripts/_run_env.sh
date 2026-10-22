# Shared settings for the release scripts. Override with env vars.
PROJECT="${PROJECT:-invoice-mlops}"
REGION="${REGION:-europe-west4}"
SERVICE="${SERVICE:-invoice-mlops-api}"

run_describe() {
  gcloud run services describe "$SERVICE" --project="$PROJECT" --region="$REGION" "$@"
}

# Revision currently carrying the most traffic (the stable one in an 80/20 split).
stable_revision() {
  run_describe --flatten=status.traffic \
    --format='csv[no-heading](status.traffic.revisionName,status.traffic.percent)' \
    | awk -F, '$2 != "" {print $2, $1}' | sort -nr | head -1 | cut -d' ' -f2
}

show_traffic() {
  run_describe --format='yaml(status.traffic)'
}
