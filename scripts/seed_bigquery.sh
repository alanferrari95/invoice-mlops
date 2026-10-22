#!/usr/bin/env bash
# Load the raw synthetic invoices (no derived features) into BigQuery.
# The pipeline computes features from this table with src/features/features.sql.
set -euo pipefail
TABLE="${TABLE:-invoice-mlops:invoice_mlops.invoices}"
SRC="${1:-data/invoices.csv}"
TMP="$(mktemp --suffix=.csv)"

python3 - "$SRC" "$TMP" <<'PY'
import sys
import pandas as pd
cols = ["invoice_id", "vendor_id", "vendor_name", "amount", "lines_sum", "hour", "is_new_vendor", "risk"]
pd.read_csv(sys.argv[1])[cols].to_csv(sys.argv[2], index=False)
PY

bq load --replace --source_format=CSV --skip_leading_rows=1 "$TABLE" "$TMP" \
  invoice_id:INTEGER,vendor_id:INTEGER,vendor_name:STRING,amount:FLOAT,lines_sum:FLOAT,hour:INTEGER,is_new_vendor:INTEGER,risk:INTEGER
bq query --use_legacy_sql=false --format=pretty \
  "SELECT COUNT(*) AS n, ROUND(AVG(risk), 4) AS risk_rate FROM \`${TABLE/:/.}\`"
rm -f "$TMP"
