# BigQuery: training data, prediction log (Engine A) and extraction log (Engine B).

resource "google_bigquery_dataset" "main" {
  dataset_id = "invoice_mlops"
  location   = var.region
}

resource "google_bigquery_dataset_iam_member" "work_writes" {
  dataset_id = google_bigquery_dataset.main.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = local.work_member
}

# Raw synthetic invoices; the pipeline derives features from it with features.sql.
# Loaded by scripts/seed_bigquery.sh.
resource "google_bigquery_table" "invoices" {
  dataset_id          = google_bigquery_dataset.main.dataset_id
  table_id            = "invoices"
  deletion_protection = true
  schema = jsonencode([
    { name = "invoice_id", type = "INTEGER", mode = "REQUIRED" },
    { name = "vendor_id", type = "INTEGER", mode = "REQUIRED" },
    { name = "vendor_name", type = "STRING", mode = "NULLABLE" },
    { name = "amount", type = "FLOAT", mode = "REQUIRED" },
    { name = "lines_sum", type = "FLOAT", mode = "REQUIRED" },
    { name = "hour", type = "INTEGER", mode = "REQUIRED" },
    { name = "is_new_vendor", type = "INTEGER", mode = "REQUIRED" },
    { name = "risk", type = "INTEGER", mode = "REQUIRED" },
  ])
}

# One row per /score call (created by hand in week 5, imported).
resource "google_bigquery_table" "predictions" {
  dataset_id          = google_bigquery_dataset.main.dataset_id
  table_id            = "predictions"
  deletion_protection = true
  schema = jsonencode([
    { name = "ts", type = "TIMESTAMP", mode = "NULLABLE" },
    { name = "amount", type = "FLOAT", mode = "NULLABLE" },
    { name = "lines_sum", type = "FLOAT", mode = "NULLABLE" },
    { name = "hour", type = "INTEGER", mode = "NULLABLE" },
    { name = "is_new_vendor", type = "INTEGER", mode = "NULLABLE" },
    { name = "vendor_avg_amount", type = "FLOAT", mode = "NULLABLE" },
    { name = "amount_vs_avg_ratio", type = "FLOAT", mode = "NULLABLE" },
    { name = "risk", type = "INTEGER", mode = "NULLABLE" },
    { name = "score", type = "FLOAT", mode = "NULLABLE" },
    { name = "model_version", type = "STRING", mode = "NULLABLE" },
    { name = "latency_ms", type = "FLOAT", mode = "NULLABLE" },
  ])
}

# One row per /extract call: metadata only (hash, pages, tokens, latency), never content.
resource "google_bigquery_table" "extractions" {
  dataset_id          = google_bigquery_dataset.main.dataset_id
  table_id            = "extractions"
  deletion_protection = true

  time_partitioning {
    type  = "DAY"
    field = "ts"
  }

  schema = jsonencode([
    { name = "ts", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "doc_sha256", type = "STRING", mode = "REQUIRED" },
    { name = "model", type = "STRING", mode = "REQUIRED" },
    { name = "prompt_version", type = "STRING", mode = "REQUIRED" },
    { name = "pages_total", type = "INTEGER", mode = "NULLABLE" },
    { name = "pages_sent", type = "INTEGER", mode = "NULLABLE" },
    { name = "prompt_tokens", type = "INTEGER", mode = "NULLABLE" },
    { name = "output_tokens", type = "INTEGER", mode = "NULLABLE" },
    { name = "thoughts_tokens", type = "INTEGER", mode = "NULLABLE" },
    { name = "llm_latency_ms", type = "FLOAT", mode = "NULLABLE" },
    { name = "total_latency_ms", type = "FLOAT", mode = "NULLABLE" },
    { name = "fields_found", type = "INTEGER", mode = "NULLABLE" },
  ])
}
