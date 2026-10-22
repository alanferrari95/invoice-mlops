-- Training features for the risk model, computed in BigQuery from the raw invoice table.
-- vendor_avg_amount is the vendor's mean amount over its *other* invoices (leave-one-out),
-- so an invoice never contributes to its own baseline. A vendor with a single invoice uses
-- its own amount (ratio = 1). Must stay in sync with build_features.compute_features().
SELECT
  invoice_id,
  amount,
  lines_sum,
  hour,
  is_new_vendor,
  vendor_avg_amount,
  amount / vendor_avg_amount AS amount_vs_avg_ratio,
  risk
FROM (
  SELECT
    *,
    IF(
      COUNT(*) OVER vendor > 1,
      (SUM(amount) OVER vendor - amount) / (COUNT(*) OVER vendor - 1),
      amount
    ) AS vendor_avg_amount
  FROM `{source_table}`
  WINDOW vendor AS (PARTITION BY vendor_id)
)
ORDER BY invoice_id
