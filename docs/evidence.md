# Evidence log

Facts taken from the live project (`invoice-mlops`, `europe-west4`) with read-only commands, appended as work happens. Every number in the README comes from a row here. IDs are copied from command output, not typed by hand.

## State found on 2026-10-22 (before the weeks 4–15 completion work)

| What | Evidence | Command |
|---|---|---|
| Vertex AI Training custom jobs (week 4) | `risk-classifier-custom-job` (2026-09-02), `risk-train-n200-d3-custom-job`, `risk-train-n50-d6-custom-job` (2026-09-03), all `JOB_STATE_SUCCEEDED`; artifacts in `gs://invoice-mlops-artifacts/models/exp-n{100,200,50}-d{4,3,6}/` | `gcloud ai custom-jobs list --region=europe-west4` |
| Vertex AI Experiments | none recorded (the week-4 runs were compared from `metrics.json`, not logged) | `aiplatform.Experiment.list()` |
| Model Registry | `risk-classifier` (id `429981614227849216`), 1 version, alias `default`, artifact `gs://invoice-mlops-artifacts/models/0.2.1-xgb21-bst/` (`model.bst`) | `gcloud ai models list-version 429981614227849216 --region=europe-west4` |
| Pipeline runs | `risk-train-eval` 2026-09-09 SUCCEEDED; `risk-train-eval-week8-fail` 2026-09-10 FAILED (impossible threshold, alert sent); 5 runs from a Vertex schedule on 2026-09-10; staging runs 2026-09-15, 09-25, 09-27 SUCCEEDED | `aiplatform.PipelineJob.list()` |
| Weekly trigger | Vertex AI Pipelines schedule `risk-train-eval-scheduled` (`0 8 * * 1`, Europe/Madrid), **paused**; Cloud Scheduler API not enabled | `aiplatform.PipelineJobSchedule.list()`, `gcloud scheduler jobs list` |
| CI/CD | trigger `test-pr` (PRs to `main`, `cloudbuild-pr.yaml`) and `deploy-main` (push to `main`, `cloudbuild.yaml`), both running as the Compute Engine default SA | `gcloud builds triggers list --region=europe-west4` |
| Red PR on a schema bug (week 9) | build `07f7309b-53ca-4bbd-aa96-ff92483c81a8`, `test-pr`, commit `b2c3cc1`, FAILURE | `gcloud builds list --region=europe-west4` |
| Last green deploy | build `9c4e47a1-7778-4b29-a2e8-4a7cf2cd560d`, `deploy-main`, commit `e637a1e`, SUCCESS → revision `invoice-mlops-api-00011-cdn` | same |
| Cloud Run traffic | 100% on `invoice-mlops-api-00007-7c4` (image `api:8265bec`); revisions 00010 and 00011 deployed with 0% because traffic was pinned by the week-10 rollback | `gcloud run services describe invoice-mlops-api --format='yaml(status.traffic)'` |
| Service identity | Cloud Run runs as `invoice-mlops-work@…`, no user-managed keys | `gcloud iam service-accounts keys list --managed-by=user` → 0 items |
| Least-privilege gap | Compute Engine default SA has `roles/editor` and runs the Cloud Build triggers | `gcloud projects get-iam-policy invoice-mlops` |
| Drift alert plumbing | topic `invoice-mlops-drift-alerts` + subscription `drift-alerts-debug` exist | `gcloud pubsub topics list` |
| Load test (week 10) | see [runbook](runbook.md): p95 0.659 s stable / 0.323 s canary, per revision, 2026-09-18 | Cloud Logging |
