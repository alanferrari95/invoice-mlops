# Resources that existed before Terraform covered them (created by hand or by gcloud in
# weeks 3-11). They are imported, never recreated. The two buckets, the work SA and its
# bucket grants were created by Terraform in week 0 and are already in the state.
#
# In a fresh project delete this file: every resource is then created from scratch.

import {
  to = google_artifact_registry_repository.images
  id = "projects/invoice-mlops/locations/europe-west4/repositories/invoice-mlops"
}

import {
  to = google_bigquery_dataset.main
  id = "projects/invoice-mlops/datasets/invoice_mlops"
}

import {
  to = google_bigquery_table.predictions
  id = "projects/invoice-mlops/datasets/invoice_mlops/tables/predictions"
}

import {
  to = google_cloud_run_v2_service.api
  id = "projects/invoice-mlops/locations/europe-west4/services/invoice-mlops-api"
}

import {
  to = google_cloudbuild_trigger.test_pr
  id = "projects/invoice-mlops/locations/europe-west4/triggers/6e58981d-dfac-44c1-ae6c-425f710f9cd8"
}

import {
  to = google_cloudbuild_trigger.deploy_main
  id = "projects/invoice-mlops/locations/europe-west4/triggers/31728d0e-4070-4af8-b826-ec9b4c926be2"
}

import {
  to = google_pubsub_topic.pipeline_alerts
  id = "projects/invoice-mlops/topics/invoice-mlops-pipeline-alerts"
}

import {
  to = google_pubsub_topic.drift_alerts
  id = "projects/invoice-mlops/topics/invoice-mlops-drift-alerts"
}

import {
  to = google_pubsub_subscription.drift_alerts_debug
  id = "projects/invoice-mlops/subscriptions/drift-alerts-debug"
}
