# Weeks 8 and 11: continuous training and drift monitoring.
#
#   Cloud Scheduler (daily)  -> Cloud Run Job drift-check -> Pub/Sub drift alert
#                                                         -> launcher -> Vertex pipeline
#   Cloud Scheduler (weekly) -> Cloud Run Job risk-pipeline-launcher -> Vertex pipeline
#
# Both jobs run the monitoring image (Dockerfile.monitor) as the work service account.
# Cloud Build updates their image on every merge, so Terraform ignores the image tag.

locals {
  monitor_image  = "${local.image_repo}/monitor:latest"
  registry_model = "projects/${local.project}/locations/${local.region}/models/${local.risk_model_id}"
}

resource "google_project_service" "scheduler" {
  service            = "cloudscheduler.googleapis.com"
  disable_on_destroy = false
}

# --- Pub/Sub ---------------------------------------------------------------

resource "google_pubsub_topic" "pipeline_alerts" {
  name = "invoice-mlops-pipeline-alerts"
}

resource "google_pubsub_topic" "drift_alerts" {
  name = "invoice-mlops-drift-alerts"
}

# Pull subscription used to read alerts by hand (gcloud pubsub subscriptions pull).
resource "google_pubsub_subscription" "drift_alerts_debug" {
  name                       = "drift-alerts-debug"
  topic                      = google_pubsub_topic.drift_alerts.id
  message_retention_duration = "86400s"
  expiration_policy {
    ttl = "2678400s"
  }
}

resource "google_pubsub_topic" "model_promoted" {
  name = "invoice-mlops-model-promoted"
}

resource "google_pubsub_topic_iam_member" "work_publishes" {
  for_each = {
    pipeline_alerts = google_pubsub_topic.pipeline_alerts.name
    drift_alerts    = google_pubsub_topic.drift_alerts.name
    model_promoted  = google_pubsub_topic.model_promoted.name
  }
  topic  = each.value
  role   = "roles/pubsub.publisher"
  member = local.work_member
}

# --- Cloud Run Jobs ----------------------------------------------------------

resource "google_cloud_run_v2_job" "drift_check" {
  name                = "drift-check"
  location            = local.region
  deletion_protection = false

  template {
    task_count = 1
    template {
      service_account = google_service_account.work.email
      max_retries     = 0
      timeout         = "900s"
      containers {
        image = local.monitor_image
        resources {
          limits = { cpu = "1", memory = "1Gi" }
        }
        env {
          name  = "REGISTRY_MODEL"
          value = local.registry_model
        }
        env {
          name  = "MODEL_ALIAS"
          value = "production"
        }
        env {
          name  = "DRIFT_TOPIC"
          value = google_pubsub_topic.drift_alerts.id
        }
        env {
          name  = "DRIFT_RETRAIN"
          value = "1"
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [template[0].template[0].containers[0].image, client, client_version]
  }
}

resource "google_cloud_run_v2_job" "pipeline_launcher" {
  name                = "risk-pipeline-launcher"
  location            = local.region
  deletion_protection = false

  template {
    task_count = 1
    template {
      service_account = google_service_account.work.email
      max_retries     = 0
      timeout         = "300s"
      containers {
        image   = local.monitor_image
        command = ["python", "-m", "src.launcher"]
        args    = ["--trigger", "schedule"]
        resources {
          limits = { cpu = "1", memory = "512Mi" }
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [template[0].template[0].containers[0].image, client, client_version]
  }
}

# --- Cloud Scheduler -----------------------------------------------------------

resource "google_service_account" "scheduler" {
  account_id   = "invoice-mlops-scheduler"
  display_name = "invoice-mlops scheduler"
  description  = "Cloud Scheduler identity: can only start the two Cloud Run Jobs. No key."
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_runs" {
  for_each = {
    drift    = google_cloud_run_v2_job.drift_check.name
    launcher = google_cloud_run_v2_job.pipeline_launcher.name
  }
  name     = each.value
  location = local.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_cloud_scheduler_job" "drift_daily" {
  name        = "drift-check-daily"
  region      = var.scheduler_region
  description = "Daily drift check on the last 24h of predictions"
  schedule    = "0 6 * * *"
  time_zone   = "Europe/Madrid"

  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/${google_cloud_run_v2_job.drift_check.id}:run"
    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }

  depends_on = [google_project_service.scheduler, google_cloud_run_v2_job_iam_member.scheduler_runs]
}

resource "google_cloud_scheduler_job" "pipeline_weekly" {
  name        = "risk-pipeline-weekly"
  region      = var.scheduler_region
  description = "Weekly continuous-training run of risk-train-eval"
  schedule    = "0 8 * * 1"
  time_zone   = "Europe/Madrid"

  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/${google_cloud_run_v2_job.pipeline_launcher.id}:run"
    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }

  depends_on = [google_project_service.scheduler, google_cloud_run_v2_job_iam_member.scheduler_runs]
}
