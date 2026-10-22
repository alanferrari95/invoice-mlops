# Cloud Run service. Terraform owns its existence, identity, scaling and public access;
# Cloud Build owns what runs in it (image, pinned model/extractor versions) and the traffic
# split (canary.sh / promote.sh / rollback.sh), so those fields are ignored here.

resource "google_cloud_run_v2_service" "api" {
  name                = "invoice-mlops-api"
  location            = var.region
  deletion_protection = true
  ingress             = "INGRESS_TRAFFIC_ALL"

  # Public demo API without auth: the invoker IAM check is disabled instead of granting
  # roles/run.invoker to allUsers. Everything behind the API runs as the work SA.
  invoker_iam_disabled = true

  template {
    service_account                  = google_service_account.work.email
    max_instance_request_concurrency = 80
    timeout                          = "300s"
    scaling {
      min_instance_count = var.api_min_instances
      max_instance_count = 20 # live value; the cost ceiling is the budget alert
    }
    containers {
      image = "${local.image_repo}/api:latest"
      ports {
        name           = "http1"
        container_port = 8080
      }
      resources {
        limits            = { cpu = "1000m", memory = "512Mi" }
        cpu_idle          = true
        startup_cpu_boost = true
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
      template[0].containers[0].env,
      template[0].revision,
      template[0].labels,
      template[0].annotations,
      traffic,
      client,
      client_version,
      scaling,
    ]
  }
}
