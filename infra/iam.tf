# Least privilege. Four identities, no JSON keys anywhere, no Owner/Editor for workloads:
#
#   invoice-mlops-work       runtime: Cloud Run, Cloud Run Jobs, Vertex pipeline tasks
#   invoice-mlops-build      Cloud Build on main and on model promotion: build, push, deploy
#   invoice-mlops-ci         Cloud Build on pull requests: runs untrusted branch code, so it
#                            can only write build logs
#   invoice-mlops-scheduler  Cloud Scheduler: can only start the two Cloud Run Jobs (monitoring.tf)

data "google_project" "this" {}

# --- work SA ---------------------------------------------------------------------

resource "google_project_iam_member" "work" {
  for_each = toset([
    "roles/aiplatform.user",  # pipelines, Model Registry, Gemini
    "roles/bigquery.jobUser", # run queries (drift job, feature step); data access is per dataset
  ])
  project = var.project_id
  role    = each.value
  member  = local.work_member
}

# Submitting a PipelineJob that runs as the work SA requires actAs on it (launcher, drift job).
resource "google_service_account_iam_member" "work_acts_as_itself" {
  service_account_id = google_service_account.work.name
  role               = "roles/iam.serviceAccountUser"
  member             = local.work_member
}

# --- build SA (main + model-promoted triggers) -------------------------------------

resource "google_service_account" "build" {
  account_id   = "invoice-mlops-build"
  display_name = "invoice-mlops build"
  description  = "Cloud Build for main and model promotions. No key."
}

resource "google_project_iam_member" "build" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/run.developer",   # deploy revisions, move traffic, update jobs (not IAM)
    "roles/aiplatform.user", # submit the staging pipeline, read Registry versions
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.build.email}"
}

resource "google_artifact_registry_repository_iam_member" "build_pushes" {
  repository = google_artifact_registry_repository.images.name
  location   = var.region
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.build.email}"
}

resource "google_storage_bucket_iam_member" "build_publishes" {
  for_each = {
    raw       = google_storage_bucket.raw.name       # gold sets
    artifacts = google_storage_bucket.artifacts.name # pipeline templates
  }
  bucket = each.value
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.build.email}"
}

# Deploy Cloud Run as, and submit pipelines as, the work SA.
resource "google_service_account_iam_member" "build_acts_as_work" {
  service_account_id = google_service_account.work.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.build.email}"
}

# --- CI SA (pull requests) ------------------------------------------------------------

resource "google_service_account" "ci" {
  account_id   = "invoice-mlops-ci"
  display_name = "invoice-mlops CI (pull requests)"
  description  = "Runs tests and compiles pipelines on PRs. Logs only. No key."
}

resource "google_project_iam_member" "ci_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.ci.email}"
}

# --- Default Compute Engine SA ---------------------------------------------------------

# GCP grants it Editor on the whole project, and the original triggers ran as it.
# Nothing uses it any more, so the grant is removed.
resource "google_project_iam_member_remove" "compute_default_editor" {
  project = var.project_id
  role    = "roles/editor"
  member  = "serviceAccount:${data.google_project.this.number}-compute@developer.gserviceaccount.com"

  depends_on = [google_cloudbuild_trigger.deploy_main, google_cloudbuild_trigger.test_pr]
}
