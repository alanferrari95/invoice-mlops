locals {
  project       = var.project_id
  region        = var.region
  image_repo    = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
  work_member   = "serviceAccount:${google_service_account.work.email}"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "logging.googleapis.com",
    "pubsub.googleapis.com",
    "run.googleapis.com",
    "storage.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# --- Storage -------------------------------------------------------------------

# raw/: training data, gold sets.
resource "google_storage_bucket" "raw" {
  name                        = "${var.project_id}-raw"
  location                    = upper(var.region)
  uniform_bucket_level_access = true
  force_destroy               = false
}

# models, pipeline root, compiled pipeline templates.
resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-artifacts"
  location                    = upper(var.region)
  uniform_bucket_level_access = true
  force_destroy               = false
}

resource "google_artifact_registry_repository" "images" {
  repository_id = "invoice-mlops"
  location      = var.region
  format        = "DOCKER"
  description   = "api, train (pipeline) and monitor images"
  # No cleanup policy on purpose: a Cloud Run revision that scales from zero pulls its image
  # again, so deleting an old image would break rollback to that revision.
}

# --- Workload identity -------------------------------------------------------------

# Runs Cloud Run, the Cloud Run Jobs and every Vertex AI pipeline task. No JSON key exists
# for it (or for any other SA here): workloads get short-lived tokens from the metadata server.
resource "google_service_account" "work" {
  account_id   = "invoice-mlops-work"
  display_name = "invoice-mlops work"
  description  = "Work service account, no JSON key"
}

resource "google_storage_bucket_iam_member" "raw_object_admin" {
  bucket = google_storage_bucket.raw.name
  role   = "roles/storage.objectAdmin"
  member = local.work_member
}

resource "google_storage_bucket_iam_member" "artifacts_object_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = local.work_member
}
