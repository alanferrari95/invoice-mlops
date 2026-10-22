terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

locals {
  project       = "invoice-mlops"
  region        = "europe-west4"
  image_repo    = "${local.region}-docker.pkg.dev/${local.project}/invoice-mlops"
  risk_model_id = "429981614227849216" # Vertex AI Model Registry: risk-classifier
  work_member   = "serviceAccount:${google_service_account.work.email}"
}

provider "google" {
  project = local.project
  region  = local.region
}

resource "google_storage_bucket" "raw" {
  name                        = "invoice-mlops-raw"
  location                    = "EUROPE-WEST4"
  uniform_bucket_level_access = true
  force_destroy               = false
}

resource "google_storage_bucket" "artifacts" {
  name                        = "invoice-mlops-artifacts"
  location                    = "EUROPE-WEST4"
  uniform_bucket_level_access = true
  force_destroy               = false
}

resource "google_service_account" "work" {
  account_id   = "invoice-mlops-work"
  display_name = "invoice-mlops work"
  description  = "Work service account, no JSON key"
}

resource "google_storage_bucket_iam_member" "raw_object_admin" {
  bucket = google_storage_bucket.raw.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.work.email}"
}

resource "google_storage_bucket_iam_member" "artifacts_object_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.work.email}"
}
